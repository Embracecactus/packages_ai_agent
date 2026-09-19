/*
 * Copyright (C) 2026 Xiaomi Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "voice/voice_tts.h"

#include <errno.h>
#include <pthread.h>
#include <string.h>
#include <syslog.h>

#define VOICE_TTS_MAX_BACKENDS 4

static const char* TAG = "voice_tts";

static const voice_tts_ops_t* s_backends[VOICE_TTS_MAX_BACKENDS];
static int s_backend_count;
static const voice_tts_ops_t* s_active;
static pthread_mutex_t s_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned int s_busy;

/* One operation per backend; resource ownership lasts through cleanup. */
static const voice_tts_ops_t* acquire(int (*check)(void *), void *request)
{
    const voice_tts_ops_t* ops = NULL;
    pthread_mutex_lock(&s_lock);
    if (s_busy) {
        errno = EBUSY;
    } else if (!s_active) {
        errno = ENODEV;
    } else {
        ops = s_active;
        int ret = ops->prepare_request ? ops->prepare_request() : 0;
        if (!ret && check) ret = check(request);
        if (ret != 0) {
            errno = ret < 0 ? -ret : EIO;
            ops = NULL;
        } else {
            s_busy++;
        }
    }
    pthread_mutex_unlock(&s_lock);
    return ops;
}

static void release(void)
{
    pthread_mutex_lock(&s_lock);
    s_busy--;
    pthread_mutex_unlock(&s_lock);
}

int voice_tts_register(const voice_tts_ops_t* ops)
{
    if (!ops || !ops->name || (!ops->synthesize && !ops->synthesize_stream)) {
        return -EINVAL;
    }

    pthread_mutex_lock(&s_lock);
    for (int i = 0; i < s_backend_count; i++) {
        if (strcmp(s_backends[i]->name, ops->name) == 0) {
            int ret = s_backends[i] == ops ? 0 : -EEXIST;
            pthread_mutex_unlock(&s_lock);
            return ret;
        }
    }
    if (s_backend_count >= VOICE_TTS_MAX_BACKENDS) {
        pthread_mutex_unlock(&s_lock);
        syslog(LOG_ERR, "[%s] Too many TTS backends\n", TAG);
        return -ENOMEM;
    }

    s_backends[s_backend_count++] = ops;
    if (!s_active) {
        s_active = ops;
        if (ops->init) {
            /* Preserve the existing lazy-config registration contract.
             * Explicit selection below reports activation failures. */
            (void)ops->init();
        }
    }
    pthread_mutex_unlock(&s_lock);
    return 0;
}

int voice_tts_set_backend(const char* name)
{
    if (!name) {
        return -EINVAL;
    }

    pthread_mutex_lock(&s_lock);
    for (int i = 0; i < s_backend_count; i++) {
        if (strcmp(s_backends[i]->name, name) == 0) {
            const voice_tts_ops_t* next = s_backends[i];

            if (s_busy) {
                pthread_mutex_unlock(&s_lock);
                return -EBUSY;
            }

            /* A device model need not coexist with its replacement. */
            if (s_active && s_active->deinit) s_active->deinit();
            s_active = NULL;
            int ret = next->init ? next->init() : 0;
            if (ret != 0) {
                pthread_mutex_unlock(&s_lock);
                return ret;
            }

            s_active = next;

            syslog(LOG_INFO, "[%s] Backend set to: %s\n",
                TAG, name);
            pthread_mutex_unlock(&s_lock);
            return 0;
        }
    }

    syslog(LOG_ERR, "[%s] Backend not found: %s\n", TAG, name);
    pthread_mutex_unlock(&s_lock);
    return -ENOENT;
}

const char* voice_tts_get_backend(void)
{
    const char* name;
    pthread_mutex_lock(&s_lock);
    name = s_active ? s_active->name : NULL;
    pthread_mutex_unlock(&s_lock);
    return name;
}

int voice_tts_speak(const char* text, unsigned char* pcm_out,
    size_t pcm_cap, size_t* pcm_len)
{
    return voice_tts_speak_pcm(text, pcm_out, pcm_cap, pcm_len, NULL);
}

int voice_tts_speak_pcm(const char* text, unsigned char* pcm_out,
    size_t pcm_cap, size_t* pcm_len, voice_tts_capabilities_t* format)
{
    return voice_tts_speak_pcm_checked(text, pcm_out, pcm_cap, pcm_len, format, NULL, NULL);
}

int voice_tts_speak_pcm_checked(const char* text, unsigned char* pcm_out,
    size_t pcm_cap, size_t* pcm_len, voice_tts_capabilities_t* format,
    int (*check)(void *), void *request)
{
    if (!text || !pcm_out || !pcm_len || pcm_cap == 0) return -EINVAL;
    *pcm_len = 0;
    const voice_tts_ops_t* ops = acquire(check, request);
    if (!ops) return -errno;
    voice_tts_capabilities_t caps = {
        .batch_sample_rate = 16000, .channels = 1, .bits = 16,
    };
    int ret = ops->get_capabilities ? ops->get_capabilities(&caps) : 0;
    if (ret == 0 && (!caps.batch_sample_rate || !caps.channels || caps.bits != 16))
        ret = -ENOTSUP;
    if (ret == 0 && !format &&
        (caps.batch_sample_rate != 16000 || caps.channels != 1)) ret = -ENOTSUP;
    if (ret == 0)
        ret = ops->synthesize ?
            ops->synthesize(text, pcm_out, pcm_cap, pcm_len) : -ENOTSUP;
    if (ret == 0 && format) *format = caps;
    release();
    if (ret != 0) *pcm_len = 0;
    return ret;
}

int voice_tts_speak_stream(const char* text, voice_tts_chunk_cb cb,
    void* user_data)
{
    return voice_tts_speak_stream_checked(text, cb, user_data, NULL, NULL);
}

int voice_tts_speak_stream_checked(const char* text, voice_tts_chunk_cb cb,
    void* user_data, int (*check)(void *), void *request)
{
    if (!text || !cb) return -EINVAL;
    const voice_tts_ops_t* ops = acquire(check, request);
    if (!ops) return -errno;
    int ret = ops->synthesize_stream ?
        ops->synthesize_stream(text, cb, user_data) : -ENOTSUP;
    release();
    return ret;
}

int voice_tts_get_capabilities(voice_tts_capabilities_t* caps)
{
    if (!caps) return -EINVAL;
    memset(caps, 0, sizeof(*caps));
    pthread_mutex_lock(&s_lock);
    int ret = !s_active ? -ENODEV : !s_active->get_capabilities ? -ENOTSUP :
        s_active->get_capabilities(caps);
    pthread_mutex_unlock(&s_lock);
    return ret;
}

bool voice_tts_stream_supported(void)
{
    pthread_mutex_lock(&s_lock);
    bool supported = s_active && s_active->synthesize_stream;
    pthread_mutex_unlock(&s_lock);
    return supported;
}

bool voice_tts_is_busy(void)
{
    pthread_mutex_lock(&s_lock);
    bool busy = s_busy != 0;
    pthread_mutex_unlock(&s_lock);
    return busy;
}

int voice_tts_cancel(void)
{
    pthread_mutex_lock(&s_lock);
    const voice_tts_ops_t* ops = s_busy ? s_active : NULL;
    if (ops) s_busy++; /* Pin until the interrupt callback has returned. */
    pthread_mutex_unlock(&s_lock);
    if (!ops) return 0;
    int ret = ops->cancel ? ops->cancel() : -ENOTSUP;
    release();
    return ret;
}
