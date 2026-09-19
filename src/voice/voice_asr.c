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

#include "voice/voice_asr.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>

#define VOICE_ASR_MAX_BACKENDS 4

static const char* TAG = "voice_asr";

static const voice_asr_ops_t* s_backends[VOICE_ASR_MAX_BACKENDS];
static int s_backend_count;
static const voice_asr_ops_t* s_active;
static pthread_mutex_t s_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned int s_busy;

/* One operation per backend; resource ownership lasts through cleanup. */
static const voice_asr_ops_t* acquire(int (*check)(void *), void *request)
{
    const voice_asr_ops_t* ops = NULL;
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

int voice_asr_register(const voice_asr_ops_t* ops)
{
    if (!ops || !ops->name || !ops->recognize) {
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
    if (s_backend_count >= VOICE_ASR_MAX_BACKENDS) {
        pthread_mutex_unlock(&s_lock);
        syslog(LOG_ERR, "[%s] Too many ASR backends\n", TAG);
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

int voice_asr_set_backend(const char* name)
{
    if (!name) {
        return -EINVAL;
    }

    pthread_mutex_lock(&s_lock);
    for (int i = 0; i < s_backend_count; i++) {
        if (strcmp(s_backends[i]->name, name) == 0) {
            const voice_asr_ops_t* next = s_backends[i];

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

const char* voice_asr_get_backend(void)
{
    const char* name;
    pthread_mutex_lock(&s_lock);
    name = s_active ? s_active->name : NULL;
    pthread_mutex_unlock(&s_lock);
    return name;
}

int voice_asr_recognize(const unsigned char* pcm_data,
    size_t pcm_len,
    char* text_out,
    size_t text_cap)
{
    return voice_asr_recognize_checked(pcm_data, pcm_len, text_out, text_cap, NULL, NULL);
}

int voice_asr_recognize_checked(const unsigned char* pcm_data, size_t pcm_len,
    char* text_out, size_t text_cap, int (*check)(void *), void *request)
{
    if (!pcm_data || !pcm_len || (pcm_len & 1) || !text_out || !text_cap)
        return -EINVAL;
    text_out[0] = '\0';
    const voice_asr_ops_t* ops = acquire(check, request);
    if (!ops) return -errno;

    int ret = ops->recognize(pcm_data, pcm_len, text_out, text_cap);
    release();
    return ret;
}

int voice_asr_cancel(void)
{
    pthread_mutex_lock(&s_lock);
    const voice_asr_ops_t* ops = s_busy ? s_active : NULL;
    if (ops) s_busy++;
    pthread_mutex_unlock(&s_lock);
    if (!ops) return 0;
    int ret = ops->cancel ? ops->cancel() : -ENOTSUP;
    release();
    return ret;
}

/* ── Streaming ASR ───────────────────────────────────────────── */

struct voice_asr_stream {
    const voice_asr_ops_t* ops;
    voice_asr_stream_handle_t inner;
};

voice_asr_stream_t* voice_asr_stream_open(void)
{
    const voice_asr_ops_t* ops = acquire(NULL, NULL);
    if (!ops) {
        syslog(LOG_ERR, "[%s] No ASR backend for streaming\n", TAG);
        return NULL;
    }

    if (!ops->stream_open || !ops->stream_send ||
        !ops->stream_finish || !ops->stream_abort) {
        release();
        errno = ENOTSUP;
        return NULL;
    }

    errno = 0;
    voice_asr_stream_handle_t inner = ops->stream_open();

    if (!inner) {
        int error = errno ? errno : EIO;
        release();
        errno = error;
        return NULL;
    }

    voice_asr_stream_t* s = calloc(1, sizeof(*s));

    if (!s) {
        ops->stream_abort(inner);
        release();
        errno = ENOMEM;
        return NULL;
    }

    s->ops = ops;
    s->inner = inner;
    return s;
}

int voice_asr_stream_send(voice_asr_stream_t* s,
    const unsigned char* pcm, size_t len)
{
    if (!s || !s->inner || !pcm || (len & 1)) {
        return -EINVAL;
    }

    return s->ops->stream_send(s->inner, pcm, len);
}

int voice_asr_stream_finish(voice_asr_stream_t* s,
    char* text_out, size_t text_cap)
{
    if (!s) {
        return -EINVAL;
    }

    int ret = s->ops->stream_finish(s->inner, text_out, text_cap);

    free(s);
    release();
    return ret;
}

void voice_asr_stream_abort(voice_asr_stream_t* s)
{
    if (!s) {
        return;
    }

    s->ops->stream_abort(s->inner);
    free(s);
    release();
}

bool voice_asr_stream_supported(void)
{
    pthread_mutex_lock(&s_lock);
    const voice_asr_ops_t* ops = s_active;
    bool supported = ops && ops->stream_open && ops->stream_send &&
        ops->stream_finish && ops->stream_abort;
    pthread_mutex_unlock(&s_lock);
    return supported;
}

bool voice_asr_is_busy(void)
{
    bool busy;
    pthread_mutex_lock(&s_lock);
    busy = s_busy != 0;
    pthread_mutex_unlock(&s_lock);
    return busy;
}
