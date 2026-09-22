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

#include "voice/voice_channel.h"
#ifdef CONFIG_AI_AGENT_AUDIO_PREPROCESS
#include <compexp.h>
#include <echo_canceller.h>
#include <heap_api.h>
#endif
#include "core/message_bus.h"
#include "infra/config_store.h"
#include "agent_compat.h"
#include "agent_config.h"
#include "voice/audio_capture.h"
#include "voice/audio_playback.h"
#include "voice/voice_asr.h"
#include "voice/voice_tts.h"
#include "voice/volc_asr.h"
#include "voice/volc_tts.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "llm/llm_proxy.h"

static const char* TAG = "voice";
#define AUTO_ENDPOINT_SILENCE_MS 900u
#define AUTO_ENDPOINT_MAX_MS 15000u
#define AUTO_ENDPOINT_WAIT_MS 5000u
#define AUTO_TURN_TIMEOUT_MS 180000u
#define AUTO_PCM_BYTES (AUTO_ENDPOINT_MAX_MS * (AGENT_VOICE_SAMPLE_RATE / 1000u) * AGENT_VOICE_CHANNELS * (AGENT_VOICE_BITS / 8u))
#define AUTO_ENDPOINT_MIN_SPEECH_MS 200u
#define AUTO_ENDPOINT_MEAN_ABS 180u
static int s_backends_registered;

/* ── Audio preprocessing (NS + AGC via BES ec2float) ────── */

#ifdef CONFIG_AI_AGENT_AUDIO_PREPROCESS

#define PREPROC_HEAP_SIZE (150 * 1024)
#define PREPROC_FRAME_MS 15
#define PREPROC_FRAME_SIZE \
    (AGENT_VOICE_SAMPLE_RATE / 1000 * PREPROC_FRAME_MS)

static const Ec2FloatConfig s_ns_cfg = {
    .bypass = 0,
    .hpf_enabled = 1,
    .af_enabled = 0, /* no AEC */
    .adprop_enabled = 0,
    .varistep_enabled = 0,
    .nlp_enabled = 0, /* no NLP */
    .clip_enabled = 0,
    .stsupp_enabled = 0,
    .hfsupp_enabled = 0,
    .constrain_enabled = 0,
    .ns_enabled = 1, /* noise suppression ON */
    .cng_enabled = 0,
    .blocks = 1,
    .delay = 0,
    .gamma = 0.9f,
    .echo_band_start = 300,
    .echo_band_end = 1800,
    .min_ovrd = 2,
    .target_supp = -40,
    .highfre_band_start = 4000,
    .highfre_supp = 8.f,
    .noise_supp = -15,
    .cng_type = 0,
    .cng_level = -60,
    .clip_threshold = -20.f,
    .banks = 64,
};

static const CompexpConfig s_agc_cfg = {
    .bypass = 0,
    .type = 0,
    .comp_threshold = -20.f,
    .comp_ratio = 2.f,
    .expand_threshold = -45.f,
    .expand_ratio = 0.5556f,
    .attack_time = 0.001f,
    .release_time = 0.006f,
    .makeup_gain = 6,
    .delay = 32,
    .tav = 0.2f,
};

typedef struct {
    void* heap;
    Ec2FloatState* ns;
    CompexpState* agc;
} audio_preproc_t;

static audio_preproc_t* preproc_create(void)
{
    audio_preproc_t* pp = calloc(1, sizeof(*pp));
    if (!pp) {
        return NULL;
    }

    pp->heap = malloc(PREPROC_HEAP_SIZE);
    if (!pp->heap) {
        free(pp);
        return NULL;
    }
    med_heap_init(pp->heap, PREPROC_HEAP_SIZE);

    pp->ns = ec2float_create(AGENT_VOICE_SAMPLE_RATE,
        PREPROC_FRAME_SIZE, 0, &s_ns_cfg);
    pp->agc = compexp_create(AGENT_VOICE_SAMPLE_RATE,
        PREPROC_FRAME_SIZE, &s_agc_cfg);

    syslog(LOG_INFO, "[%s] preproc: NS=%p AGC=%p heap=%dKB\n",
        TAG, (void*)pp->ns, (void*)pp->agc,
        PREPROC_HEAP_SIZE / 1024);
    return pp;
}

static void preproc_destroy(audio_preproc_t* pp)
{
    if (!pp) {
        return;
    }
    if (pp->agc) {
        compexp_destroy(pp->agc);
    }
    if (pp->ns) {
        ec2float_destroy(pp->ns);
    }
    free(pp->heap);
    free(pp);
}

static void preproc_chunk(audio_preproc_t* pp,
    unsigned char* buf, int bytes)
{
    if (!pp || (!pp->ns && !pp->agc)) {
        return;
    }

    int16_t* samples = (int16_t*)buf;
    int total = bytes / 2;
    int fsz = PREPROC_FRAME_SIZE;
    int32_t in32[PREPROC_FRAME_SIZE];
    int32_t ref32[PREPROC_FRAME_SIZE];
    int32_t out32[PREPROC_FRAME_SIZE];

    memset(ref32, 0, sizeof(ref32));

    for (int off = 0; off < total; off += fsz) {
        int n = (off + fsz <= total) ? fsz : (total - off);

        for (int i = 0; i < n; i++) {
            in32[i] = samples[off + i];
        }

        if (pp->ns) {
            ec2float_process(pp->ns, in32, ref32, n, out32);
        } else {
            memcpy(out32, in32, n * sizeof(int32_t));
        }

        if (pp->agc) {
            for (int i = 0; i < n; i++) {
                out32[i] <<= 8;
            }
            compexp_process_int24(pp->agc, out32, n);
            for (int i = 0; i < n; i++) {
                out32[i] >>= 8;
            }
        }

        for (int i = 0; i < n; i++) {
            int32_t v = out32[i];
            if (v > 32767) {
                v = 32767;
            } else if (v < -32768) {
                v = -32768;
            }
            samples[off + i] = (int16_t)v;
        }
    }
}

#endif /* CONFIG_AI_AGENT_AUDIO_PREPROCESS */

/* ── Recording state machine (PTT mode) ─────────────────── */

enum voice_state {
    VOICE_IDLE = 0,
    VOICE_RECORDING,
    VOICE_STOPPING,
    VOICE_STARTING,
    VOICE_PROCESSING,
    VOICE_SPEAKING
};

static struct {
    enum voice_state state;
    pthread_mutex_t lock;
    audio_capture_t* cap;
    unsigned char* pcm_buf; /* accumulated PCM data */
    size_t pcm_len; /* bytes recorded so far */
    size_t pcm_cap; /* buffer capacity */
    pthread_t rec_thread;
    voice_asr_stream_t* asr_stream; /* streaming ASR handle */
    sem_t rec_ready; /* posted by recording_thread when ready to consume */
    int capture_error;
    atomic_int tts_abort; /* set to 1 to stop active TTS playback */
    int tts_active;
    audio_playback_t* tts_pb; /* active TTS playback handle (or NULL) */
    pthread_mutex_t speak_lock; /* serialize concurrent speak calls */
    int auto_endpoint;
    sem_t rec_done;
    int initialized;
    int turn_active;
    int canceled;
    int cleanup_in_progress;
    int capture_cleanup_pending;
    int capture_cleanup_result;
    int tts_cleanup_pending;
    int tts_cleanup_result;
    uint64_t request_id;
    uint64_t turn_started_ms;
    voice_channel_event_cb event_cb;
} s_voice = {
    .state = VOICE_IDLE,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .speak_lock = PTHREAD_MUTEX_INITIALIZER,
};

/* ── File-based helpers (kept for QEMU test commands) ──── */

static int read_pcm_file(const char* path,
    unsigned char** out, size_t* out_len)
{
    int fd = open(path, O_RDONLY);

    if (fd < 0) {
        syslog(LOG_ERR, "[%s] Cannot open %s: %d\n",
            TAG, path, errno);
        return -errno;
    }

    off_t fsize = lseek(fd, 0, SEEK_END);

    if (fsize <= 0 || (size_t)fsize > AGENT_VOICE_PCM_BUF_SIZE) {
        syslog(LOG_ERR, "[%s] File too large or empty: %ld\n",
            TAG, (long)fsize);
        close(fd);
        return -EFBIG;
    }

    lseek(fd, 0, SEEK_SET);

    unsigned char* buf = malloc((size_t)fsize);

    if (!buf) {
        close(fd);
        return -ENOMEM;
    }

    ssize_t nread = read(fd, buf, (size_t)fsize);

    close(fd);

    if (nread != (ssize_t)fsize) {
        free(buf);
        return -EIO;
    }

    *out = buf;
    *out_len = (size_t)fsize;
    return 0;
}

static int write_pcm_file(const char* path,
    const unsigned char* data, size_t len)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);

    if (fd < 0) {
        syslog(LOG_ERR, "[%s] Cannot create %s: %d\n",
            TAG, path, errno);
        return -errno;
    }

    ssize_t nw = write(fd, data, len);

    close(fd);
    return (nw == (ssize_t)len) ? 0 : -EIO;
}

/* ── Recording cleanup helper ────────────────────────────── */

/* Allocate fallback PCM buffer (must hold lock) */
static int alloc_fallback_buffer_locked(void)
{
    if (s_voice.pcm_buf) {
        return 0;
    }

    s_voice.pcm_buf = malloc(s_voice.pcm_cap);
    if (!s_voice.pcm_buf) {
        syslog(LOG_ERR, "[%s] fallback buffer alloc failed\n", TAG);
        return -ENOMEM;
    }

    s_voice.pcm_len = 0;
    return 0;
}

/* Accumulate PCM chunk to fallback buffer (must hold lock) */
static int accumulate_pcm_locked(const unsigned char* chunk, size_t len)
{
    if (s_voice.pcm_buf && s_voice.pcm_len + len <= s_voice.pcm_cap) {
        memcpy(s_voice.pcm_buf + s_voice.pcm_len, chunk, len);
        s_voice.pcm_len += len;
        return 0;
    }
    return -ENOSPC;
}

/* Process one audio chunk: send to ASR or accumulate for fallback */
static int process_audio_chunk(voice_asr_stream_t** stream_ptr,
    const unsigned char* chunk, size_t len, int* need_fallback,
    size_t* bytes_sent)
{
    if (*need_fallback) {
        pthread_mutex_lock(&s_voice.lock);
        int ret = accumulate_pcm_locked(chunk, len);
        pthread_mutex_unlock(&s_voice.lock);
        if (ret < 0) return ret;
    }

    if (*stream_ptr) {
        int ret = voice_asr_stream_send(*stream_ptr, chunk, len);

        if (ret != 0) {
            syslog(LOG_ERR,
                "[%s] stream send failed: %d, recording rejected\n", TAG, ret);
            voice_asr_stream_abort(*stream_ptr);
            *stream_ptr = NULL;
            /* Earlier streaming PCM was not retained. Never retry a tail
             * as a complete recording, even against the same backend. */
            return ret;
        } else {
            *bytes_sent += len;
        }
    }

    return 0;
}

/* ── Recording thread (PTT mode — streaming ASR) ─────────── */

static void notify_channel_event(int event, int result)
{
    voice_channel_event_cb cb;
    pthread_mutex_lock(&s_voice.lock);
    cb = s_voice.event_cb;
    pthread_mutex_unlock(&s_voice.lock);
    if (cb) cb(event, result);
}

static uint64_t voice_now_ms(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000u + now.tv_nsec / 1000000u;
}

void voice_channel_service_ready(int result)
{
    notify_channel_event(VOICE_CHANNEL_EVENT_SERVICE_READY, result);
}

static int voice_request_status(uint64_t id)
{
    pthread_mutex_lock(&s_voice.lock);
    int ret = !s_voice.turn_active || id != s_voice.request_id ? -ESTALE :
        s_voice.canceled ? -ECANCELED :
        voice_now_ms() - s_voice.turn_started_ms >= AUTO_TURN_TIMEOUT_MS ?
        -ETIMEDOUT : 0;
    pthread_mutex_unlock(&s_voice.lock);
    return ret;
}

static int voice_request_check(void *request)
{ return voice_request_status(*(const uint64_t *)request); }

static void voice_request_complete(uint64_t id, int result)
{
    pthread_mutex_lock(&s_voice.lock);
    if (!s_voice.turn_active || id != s_voice.request_id) {
        pthread_mutex_unlock(&s_voice.lock);
        return;
    }
    /* Completion is called only by the current resource owner after cleanup. */
    if (s_voice.cap || s_voice.tts_active ||
        s_voice.capture_cleanup_pending || s_voice.tts_cleanup_pending ||
        s_voice.state == VOICE_STOPPING) {
        pthread_mutex_unlock(&s_voice.lock);
        return;
    }
    if (s_voice.canceled) result = -ECANCELED;
    s_voice.turn_active = 0;
    s_voice.auto_endpoint = 0;
    s_voice.state = VOICE_IDLE;
    pthread_mutex_unlock(&s_voice.lock);
    syslog(LOG_INFO, "[%s] request=%llu complete=%d\n", TAG,
        (unsigned long long)id, result);
    notify_channel_event(VOICE_CHANNEL_EVENT_TURN_COMPLETE, result);
}

static int voice_channel_start_internal(int automatic);

static void* auto_finalize_task(void* arg)
{
    (void)arg;
    pthread_mutex_lock(&s_voice.lock);
    uint64_t id = s_voice.request_id;
    pthread_mutex_unlock(&s_voice.lock);
    int ret = voice_channel_start_internal(1);
    char text[512] = { 0 };
    if (ret == 0) {
        int waited;
        do { waited = sem_wait(&s_voice.rec_done); } while (waited < 0 && errno == EINTR);
        if (waited < 0) voice_channel_cancel();
        ret = voice_channel_stop_with_text(text, sizeof(text));
    }
    if (ret == 0) ret = voice_request_status(id);
    if (ret == 0) {
        agent_msg_t msg = { 0 };
        strncpy(msg.channel, AGENT_CHAN_VOICE, sizeof(msg.channel) - 1);
        strncpy(msg.chat_id, "voice", sizeof(msg.chat_id) - 1);
        msg.request_id = id;
        msg.request_status = voice_request_status;
        msg.request_complete = voice_request_complete;
        msg.content = strdup(text);
        ret = msg.content ? message_bus_push_inbound(&msg) : -ENOMEM;
        if (ret != 0) free(msg.content);
    }
    memset(text, 0, sizeof(text));
    if (ret != 0) voice_request_complete(id, ret);
    return NULL;
}

static void* recording_thread(void* arg)
{
    (void)arg;
    unsigned char chunk[AGENT_ASR_CHUNK_SIZE];
    int need_fallback = 0;
    // int dump_fd = -1;
    size_t total_bytes_read = 0;
    size_t total_bytes_sent = 0;
    int chunk_count = 0;
    int abnormal_exit = 0;
    unsigned int recorded_ms = 0, speech_ms = 0, silence_ms = 0;
    uint64_t started_ms = voice_now_ms();
    uint64_t last_data_ms = started_ms;

    syslog(LOG_INFO, "[%s] recording worker started\n", TAG);

    /* Mode is selected before capture. A batch-only backend never attempts
     * a streaming connection, and an in-flight failure is terminal. */
    pthread_mutex_lock(&s_voice.lock);
    voice_asr_stream_t* stream = s_voice.asr_stream;
    s_voice.asr_stream = NULL; /* thread now owns it */
    pthread_mutex_unlock(&s_voice.lock);

    if (!stream) {
        syslog(LOG_INFO, "[%s] ASR mode=batch\n", TAG);
        need_fallback = 1;

        pthread_mutex_lock(&s_voice.lock);
        int allocation = alloc_fallback_buffer_locked();
        if (allocation != 0) {
            s_voice.capture_error = allocation;
            pthread_mutex_unlock(&s_voice.lock);
            sem_post(&s_voice.rec_ready);
            sem_post(&s_voice.rec_done);
            return NULL;
        }
        pthread_mutex_unlock(&s_voice.lock);
    } else {
        syslog(LOG_INFO, "[%s] using pre-connected ASR stream\n", TAG);
    }

    /* Signal voice_channel_start() that we are ready to consume
     * audio frames. Without this, sem_wait in start() deadlocks. */
    sem_post(&s_voice.rec_ready);

    /* No flush needed: the new start sequence guarantees that
     * audio_capture_start() runs AFTER this thread is ready,
     * so there is no stale data in the queue. The old flush
     * was harmful — audio_capture_read() blocks, so it would
     * consume real audio frames instead of stale ones. */

    while (1) {
        pthread_mutex_lock(&s_voice.lock);
        enum voice_state st = s_voice.state;
        int canceled = s_voice.canceled;
        pthread_mutex_unlock(&s_voice.lock);

        if (st != VOICE_RECORDING || canceled) {
            if (canceled) {
                pthread_mutex_lock(&s_voice.lock);
                s_voice.capture_error = -ECANCELED;
                pthread_mutex_unlock(&s_voice.lock);
            }
            break;
        }
        if (s_voice.auto_endpoint &&
            ((speech_ms < AUTO_ENDPOINT_MIN_SPEECH_MS &&
              voice_now_ms() - started_ms >= AUTO_ENDPOINT_WAIT_MS) ||
             voice_now_ms() - last_data_ms >= 2000u)) {
            pthread_mutex_lock(&s_voice.lock);
            s_voice.capture_error = speech_ms >= AUTO_ENDPOINT_MIN_SPEECH_MS ?
                -ETIMEDOUT : -ENODATA;
            pthread_mutex_unlock(&s_voice.lock);
            break;
        }

        int n = audio_capture_read(s_voice.cap,
            chunk, sizeof(chunk));

        if (n == -EAGAIN || n == -EWOULDBLOCK) {
            /* Non-blocking capture socket has no data yet (common right
             * after start before the first frames arrive). Retry instead
             * of treating it as a fatal error. */
            usleep(10000);
            continue;
        }

        if (n <= 0) {
            syslog(LOG_WARNING,
                "[%s] capture read returned %d, stopping\n", TAG, n);
            abnormal_exit = 1;
            break;
        }

        last_data_ms = voice_now_ms();
        if ((size_t)n % (AGENT_VOICE_CHANNELS * sizeof(int16_t))) {
            pthread_mutex_lock(&s_voice.lock);
            s_voice.capture_error = -EPROTO;
            pthread_mutex_unlock(&s_voice.lock);
            break;
        }
        total_bytes_read += (size_t)n;
        chunk_count++;

        /* Dump raw PCM to file */
        // if (dump_fd >= 0) {
        //     write(dump_fd, chunk, (size_t)n);
        // }

        /* Log audio level for first few chunks and periodically
         * to diagnose silence vs actual speech from QEMU mic. */
        if (chunk_count <= 3 || chunk_count % 50 == 0) {
            int16_t* samples = (int16_t*)chunk;
            int sample_count = n / 2;
            int32_t peak = 0;

            for (int i = 0; i < sample_count; i++) {
                int32_t v = samples[i] < 0 ? -samples[i] : samples[i];
                if (v > peak) {
                    peak = v;
                }
            }
            syslog(LOG_INFO,
                "[%s] chunk#%d: %d bytes, peak=%ld\n",
                TAG, chunk_count, n, (long)peak);
        }

#ifdef CONFIG_AI_AGENT_AUDIO_PREPROCESS
        preproc_chunk(pp, chunk, n);
#endif

        int endpoint = 0;
        if (s_voice.auto_endpoint) {
            uint64_t sum = 0;
            int16_t *samples = (int16_t *)chunk;
            unsigned int samples_count = n / sizeof(int16_t);
            for (unsigned int i = 0; i < samples_count; i++) {
                int32_t v = samples[i];
                sum += v < 0 ? -v : v;
            }
            unsigned int chunk_ms = samples_count * 1000u / AGENT_VOICE_SAMPLE_RATE;
            unsigned int mean = samples_count ? sum / samples_count : 0;
            /* Waiting silence consumes time, not upload memory. */
            if (!speech_ms && mean < AUTO_ENDPOINT_MEAN_ABS) continue;
            recorded_ms += chunk_ms;
            if (mean >= AUTO_ENDPOINT_MEAN_ABS) { speech_ms += chunk_ms; silence_ms = 0; }
            else silence_ms += chunk_ms;
            endpoint = speech_ms >= AUTO_ENDPOINT_MIN_SPEECH_MS &&
                silence_ms >= AUTO_ENDPOINT_SILENCE_MS;
            if (recorded_ms > AUTO_ENDPOINT_MAX_MS ||
                (recorded_ms == AUTO_ENDPOINT_MAX_MS && !endpoint)) {
                pthread_mutex_lock(&s_voice.lock);
                s_voice.capture_error = -EFBIG;
                pthread_mutex_unlock(&s_voice.lock);
                break;
            }
        }
        int process_ret = process_audio_chunk(&stream, chunk, (size_t)n,
            &need_fallback, &total_bytes_sent);
        if (process_ret < 0) {
            pthread_mutex_lock(&s_voice.lock);
            s_voice.capture_error = process_ret;
            pthread_mutex_unlock(&s_voice.lock);
            abnormal_exit = 1;
            break;
        }
        if (endpoint) break;
    }

#ifdef CONFIG_AI_AGENT_AUDIO_PREPROCESS
    preproc_destroy(pp);
#endif

    /* Store stream handle for voice_channel_stop to finish */
    // if (dump_fd >= 0) {
    //     close(dump_fd);
    //     syslog(LOG_INFO, "[%s] PCM dump saved to /data/cap_dump.pcm\n", TAG);
    // }

    pthread_mutex_lock(&s_voice.lock);
    s_voice.asr_stream = stream;

    if (abnormal_exit && s_voice.state == VOICE_RECORDING) {
        if (!s_voice.capture_error) s_voice.capture_error = -EIO;
        /* The controller joins this worker before closing its capture. */
    }
    pthread_mutex_unlock(&s_voice.lock);

    syslog(LOG_INFO,
        "[%s] recording thread exit: %d chunks, %zu read, %zu sent%s\n",
        TAG, chunk_count, total_bytes_read, total_bytes_sent,
        abnormal_exit ? " (abnormal)" : "");
    sem_post(&s_voice.rec_done);
    return NULL;
}

/* ── ASR worker thread ──────────────────────────────────── */

#define ASR_THREAD_STACK (24 * 1024)

/* ── TTS streaming callback ──────────────────────────────── */

static struct timespec s_tts_start;
static int s_tts_first_chunk;
static struct timespec s_asr_done_ts;

typedef struct {
    audio_playback_t* pb;
    int error;
    int terminal;
    int batch;
    size_t frame_bytes;
    unsigned char partial[4];
    size_t partial_len;
    size_t written;
    voice_tts_capabilities_t format;
} tts_output_t;

static void tts_stream_cb(const unsigned char* pcm_data,
    size_t pcm_len, int is_last, void* user_data)
{
    tts_output_t* out = user_data;
    if (out->error) return;
    if (s_voice.tts_abort) {
        out->error = -ECANCELED;
    } else if (out->terminal || (!pcm_data && pcm_len)) {
        out->error = -EPROTO;
    }
    if (out->error) {
        return;
    }
    if (pcm_len > 0) {
        if (!out->pb) {
            voice_tts_capabilities_t caps = out->format;
            if (out->batch) caps.sample_rate = caps.batch_sample_rate;
            /* The synchronous streaming operation pins its backend while
             * invoking us. Querying here binds Media to that exact format;
             * a pre-request capability query cannot provide that guarantee. */
            int ret = out->batch ? 0 : voice_tts_get_capabilities(&caps);
            if (ret != 0 || caps.sample_rate == 0 || caps.bits != 16 ||
                (caps.channels != 1 && caps.channels != 2)) {
                out->error = ret ? ret : -ENOTSUP;
                return;
            }
            out->frame_bytes = caps.channels * caps.bits / 8;
            out->pb = audio_playback_open(AGENT_AUDIO_PLAYBACK_DEV,
                caps.sample_rate, caps.channels, caps.bits);
            if (!out->pb) {
                out->error = errno ? -errno : -EIO;
                return;
            }
            pthread_mutex_lock(&s_voice.lock);
            s_voice.tts_pb = out->pb;
            if (s_voice.tts_abort) audio_playback_stop(out->pb);
            pthread_mutex_unlock(&s_voice.lock);
        }
        /* 网络分片不保证 PCM 帧对齐；只在真正结束时拒绝残帧。 */
        if (out->partial_len) {
            size_t n = out->frame_bytes - out->partial_len;
            if (n > pcm_len) n = pcm_len;
            memcpy(out->partial + out->partial_len, pcm_data, n);
            out->partial_len += n;
            pcm_data += n;
            pcm_len -= n;
            if (out->partial_len == out->frame_bytes) {
                int written = audio_playback_write(out->pb, out->partial,
                    out->frame_bytes);
                if (written != (int)out->frame_bytes)
                    out->error = written < 0 ? written : -EIO;
                else out->written += (size_t)written;
                out->partial_len = 0;
            }
        }
        size_t aligned = pcm_len - pcm_len % out->frame_bytes;
        if (!out->error && aligned) {
            int written = audio_playback_write(out->pb, pcm_data, aligned);
            if (written < 0 || (size_t)written != aligned)
                out->error = written < 0 ? written : -EIO;
            else out->written += (size_t)written;
        }
        if (!out->error && pcm_len > aligned) {
            out->partial_len = pcm_len - aligned;
            memcpy(out->partial, pcm_data + aligned, out->partial_len);
        }
        if (out->error) {
            return;
        }
        if (!s_tts_first_chunk && out->written) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            long ms = (now.tv_sec - s_tts_start.tv_sec) * 1000
                + (now.tv_nsec - s_tts_start.tv_nsec) / 1000000;
            syslog(LOG_INFO, "[%s] TTS first Media write complete: %ldms (not acoustic)\n", TAG, ms);
            s_tts_first_chunk = 1;
        }
    }
    if (is_last && out->partial_len) {
        out->error = -EPROTO;
    }
    out->terminal = is_last != 0;
}

/* 接收线程只向有界队列提交 PCM，唯一消费者继续使用同一个 Media 会话。
 * 预缓冲和欠载后恢复使用相同水位；不插静音，也不等整段下载完成。 */
#define TTS_QUEUE_BYTES (64 * 1024)
#define TTS_QUEUE_PREFILL 8192
#define TTS_QUEUE_SLICE 2048

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t changed;
    pthread_t thread;
    unsigned char pcm[TTS_QUEUE_BYTES];
    size_t head;
    size_t size;
    size_t peak;
    size_t received;
    int64_t last_pcm_ms;
    int64_t max_gap_ms;
    unsigned int rebuffer;
    int done;
    int terminal;
    int format_ready;
    atomic_int error;
    tts_output_t* output;
} tts_queue_t;

static void tts_queue_wait(tts_queue_t* q)
{
    /* 有界等待让没有后续网络回调的取消也能退出。 */
    struct timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_nsec += 100000000L;
    if (until.tv_nsec >= 1000000000L) {
        until.tv_sec++;
        until.tv_nsec -= 1000000000L;
    }
    pthread_cond_timedwait(&q->changed, &q->lock, &until);
}

static void* tts_queue_play(void* arg)
{
    tts_queue_t* q = arg;
    unsigned char pcm[TTS_QUEUE_SLICE];
    int buffering = 1;
    int started = 0;
    for (;;) {
        pthread_mutex_lock(&q->lock);
        while (!q->done && !atomic_load(&q->error) &&
            !atomic_load(&s_voice.tts_abort) &&
            (!q->size || (buffering && q->size < TTS_QUEUE_PREFILL)))
            tts_queue_wait(q);
        if (atomic_load(&s_voice.tts_abort))
            atomic_store(&q->error, -ECANCELED);
        if (atomic_load(&q->error) || (q->done && !q->size)) {
            int terminal = q->terminal;
            pthread_mutex_unlock(&q->lock);
            if (!atomic_load(&q->error) && terminal)
                tts_stream_cb(NULL, 0, 1, q->output);
            break;
        }
        size_t n = q->size < sizeof(pcm) ? q->size : sizeof(pcm);
        size_t first = TTS_QUEUE_BYTES - q->head;
        if (first > n) first = n;
        memcpy(pcm, q->pcm + q->head, first);
        memcpy(pcm + first, q->pcm, n - first);
        q->head = (q->head + n) % TTS_QUEUE_BYTES;
        q->size -= n;
        buffering = 0;
        started = 1;
        pthread_cond_broadcast(&q->changed);
        pthread_mutex_unlock(&q->lock);

        tts_stream_cb(pcm, n, 0, q->output);
        pthread_mutex_lock(&q->lock);
        if (q->output->error)
            atomic_store(&q->error, q->output->error);
        if (!q->size && !q->done && started && !q->output->error) {
            buffering = 1;
            q->rebuffer++;
        }
        pthread_cond_broadcast(&q->changed);
        pthread_mutex_unlock(&q->lock);
    }
    return NULL;
}

static void tts_queue_receive(const unsigned char* pcm, size_t len,
    int is_last, void* context)
{
    tts_queue_t* q = context;
    pthread_mutex_lock(&q->lock);
    if (q->terminal || (!pcm && len))
        atomic_store(&q->error, -EPROTO);
    if (len && !q->format_ready && !atomic_load(&q->error)) {
        /* 在接收回调内查询仍被当前请求固定的后端，不能由异步播放器
         * 在请求结束后再次查询可能已经切换的默认后端。 */
        voice_tts_capabilities_t caps = {0};
        int ret = voice_tts_get_capabilities(&caps);
        if (ret || !caps.sample_rate || caps.bits != 16 ||
            (caps.channels != 1 && caps.channels != 2)) {
            atomic_store(&q->error, ret ? ret : -ENOTSUP);
        } else {
            caps.batch_sample_rate = caps.sample_rate;
            q->output->format = caps;
            q->output->batch = 1;
            q->format_ready = 1;
        }
    }
    if (len && !atomic_load(&q->error)) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        int64_t ms = (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
        if (!q->received) {
            int64_t begin = (int64_t)s_tts_start.tv_sec * 1000 +
                s_tts_start.tv_nsec / 1000000;
            syslog(LOG_INFO, "[%s] TTS first PCM received: %ldms\n",
                TAG, (long)(ms - begin));
        } else if (ms - q->last_pcm_ms > q->max_gap_ms) {
            q->max_gap_ms = ms - q->last_pcm_ms;
        }
        q->last_pcm_ms = ms;
    }
    while (len && !atomic_load(&q->error)) {
        if (atomic_load(&s_voice.tts_abort)) {
            atomic_store(&q->error, -ECANCELED);
            break;
        }
        if (q->size == TTS_QUEUE_BYTES) {
            tts_queue_wait(q);
            continue;
        }
        size_t tail = (q->head + q->size) % TTS_QUEUE_BYTES;
        size_t n = TTS_QUEUE_BYTES - tail;
        if (n > TTS_QUEUE_BYTES - q->size) n = TTS_QUEUE_BYTES - q->size;
        if (n > len) n = len;
        memcpy(q->pcm + tail, pcm, n);
        q->size += n;
        q->received += n;
        if (q->size > q->peak) q->peak = q->size;
        pcm += n;
        len -= n;
        pthread_cond_broadcast(&q->changed);
    }
    if (is_last) q->terminal = 1;
    pthread_cond_broadcast(&q->changed);
    int error = atomic_load(&q->error);
    pthread_mutex_unlock(&q->lock);
    if (error) voice_tts_cancel();
}

static int tts_speak_queued(const char* text, tts_output_t* output,
    int tracked, uint64_t* id)
{
    tts_queue_t* q = calloc(1, sizeof(*q));
    if (!q) return -ENOMEM;
    int ret = pthread_mutex_init(&q->lock, NULL);
    if (ret) { free(q); return -ret; }
    ret = pthread_cond_init(&q->changed, NULL);
    if (ret) {
        pthread_mutex_destroy(&q->lock);
        free(q);
        return -ret;
    }
    q->output = output;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    ret = pthread_attr_setstacksize(&attr, 16 * 1024);
    if (!ret) ret = pthread_create(&q->thread, &attr, tts_queue_play, q);
    pthread_attr_destroy(&attr);
    if (!ret) {
        ret = voice_tts_speak_stream_checked(text, tts_queue_receive, q,
            tracked ? voice_request_check : NULL, id);
        pthread_mutex_lock(&q->lock);
        q->done = 1;
        pthread_cond_broadcast(&q->changed);
        pthread_mutex_unlock(&q->lock);
        /* 回收消费者后才允许关闭 Media，取消不能留下悬空的队列或句柄。 */
        pthread_join(q->thread, NULL);
        if (atomic_load(&q->error)) ret = atomic_load(&q->error);
        syslog(LOG_INFO, "[%s] TTS queue: pcm=%zu peak=%zu rebuffer=%u max_receive_gap_ms=%ld\n",
            TAG, q->received, q->peak, q->rebuffer, (long)q->max_gap_ms);
    } else {
        ret = -ret;
    }
    pthread_cond_destroy(&q->changed);
    pthread_mutex_destroy(&q->lock);
    free(q);
    return ret;
}

/* ── Public API ──────────────────────────────────────────── */

int voice_channel_init(void)
{
    syslog(LOG_INFO, "[%s] Voice channel initializing\n", TAG);

    /* Reset state on init — static vars may persist across process restarts
     * in NuttX depending on how the binary is loaded. */
    s_voice.state = VOICE_IDLE;
    s_voice.cap = NULL;
    s_voice.pcm_buf = NULL;
    s_voice.pcm_len = 0;
    s_voice.asr_stream = NULL;
    s_voice.capture_error = 0;
    s_voice.tts_abort = 0;
    s_voice.tts_active = 0;
    s_voice.tts_pb = NULL;
    s_voice.auto_endpoint = 0;
    s_voice.turn_active = 0;
    s_voice.canceled = 0;
    s_voice.turn_started_ms = 0;
    s_voice.cleanup_in_progress = 0;
    s_voice.capture_cleanup_pending = 0;
    s_voice.capture_cleanup_result = 0;
    s_voice.tts_cleanup_pending = 0;
    s_voice.tts_cleanup_result = 0;
    sem_init(&s_voice.rec_ready, 0, 0);
    sem_init(&s_voice.rec_done, 0, 0);

    if (!s_backends_registered) {
        int ret = volc_tts_register();
        if (ret != 0) return ret;
        ret = volc_asr_register();
        if (ret != 0) return ret;
        s_backends_registered = 1;
    }
    char backend[64];
    if (claw_config_get(AGENT_CFG_KEY_TTS_BACKEND, backend, sizeof(backend)) == 0 &&
        backend[0]) {
        int ret = voice_tts_set_backend(backend);
        /* A protected service backend may be registered before its product
         * configuration owner has restored credentials and trust. Defer only
         * that explicit missing-key state; all other activation errors fail. */
        if (ret != 0 && ret != -ENOKEY) return ret;
        if (ret == -ENOKEY)
            syslog(LOG_INFO, "[%s] TTS backend %s awaiting protected config\n",
                TAG, backend);
    }

    pthread_mutex_lock(&s_voice.lock);
    s_voice.initialized = 1;
    pthread_mutex_unlock(&s_voice.lock);
    notify_channel_event(VOICE_CHANNEL_EVENT_INITIALIZED, 0);
    return 0;
}

static int voice_channel_start_internal(int automatic)
{
    pthread_mutex_lock(&s_voice.lock);
    if (!s_voice.initialized || (automatic ? s_voice.state != VOICE_STARTING :
        s_voice.state != VOICE_IDLE || s_voice.turn_active)) {
        pthread_mutex_unlock(&s_voice.lock);
        syslog(LOG_WARNING, "[%s] already recording\n", TAG);
        return -EBUSY;
    }

    /* Drain stale sem posts from previous sessions to prevent
     * sem_wait returning immediately before thread is ready. */
    while (sem_trywait(&s_voice.rec_ready) == 0)
        ;

    /* ── AEC workaround: stop active TTS to prevent echo ── */
    if (s_voice.tts_active) {
        syslog(LOG_INFO,
            "[%s] PTT: stopping active TTS to prevent echo\n", TAG);
        s_voice.tts_abort = 1;
        if (s_voice.tts_pb) audio_playback_stop(s_voice.tts_pb);
        int canceled = voice_tts_cancel();
        if (canceled < 0)
            syslog(LOG_WARNING, "[%s] TTS cancel unavailable: %d\n", TAG, canceled);

        /* Release lock so the speak thread can finish closing the
         * playback device.  Without this, we hold the lock through
         * the entire ASR pre-connect (300-800ms), blocking the speak
         * thread from calling audio_playback_close().  The playback
         * device stays open while we open the capture device, and on
         * BES hardware the shared codec (ci2s) gets confused, causing
         * incomplete or corrupted recording. */
        pthread_mutex_unlock(&s_voice.lock);

        /* Wait for the speak thread to finish — speak_lock is held
         * for the entire duration of voice_channel_speak(). */
        pthread_mutex_lock(&s_voice.speak_lock);
        pthread_mutex_unlock(&s_voice.speak_lock);

        syslog(LOG_INFO,
            "[%s] PTT: TTS playback fully closed\n", TAG);

        /* Re-acquire state lock and re-check — another thread may
         * have started recording while we released the lock. */
        pthread_mutex_lock(&s_voice.lock);
        if (s_voice.state != VOICE_IDLE) {
            pthread_mutex_unlock(&s_voice.lock);
            syslog(LOG_WARNING, "[%s] state changed during TTS stop\n", TAG);
            return -EBUSY;
        }
    }

    /* Batch-only backends retain the complete recording. Streaming failures
     * are terminal because earlier PCM is not buffered. */
    while (sem_trywait(&s_voice.rec_done) == 0) {}
    s_voice.auto_endpoint = automatic;
    s_voice.pcm_cap = automatic ? AUTO_PCM_BYTES : AGENT_VOICE_PCM_BUF_SIZE;
    s_voice.pcm_buf = NULL;
    s_voice.pcm_len = 0;
    s_voice.asr_stream = NULL;

    /* Prepare only supported streaming backends before capture. */
    s_voice.capture_error = 0;
    voice_asr_stream_t* pre_stream = NULL;
    if (voice_asr_stream_supported()) {
        pre_stream = voice_asr_stream_open();
        if (!pre_stream) {
            int error = errno;
            pthread_mutex_unlock(&s_voice.lock);
            return -error;
        }
    }
    if (pre_stream) {
        syslog(LOG_INFO, "[%s] ASR pre-connected\n", TAG);
        s_voice.asr_stream = pre_stream;
    }

    /* Open capture device (but do NOT start yet — start after the
     * recording thread is spawned so the consumer is ready before
     * audio frames begin flowing, preventing media_recorder queue
     * overflow "data queue is more than max count(4)"). */
    s_voice.cap = audio_capture_open(
        AGENT_AUDIO_CAPTURE_DEV,
        AGENT_VOICE_SAMPLE_RATE,
        AGENT_VOICE_CHANNELS,
        AGENT_VOICE_BITS);

    if (!s_voice.cap) {
        int open_error = errno ? -errno : -EIO;
        if (pre_stream) {
            voice_asr_stream_abort(pre_stream);
            s_voice.asr_stream = NULL;
        }
        pthread_mutex_unlock(&s_voice.lock);
        int cleanup = audio_capture_cleanup(5000);
        pthread_mutex_lock(&s_voice.lock);
        s_voice.capture_cleanup_pending = cleanup < 0;
        s_voice.capture_cleanup_result = cleanup;
        s_voice.state = cleanup < 0 ? VOICE_STOPPING :
            (s_voice.turn_active ? VOICE_PROCESSING : VOICE_IDLE);
        pthread_mutex_unlock(&s_voice.lock);
        syslog(LOG_ERR, "[%s] capture open failed: %d cleanup=%d\n",
            TAG, open_error, cleanup);
        return cleanup < 0 ? cleanup : open_error;
    }

    s_voice.state = VOICE_RECORDING;
    pthread_mutex_unlock(&s_voice.lock);

    /* Spawn recording thread BEFORE starting capture */
    pthread_attr_t attr;

    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, AGENT_VOICE_STACK);

    int ret = pthread_create(&s_voice.rec_thread, &attr,
        recording_thread, NULL);

    pthread_attr_destroy(&attr);

    if (ret != 0) {
        pthread_mutex_lock(&s_voice.lock);
        s_voice.state = VOICE_STOPPING;
        audio_capture_t* cap = s_voice.cap;
        voice_asr_stream_t* stream = s_voice.asr_stream;
        unsigned char* pcm = s_voice.pcm_buf;
        size_t pcm_len = s_voice.pcm_len;
        s_voice.asr_stream = NULL;
        s_voice.pcm_buf = NULL;
        s_voice.pcm_len = 0;
        pthread_mutex_unlock(&s_voice.lock);
        if (stream) voice_asr_stream_abort(stream);
        if (pcm) memset(pcm, 0, pcm_len);
        free(pcm);
        int close_ret = audio_capture_close(cap);
        pthread_mutex_lock(&s_voice.lock);
        if (close_ret == 0) {
            s_voice.cap = NULL;
            s_voice.state = s_voice.turn_active ?
                VOICE_PROCESSING : VOICE_IDLE;
        } else {
            s_voice.capture_cleanup_pending = 1;
            s_voice.capture_cleanup_result = close_ret;
        }
        pthread_mutex_unlock(&s_voice.lock);
        return close_ret < 0 ? close_ret : -ret;
    }

    /* The worker owns its stream until it has joined. A start failure must
     * not expose IDLE, free a stream, or reuse its PCM while it can still run. */
    int ready;
    do {
        ready = sem_wait(&s_voice.rec_ready);
    } while (ready < 0 && errno == EINTR);
    int start_error = ready < 0 ? -errno : 0;
    pthread_mutex_lock(&s_voice.lock);
    if (!start_error) start_error = s_voice.capture_error;
    pthread_mutex_unlock(&s_voice.lock);
    if (!start_error) start_error = audio_capture_start(s_voice.cap);
    if (start_error < 0) {
        pthread_mutex_lock(&s_voice.lock);
        s_voice.state = VOICE_STOPPING;
        pthread_mutex_unlock(&s_voice.lock);
        audio_capture_abort(s_voice.cap);
        pthread_join(s_voice.rec_thread, NULL);
        pthread_mutex_lock(&s_voice.lock);
        audio_capture_t* cap = s_voice.cap;
        voice_asr_stream_t* stream = s_voice.asr_stream;
        unsigned char* pcm = s_voice.pcm_buf;
        size_t pcm_len = s_voice.pcm_len;
        s_voice.asr_stream = NULL;
        s_voice.pcm_buf = NULL;
        s_voice.pcm_len = 0;
        pthread_mutex_unlock(&s_voice.lock);
        if (stream) voice_asr_stream_abort(stream);
        if (pcm) memset(pcm, 0, pcm_len);
        free(pcm);
        int close_ret = audio_capture_close(cap);
        pthread_mutex_lock(&s_voice.lock);
        if (close_ret == 0) {
            s_voice.cap = NULL;
            s_voice.state = s_voice.turn_active ?
                VOICE_PROCESSING : VOICE_IDLE;
        } else {
            s_voice.capture_cleanup_pending = 1;
            s_voice.capture_cleanup_result = close_ret;
        }
        pthread_mutex_unlock(&s_voice.lock);
        return close_ret < 0 ? close_ret : start_error;
    }

    syslog(LOG_INFO, "[%s] recording started mode=%s\n", TAG,
        automatic ? "auto" : "ptt");
    if (!automatic) printf("Voice: recording... (use voice_stop to finish)\n");
    return 0;
}

int voice_channel_start(void)
{
    return voice_channel_start_internal(0);
}

int voice_channel_start_auto(void)
{
    pthread_mutex_lock(&s_voice.lock);
    if (!s_voice.initialized || s_voice.turn_active || s_voice.state != VOICE_IDLE ||
        s_voice.tts_active) {
        pthread_mutex_unlock(&s_voice.lock);
        return -EBUSY;
    }
    s_voice.state = VOICE_STARTING;
    s_voice.turn_active = 1;
    s_voice.canceled = 0;
    s_voice.request_id++;
    if (!s_voice.request_id) s_voice.request_id++;
    s_voice.turn_started_ms = voice_now_ms();
    uint64_t id = s_voice.request_id;
    pthread_mutex_unlock(&s_voice.lock);
    int ret = agent_task_create(auto_finalize_task, "voice_auto", 24 * 1024,
        NULL, AGENT_VOICE_PRIO);
    if (ret != OK) voice_request_complete(id, ret < 0 ? ret : -EIO);
    return ret;
}

int voice_channel_is_idle(void)
{
    pthread_mutex_lock(&s_voice.lock);
    int idle = s_voice.initialized && !s_voice.turn_active &&
        s_voice.state == VOICE_IDLE && !s_voice.tts_active;
    pthread_mutex_unlock(&s_voice.lock);
    return idle;
}

int voice_channel_cancel(void)
{
    pthread_mutex_lock(&s_voice.lock);
    if (!s_voice.turn_active) { pthread_mutex_unlock(&s_voice.lock); return -ENOENT; }
    s_voice.canceled = 1;
    s_voice.tts_abort = 1;
    if (s_voice.cap) audio_capture_abort(s_voice.cap);
    if (s_voice.tts_pb) audio_playback_stop(s_voice.tts_pb);
    pthread_mutex_unlock(&s_voice.lock);
    voice_asr_cancel();
    voice_tts_cancel();
    llm_cancel_request();
    /* Owners join/finish before emitting completion. */
    return 0;
}

int voice_channel_recover(void)
{
    enum { CLEANUP_NONE, CLEANUP_CAPTURE, CLEANUP_PLAYBACK } kind = CLEANUP_NONE;
    uint64_t id = 0;
    int result = 0;

    pthread_mutex_lock(&s_voice.lock);
    if (s_voice.cleanup_in_progress) {
        pthread_mutex_unlock(&s_voice.lock);
        return -EBUSY;
    }
    if (s_voice.capture_cleanup_pending) {
        kind = CLEANUP_CAPTURE;
        result = s_voice.capture_cleanup_result;
    } else if (s_voice.tts_cleanup_pending) {
        kind = CLEANUP_PLAYBACK;
        result = s_voice.tts_cleanup_result;
    }
    if (kind != CLEANUP_NONE) {
        s_voice.cleanup_in_progress = 1;
        id = s_voice.request_id;
    }
    pthread_mutex_unlock(&s_voice.lock);
    if (kind == CLEANUP_NONE) return 0;

    int ret = kind == CLEANUP_CAPTURE ?
        audio_capture_cleanup(100) : audio_playback_cleanup(100);

    pthread_mutex_lock(&s_voice.lock);
    s_voice.cleanup_in_progress = 0;
    int current = s_voice.turn_active && id == s_voice.request_id;
    if (ret == 0 && kind == CLEANUP_CAPTURE &&
        s_voice.capture_cleanup_pending) {
        s_voice.capture_cleanup_pending = 0;
        s_voice.capture_cleanup_result = 0;
        s_voice.cap = NULL;
        if (s_voice.state == VOICE_STOPPING)
            s_voice.state = current ? VOICE_PROCESSING : VOICE_IDLE;
    } else if (ret == 0 && kind == CLEANUP_PLAYBACK &&
               s_voice.tts_cleanup_pending) {
        s_voice.tts_cleanup_pending = 0;
        s_voice.tts_cleanup_result = 0;
        s_voice.tts_active = 0;
        s_voice.tts_abort = 0;
    }
    pthread_mutex_unlock(&s_voice.lock);

    if (ret == 0 && current) voice_request_complete(id, result);
    return ret;
}

int voice_channel_speak_reply(uint64_t id, const char *text)
{
    int ret = voice_request_status(id);
    if (ret != 0) return ret;
    pthread_mutex_lock(&s_voice.lock);
    if (id != s_voice.request_id || s_voice.state != VOICE_PROCESSING || s_voice.canceled) {
        pthread_mutex_unlock(&s_voice.lock);
        return -ECANCELED;
    }
    s_voice.state = VOICE_SPEAKING;
    pthread_mutex_unlock(&s_voice.lock);
    return voice_channel_speak(text);
}

int voice_channel_set_event_callback(voice_channel_event_cb callback)
{
    pthread_mutex_lock(&s_voice.lock);
    if (s_voice.state != VOICE_IDLE || s_voice.tts_active) {
        pthread_mutex_unlock(&s_voice.lock);
        return -EBUSY;
    }
    s_voice.event_cb = callback;
    pthread_mutex_unlock(&s_voice.lock);
    return 0;
}

int voice_channel_stop(void)
{
    pthread_mutex_lock(&s_voice.lock);
    if (s_voice.turn_active || s_voice.state != VOICE_RECORDING) {
        pthread_mutex_unlock(&s_voice.lock);
        return -EBUSY;
    }
    s_voice.turn_active = 1;
    s_voice.canceled = 0;
    s_voice.request_id++;
    if (!s_voice.request_id) s_voice.request_id++;
    s_voice.turn_started_ms = voice_now_ms();
    uint64_t id = s_voice.request_id;
    pthread_mutex_unlock(&s_voice.lock);
    char text[512] = {0};
    int ret = voice_channel_stop_with_text(text, sizeof(text));
    if (!ret) {
        agent_msg_t msg = {0};
        strncpy(msg.channel, AGENT_CHAN_VOICE, sizeof(msg.channel) - 1);
        strncpy(msg.chat_id, "voice", sizeof(msg.chat_id) - 1);
        msg.request_id = id;
        msg.request_status = voice_request_status;
        msg.request_complete = voice_request_complete;
        msg.content = strdup(text);
        ret = msg.content ? message_bus_push_inbound(&msg) : -ENOMEM;
        if (ret) free(msg.content);
    }
    memset(text, 0, sizeof(text));
    if (ret) voice_request_complete(id, ret);
    return ret;
}

int voice_channel_stop_with_text(char* text_out, size_t text_cap)
{
    if (!text_out || !text_cap) return -EINVAL;
    text_out[0] = '\0';
    pthread_mutex_lock(&s_voice.lock);
    if (s_voice.state != VOICE_RECORDING) {
        pthread_mutex_unlock(&s_voice.lock);
        return -EINVAL;
    }
    s_voice.state = VOICE_STOPPING;
    audio_capture_t *cap = s_voice.cap;
    pthread_mutex_unlock(&s_voice.lock);
    audio_capture_abort(cap);
    pthread_join(s_voice.rec_thread, NULL);
    pthread_mutex_lock(&s_voice.lock);
    unsigned char *pcm = s_voice.pcm_buf;
    size_t pcm_len = s_voice.pcm_len;
    voice_asr_stream_t *stream = s_voice.asr_stream;
    int ret = s_voice.canceled ? -ECANCELED : s_voice.capture_error;
    uint64_t id = s_voice.request_id;
    int tracked = s_voice.turn_active;
    s_voice.pcm_buf = NULL;
    s_voice.pcm_len = 0;
    s_voice.asr_stream = NULL;
    pthread_mutex_unlock(&s_voice.lock);
    int close_ret = audio_capture_close(cap);
    if (close_ret < 0) {
        syslog(LOG_ERR, "[%s] capture owner retained: %d\n",
            TAG, close_ret);
        ret = close_ret;
        if (stream) voice_asr_stream_abort(stream);
    } else if (ret != 0) {
        if (stream) voice_asr_stream_abort(stream);
    } else if (stream) {
        ret = voice_asr_stream_finish(stream, text_out, text_cap);
    } else if (!pcm || !pcm_len) {
        ret = -ENODATA;
    } else {
        ret = voice_asr_recognize_checked(pcm, pcm_len, text_out, text_cap,
            tracked ? voice_request_check : NULL, &id);
    }
    if (pcm) memset(pcm, 0, pcm_len);
    free(pcm);
    if (!ret && !text_out[0]) ret = -ENODATA;
    pthread_mutex_lock(&s_voice.lock);
    if (s_voice.canceled) ret = -ECANCELED;
    if (close_ret == 0) {
        s_voice.cap = NULL;
        s_voice.state = s_voice.turn_active ?
            VOICE_PROCESSING : VOICE_IDLE;
    } else {
        s_voice.capture_cleanup_pending = 1;
        s_voice.capture_cleanup_result = ret;
    }
    pthread_mutex_unlock(&s_voice.lock);
    if (ret) text_out[0] = '\0';
    else clock_gettime(CLOCK_MONOTONIC, &s_asr_done_ts);
    return ret;
}

/* Strip Markdown formatting that TTS would read aloud.
 * Removes: **bold**, *italic*, # headings, - list bullets.
 * Operates in-place, result is always <= original length. */
static void tts_strip_markdown(char* s)
{
    char* r = s; /* read pointer */
    char* w = s; /* write pointer */

    while (*r) {
        /* Skip ** (bold markers) */
        if (r[0] == '*' && r[1] == '*') {
            r += 2;
            continue;
        }

        /* Skip lone * (italic markers) \xe2\x80\x94 but keep * in
         * contexts like "3*4" (digit before and after) */
        if (r[0] == '*') {
            if ((r == s || !isdigit((unsigned char)r[-1]))
                || !isdigit((unsigned char)r[1])) {
                r++;
                continue;
            }
        }

        /* Skip # at line start (headings) */
        if (r[0] == '#' && (r == s || r[-1] == '\n')) {
            while (*r == '#' || *r == ' ') {
                r++;
            }
            continue;
        }

        /* Skip "- " at line start (list bullets) */
        if (r[0] == '-' && r[1] == ' '
            && (r == s || r[-1] == '\n')) {
            r += 2;
            continue;
        }

        *w++ = *r++;
    }

    *w = '\0';
}

int voice_channel_speak(const char* text)
{
    if (!text || text[0] == '\0') {
        return -EINVAL;
    }

    /* ── Serialize concurrent speak calls ── */
    /* If another thread is already speaking, abort its TTS playback
     * so we don't overlap media_player sessions (which causes the
     * media framework to attempt a ~1.3GB allocation and crash). */
    pthread_mutex_lock(&s_voice.lock);
    if (s_voice.tts_active) {
        syslog(LOG_INFO, "[%s] speak: aborting previous TTS\n", TAG);
        s_voice.tts_abort = 1;
        if (s_voice.tts_pb) audio_playback_stop(s_voice.tts_pb);
        int canceled = voice_tts_cancel();
        if (canceled < 0)
            syslog(LOG_WARNING, "[%s] TTS cancel unavailable: %d\n", TAG, canceled);
    }
    pthread_mutex_unlock(&s_voice.lock);

    pthread_mutex_lock(&s_voice.speak_lock);

    /* ── AEC workaround: reject TTS while PTT is recording ── */
    pthread_mutex_lock(&s_voice.lock);
    if (s_voice.state != VOICE_IDLE && s_voice.state != VOICE_SPEAKING) {
        pthread_mutex_unlock(&s_voice.lock);
        pthread_mutex_unlock(&s_voice.speak_lock);
        syslog(LOG_INFO,
            "[%s] speak: skipped, recording active\n", TAG);
        return -EBUSY;
    }
    pthread_mutex_unlock(&s_voice.lock);

    /* Work on a mutable copy so we can strip markdown in-place */
    size_t text_len = strlen(text);
    char* clean = malloc(text_len + 1);

    if (!clean) {
        pthread_mutex_unlock(&s_voice.speak_lock);
        return -ENOMEM;
    }

    memcpy(clean, text, text_len + 1);
    tts_strip_markdown(clean);

    /* If stripping left nothing, bail out */
    if (clean[0] == '\0') {
        free(clean);
        pthread_mutex_unlock(&s_voice.speak_lock);
        return 0;
    }

    syslog(LOG_INFO, "[%s] speak: %zu bytes\n", TAG, strlen(clean));

    clock_gettime(CLOCK_MONOTONIC, &s_tts_start);
    s_tts_first_chunk = 0;

    if (s_asr_done_ts.tv_sec > 0) {
        long llm_ms = (s_tts_start.tv_sec - s_asr_done_ts.tv_sec) * 1000
            + (s_tts_start.tv_nsec - s_asr_done_ts.tv_nsec) / 1000000;
        syslog(LOG_INFO, "[%s] LLM latency: %ldms\n", TAG, llm_ms);
    }

    int streaming = voice_tts_stream_supported();
    tts_output_t output = { .batch = !streaming };
    pthread_mutex_lock(&s_voice.lock);
    if (s_voice.turn_active && s_voice.canceled) {
        pthread_mutex_unlock(&s_voice.lock);
        free(clean);
        pthread_mutex_unlock(&s_voice.speak_lock);
        return -ECANCELED;
    }
    s_voice.tts_abort = 0;
    s_voice.tts_active = 1;
    uint64_t id = s_voice.request_id;
    int tracked = s_voice.turn_active;
    pthread_mutex_unlock(&s_voice.lock);

    int ret;
    if (streaming) {
        ret = tts_speak_queued(clean, &output, tracked, &id);
    } else {
        unsigned char* pcm = malloc(AGENT_VOICE_PCM_BUF_SIZE);
        size_t length = 0;
        ret = pcm ? voice_tts_speak_pcm_checked(clean, pcm, AGENT_VOICE_PCM_BUF_SIZE,
            &length, &output.format, tracked ? voice_request_check : NULL, &id) : -ENOMEM;
        if (ret == 0 && (length == 0 || length > AGENT_VOICE_PCM_BUF_SIZE))
            ret = -EPROTO;
        if (ret == 0) tts_stream_cb(pcm, length, 1, &output);
        free(pcm);
    }
    if (output.error) ret = output.error;
    if (ret == 0 && (!output.terminal || !output.pb)) ret = -EPROTO;

    struct timespec tts_net;
    clock_gettime(CLOCK_MONOTONIC, &tts_net);
    long net_ms = (tts_net.tv_sec - s_tts_start.tv_sec) * 1000
        + (tts_net.tv_nsec - s_tts_start.tv_nsec) / 1000000;
    syslog(LOG_INFO, "[%s] TTS synthesis done: %ldms\n", TAG, net_ms);

    /* A backend can fail after already delivering playable PCM.  Unless this
     * turn was actively canceled, close the data side and wait for Media's
     * COMPLETED event before releasing the player and rearming capture. */
    int aborted = atomic_load(&s_voice.tts_abort);
    if (output.pb && !aborted) {
        int drain_ret = audio_playback_drain(output.pb, 120000);
        if (ret == 0) ret = drain_ret;
    }

    /* Stop exposing the writer handle before teardown so cancellation cannot
     * race a freed cookie. Keep tts_active set until Media ownership is gone;
     * request completion and trigger rearm are gated on that flag. */
    pthread_mutex_lock(&s_voice.lock);
    s_voice.tts_pb = NULL;
    pthread_mutex_unlock(&s_voice.lock);

    int close_ret = audio_playback_close(output.pb);
    int closed = audio_playback_cleanup(5000);
    if (closed == 0 && close_ret < 0) close_ret = 0;
    if (ret == 0) ret = closed;

    pthread_mutex_lock(&s_voice.lock);
    if (closed == 0 && close_ret == 0) {
        s_voice.tts_active = 0;
        s_voice.tts_abort = 0;
    } else {
        s_voice.tts_cleanup_pending = 1;
        s_voice.tts_cleanup_result = ret ? ret :
            (closed ? closed : close_ret);
    }
    pthread_mutex_unlock(&s_voice.lock);
    free(clean);

    if (ret != 0) {
        if (closed != 0)
            syslog(LOG_ERR, "[%s] playback owner retained: %d\n", TAG, closed);
        syslog(LOG_ERR, "[%s] TTS stream failed: %d\n", TAG, ret);
        pthread_mutex_unlock(&s_voice.speak_lock);
        return ret;
    }

    struct timespec tend;
    clock_gettime(CLOCK_MONOTONIC, &tend);
    long total_ms = (tend.tv_sec - s_tts_start.tv_sec) * 1000
        + (tend.tv_nsec - s_tts_start.tv_nsec) / 1000000;
    syslog(LOG_INFO, "[%s] speak done: total %ldms (play wait %ldms)\n",
        TAG, total_ms, total_ms - net_ms);
    pthread_mutex_unlock(&s_voice.speak_lock);
    return 0;
}

/* ── Test commands (file-based, for QEMU) ───────────────── */

int voice_channel_test_tts(const char* text, const char* out_path)
{
    if (!text || !out_path) {
        printf("Usage: voice_test_tts <text> [output_path]\n");
        return -EINVAL;
    }

    printf("TTS: synthesizing \"%s\" ...\n", text);

    unsigned char* pcm = malloc(AGENT_VOICE_PCM_BUF_SIZE);

    if (!pcm) {
        printf("Error: out of memory\n");
        return -ENOMEM;
    }

    size_t pcm_len;
    int ret = voice_tts_speak(text, pcm,
        AGENT_VOICE_PCM_BUF_SIZE, &pcm_len);

    if (ret != 0) {
        free(pcm);
        printf("TTS failed: %d\n", ret);
        return ret;
    }

    ret = write_pcm_file(out_path, pcm, pcm_len);
    free(pcm);

    if (ret == 0) {
        printf("TTS OK: %zu bytes -> %s\n", pcm_len, out_path);
    } else {
        printf("Failed to write %s: %d\n", out_path, ret);
    }

    return ret;
}

typedef struct {
    char pcm_path[256];
} asr_task_args_t;

static void* asr_file_worker(void* arg)
{
    asr_task_args_t* a = (asr_task_args_t*)arg;
    unsigned char* pcm = NULL;
    size_t pcm_len = 0;

    int ret = read_pcm_file(a->pcm_path, &pcm, &pcm_len);

    if (ret != 0) {
        printf("Failed to read %s: %d\n", a->pcm_path, ret);
        free(a);
        return NULL;
    }

    printf("ASR: recognizing %zu bytes from %s ...\n",
        pcm_len, a->pcm_path);

    char text[512];

    ret = voice_asr_recognize(pcm, pcm_len, text, sizeof(text));
    free(pcm);

    if (ret == 0) {
        printf("ASR result: %s\n", text);

        agent_msg_t msg;

        memset(&msg, 0, sizeof(msg));
        strncpy(msg.channel, AGENT_CHAN_VOICE,
            sizeof(msg.channel) - 1);
        strncpy(msg.chat_id, "voice",
            sizeof(msg.chat_id) - 1);
        msg.content = strdup(text);
        if (msg.content) {
            message_bus_push_inbound(&msg);
            printf("Sent to agent.\n");
        }
    } else {
        printf("ASR failed: %d\n", ret);
    }

    free(a);
    return NULL;
}

int voice_channel_test_asr(const char* pcm_path)
{
    if (!pcm_path) {
        printf("Usage: voice_test_asr <pcm_file>\n");
        return -EINVAL;
    }

    asr_task_args_t* args = malloc(sizeof(asr_task_args_t));

    if (!args) {
        printf("Error: out of memory\n");
        return -ENOMEM;
    }

    strncpy(args->pcm_path, pcm_path, sizeof(args->pcm_path) - 1);
    args->pcm_path[sizeof(args->pcm_path) - 1] = '\0';

    int ret = agent_task_create(asr_file_worker, "asr_test",
        ASR_THREAD_STACK, args, AGENT_VOICE_PRIO);

    if (ret != OK) {
        printf("Error: failed to create ASR thread\n");
        free(args);
        return -EIO;
    }

    printf("ASR test started (background thread).\n");
    return 0;
}

void voice_channel_cleanup(void)
{
    voice_channel_cancel();
}
