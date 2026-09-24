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

/* audio_capture.c - Audio capture via direct ALSA or media_recorder fallback.
 *
 * The direct ALSA backend is gated behind CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT
 * because it pulls in chip-specific ALSA headers (aw-alsa-lib) that are only
 * available on boards shipping that SDK. When the option is disabled (the
 * default), the agent uses the portable media_recorder backend that works on
 * any board with the Vela media framework. */

#include "voice/audio_capture.h"
#include "agent_config.h"

#ifdef CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT
#include <aw-alsa-lib/pcm.h>
#endif
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <media_recorder.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <time.h>
#include <unistd.h>

static const char* TAG = "audio_cap";

#define CAP_OPTIONS_LEN 128
#define CAP_CLOSE_RETRY_US (50 * 1000)
#define CAP_CLOSE_TIMEOUT_MS 5000u
#define CAP_LOCAL_QUEUE_MS 2000u
#define CAP_LOCAL_HISTORY_MS 400u
#define CAP_PRODUCER_STACK 8192u
#define CAP_STATS_SECONDS 8u
#ifndef CAP_START_TIMEOUT_MS
#define CAP_START_TIMEOUT_MS 5000u
#endif
#ifdef CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT
#define CAP_ALSA_NAME_DEFAULT "default"
#endif

#ifdef CONFIG_AI_AGENT_AUDIO_CAPTURE_GAIN
#define AGENT_AUDIO_CAPTURE_GAIN CONFIG_AI_AGENT_AUDIO_CAPTURE_GAIN
#else
#define AGENT_AUDIO_CAPTURE_GAIN 6
#endif

enum audio_capture_backend {
    AUDIO_CAPTURE_BACKEND_NONE = 0,
    AUDIO_CAPTURE_BACKEND_ALSA,
    AUDIO_CAPTURE_BACKEND_MEDIA_RECORDER,
};

struct capture_stats_bucket {
    int64_t second;
    uint64_t first_sample;
    uint64_t end_sample;
    uint64_t samples;
    uint64_t sum_squares;
    int64_t sum;
    uint64_t clipped;
    uint32_t peak;
    uint32_t max_interval_ms;
};

struct audio_capture {
    enum audio_capture_backend backend;
    union {
#ifdef CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT
        snd_pcm_t* pcm;
#endif
        void* recorder;
    } handle;
    unsigned int bits_per_sample;
    unsigned int requested_channels;
    unsigned int hw_channels;
    size_t out_frame_bytes;
    size_t hw_frame_bytes;
    unsigned char* scratch;
    size_t scratch_size;
    int started;
    int route_active;
    int (*route)(int active);
    unsigned int sample_rate;
    sem_t start_event;
    atomic_int start_result;
    int start_requested;
    pthread_mutex_t lock;
    pthread_cond_t ready;
    pthread_t producer;
    unsigned char *ring;
    size_t ring_size;
    size_t history_bytes;
    uint64_t written;
    uint64_t consumed;
    int producer_valid;
    int stopping;
    int stream_error;
    int handoff;
    int local;
    int discard;
    unsigned int discard_epoch;
    uint64_t detected_sample;
    struct capture_stats_bucket stats[CAP_STATS_SECONDS];
    uint64_t stats_samples;
    int64_t stats_last_receive_ms;
};

static audio_capture_t* s_active_capture;
static int (*s_route)(int active);

static int capture_read_device(audio_capture_t *cap, void *buf, size_t len);
static void *capture_produce(void *arg);

static void capture_event(void *cookie, int event, int result,
    const char *extra)
{
    audio_capture_t *cap = cookie;
    (void)extra;
    if (event == MEDIA_EVENT_STARTED || result < 0) {
        int expected = -EINPROGRESS;
        if (atomic_compare_exchange_strong(&cap->start_result,
                &expected, result))
            sem_post(&cap->start_event);
    }
}

static int capture_wait_started(audio_capture_t *cap)
{
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += CAP_START_TIMEOUT_MS / 1000u;
    deadline.tv_nsec += (CAP_START_TIMEOUT_MS % 1000u) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    while (atomic_load(&cap->start_result) == -EINPROGRESS) {
        if (sem_clockwait(&cap->start_event, CLOCK_MONOTONIC, &deadline) < 0 &&
                errno != EINTR)
            return -errno;
    }
    return atomic_load(&cap->start_result);
}

static int (*s_route_prepare)(unsigned int, unsigned int, unsigned int);

int audio_capture_set_route_prepare(int (*prepare)(unsigned int,
    unsigned int, unsigned int))
{
    if (s_active_capture) return -EBUSY;
    s_route_prepare = prepare;
    return 0;
}

int audio_capture_set_route(int (*route)(int active))
{
    if (s_active_capture) return -EBUSY;
    s_route = route;
    return 0;
}

static int64_t capture_now_ms(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static uint32_t capture_isqrt64(uint64_t value)
{
    uint64_t bit = (uint64_t)1 << 62;
    uint64_t root = 0;
    while (bit > value) bit >>= 2;
    while (bit) {
        if (value >= root + bit) {
            value -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    return (uint32_t)root;
}

/* Called only after a complete PCM chunk is accepted. Ring producers hold
 * cap->lock; direct readers follow the product's serialized owner contract. */
static void capture_stats_add(audio_capture_t *cap, const void *pcm,
    size_t bytes)
{
    if (cap->bits_per_sample != 16 || cap->requested_channels != 1 ||
        bytes < sizeof(int16_t)) return;
    size_t count = bytes / sizeof(int16_t);
    int64_t now = capture_now_ms();
    int64_t second = now / 1000;
    struct capture_stats_bucket *bucket =
        &cap->stats[(unsigned int)(second % CAP_STATS_SECONDS)];
    if (bucket->second != second || !bucket->samples) {
        memset(bucket, 0, sizeof(*bucket));
        bucket->second = second;
        bucket->first_sample = cap->stats_samples;
    }
    if (cap->stats_last_receive_ms > 0 &&
        now > cap->stats_last_receive_ms) {
        int64_t gap = now - cap->stats_last_receive_ms;
        uint32_t bounded = gap > UINT32_MAX ? UINT32_MAX : (uint32_t)gap;
        if (bounded > bucket->max_interval_ms)
            bucket->max_interval_ms = bounded;
    }
    cap->stats_last_receive_ms = now;
    const int16_t *samples = pcm;
    for (size_t i = 0; i < count; i++) {
        int32_t value = samples[i];
        uint32_t magnitude = value < 0 ? (uint32_t)-value : (uint32_t)value;
        bucket->sum += value;
        bucket->sum_squares += (uint64_t)((int64_t)value * value);
        if (magnitude > bucket->peak) bucket->peak = magnitude;
        if (magnitude >= 32767) bucket->clipped++;
    }
    bucket->samples += count;
    cap->stats_samples += count;
    bucket->end_sample = cap->stats_samples;
}

static void capture_stats_snapshot(audio_capture_t *cap,
    audio_capture_stats_t *stats)
{
    memset(stats, 0, sizeof(*stats));
    int64_t now_second = capture_now_ms() / 1000;
    uint64_t sum_squares = 0;
    int64_t sum = 0;
    uint64_t clipped = 0;
    for (unsigned int i = 0; i < CAP_STATS_SECONDS; i++) {
        const struct capture_stats_bucket *bucket = &cap->stats[i];
        if (!bucket->samples || bucket->second > now_second ||
            now_second - bucket->second >= CAP_STATS_SECONDS) continue;
        if (!stats->samples || bucket->first_sample < stats->first_sample)
            stats->first_sample = bucket->first_sample;
        if (bucket->end_sample > stats->end_sample)
            stats->end_sample = bucket->end_sample;
        stats->samples += bucket->samples;
        sum += bucket->sum;
        sum_squares += bucket->sum_squares;
        clipped += bucket->clipped;
        if (bucket->peak > stats->peak) stats->peak = bucket->peak;
        if (bucket->max_interval_ms > stats->max_receive_interval_ms)
            stats->max_receive_interval_ms = bucket->max_interval_ms;
    }
    if (stats->samples) {
        stats->rms = capture_isqrt64(sum_squares / stats->samples);
        stats->dc = (int32_t)(sum / (int64_t)stats->samples);
        stats->clipped_permyriad =
            (uint32_t)(clipped * 10000 / stats->samples);
    }
    stats->stream_error = cap->stream_error;
}

int audio_capture_get_stats(audio_capture_t *cap,
    audio_capture_stats_t *stats)
{
    if (!cap || !stats) return -EINVAL;
    if (cap->ring) pthread_mutex_lock(&cap->lock);
    capture_stats_snapshot(cap, stats);
    if (cap->ring) pthread_mutex_unlock(&cap->lock);
    return 0;
}

static void capture_stats_log(audio_capture_t *cap, const char *boundary)
{
    audio_capture_stats_t stats;
    if (audio_capture_get_stats(cap, &stats) < 0) return;
    syslog(LOG_INFO,
        "[%s] %s raw_pcm samples=[%llu,%llu) count=%llu rms=%u peak=%u dc=%ld clipped_permyriad=%u max_receive_interval_ms=%u stream_error=%d hardware_drop=unknown\n",
        TAG, boundary, (unsigned long long)stats.first_sample,
        (unsigned long long)stats.end_sample,
        (unsigned long long)stats.samples, (unsigned int)stats.rms,
        (unsigned int)stats.peak, (long)stats.dc,
        (unsigned int)stats.clipped_permyriad,
        (unsigned int)stats.max_receive_interval_ms, stats.stream_error);
}

static int close_media_recorder(audio_capture_t* cap, unsigned int timeout_ms)
{
    if (timeout_ms == 0) return -EINVAL;
    int last_ret = -EIO;
    int64_t deadline = capture_now_ms() + timeout_ms;

    do {
        int ret = media_recorder_close(cap->handle.recorder);
        if (ret >= 0) {
            cap->handle.recorder = NULL;
            return 0;
        }
        last_ret = ret;
        syslog(LOG_WARNING, "[%s] recorder close failed: %d; retrying release\n",
            TAG, ret);
        media_recorder_stop(cap->handle.recorder);
        media_recorder_reset(cap->handle.recorder);
        if (capture_now_ms() >= deadline) break;
        usleep(CAP_CLOSE_RETRY_US);
    } while (capture_now_ms() < deadline);

    return last_ret;
}

static int release_capture_route(audio_capture_t* cap,
    unsigned int timeout_ms)
{
    if (timeout_ms == 0) return -EINVAL;
    int last_ret = -EIO;
    int64_t deadline = capture_now_ms() + timeout_ms;

    if (!cap->route_active || !cap->route) return 0;
    do {
        int ret = cap->route(0);
        if (ret >= 0) {
            cap->route_active = 0;
            return 0;
        }
        last_ret = ret;
        syslog(LOG_WARNING, "[%s] route release failed: %d; retrying\n",
            TAG, ret);
        if (capture_now_ms() >= deadline) break;
        usleep(CAP_CLOSE_RETRY_US);
    } while (capture_now_ms() < deadline);

    return last_ret;
}

static void apply_capture_gain(void* buf, size_t len)
{
#if AGENT_AUDIO_CAPTURE_GAIN > 1
    int16_t* samples = (int16_t*)buf;
    int count = (int)len / (int)sizeof(int16_t);

    for (int i = 0; i < count; i++) {
        int32_t v = (int32_t)samples[i] * AGENT_AUDIO_CAPTURE_GAIN;
        if (v > 32767) {
            v = 32767;
        } else if (v < -32768) {
            v = -32768;
        }
        samples[i] = (int16_t)v;
    }
#else
    (void)buf;
    (void)len;
#endif
}

#ifdef CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT
static const char* resolve_alsa_name(const char* dev_path)
{
    if (!dev_path || dev_path[0] == '\0') {
        return CAP_ALSA_NAME_DEFAULT;
    }

    if (strncmp(dev_path, "/dev/audio/", strlen("/dev/audio/")) == 0) {
        return CAP_ALSA_NAME_DEFAULT;
    }

    return dev_path;
}

static snd_pcm_format_t capture_format_from_bits(unsigned int bits_per_sample)
{
    switch (bits_per_sample) {
    case 16:
        return SND_PCM_FORMAT_S16_LE;
    case 24:
        return SND_PCM_FORMAT_S24_LE;
    default:
        return (snd_pcm_format_t)-1;
    }
}

static int configure_alsa_capture(snd_pcm_t* pcm, snd_pcm_format_t format,
    unsigned int sample_rate, unsigned int channels)
{
    snd_pcm_hw_params_t* hw;
    snd_pcm_sw_params_t* sw;
    snd_pcm_uframes_t period_frames = sample_rate / 10;
    snd_pcm_uframes_t buffer_frames;
    int ret;

    if (period_frames == 0) {
        period_frames = 256;
    }

    buffer_frames = period_frames * 4;

    snd_pcm_hw_params_alloca(&hw);
    ret = snd_vela_pcm_hw_params_any(pcm, hw);
    if (ret < 0) {
        return ret;
    }

    ret = snd_vela_pcm_hw_params_set_access(pcm, hw,
        SND_PCM_ACCESS_RW_INTERLEAVED);
    if (ret < 0) {
        return ret;
    }

    ret = snd_vela_pcm_hw_params_set_format(pcm, hw, format);
    if (ret < 0) {
        return ret;
    }

    ret = snd_vela_pcm_hw_params_set_channels(pcm, hw, channels);
    if (ret < 0) {
        return ret;
    }

    ret = snd_vela_pcm_hw_params_set_rate(pcm, hw, sample_rate, 0);
    if (ret < 0) {
        return ret;
    }

    ret = snd_vela_pcm_hw_params_set_period_size(pcm, hw,
        period_frames, 0);
    if (ret < 0) {
        return ret;
    }

    ret = snd_vela_pcm_hw_params_set_buffer_size(pcm, hw,
        buffer_frames);
    if (ret < 0) {
        return ret;
    }

    ret = snd_vela_pcm_hw_params(pcm, hw);
    if (ret < 0) {
        return ret;
    }

    snd_pcm_sw_params_alloca(&sw);
    ret = snd_vela_pcm_sw_params_current(pcm, sw);
    if (ret < 0) {
        return ret;
    }

    ret = snd_vela_pcm_sw_params_set_start_threshold(pcm, sw, 1);
    if (ret < 0) {
        return ret;
    }

    ret = snd_vela_pcm_sw_params_set_stop_threshold(pcm, sw, buffer_frames);
    if (ret < 0) {
        return ret;
    }

    ret = snd_vela_pcm_sw_params_set_avail_min(pcm, sw, period_frames);
    if (ret < 0) {
        return ret;
    }

    return snd_vela_pcm_sw_params(pcm, sw);
}

static int ensure_scratch_capacity(audio_capture_t* cap, size_t need)
{
    if (cap->scratch_size >= need) {
        return 0;
    }

    unsigned char* scratch = realloc(cap->scratch, need);
    if (!scratch) {
        return -ENOMEM;
    }

    cap->scratch = scratch;
    cap->scratch_size = need;
    return 0;
}

static int alsa_read_frames(audio_capture_t* cap, void* buf,
    snd_pcm_uframes_t frames)
{
    snd_pcm_uframes_t done = 0;
    unsigned char* dst = buf;

    while (done < frames) {
        snd_pcm_sframes_t ret = snd_vela_pcm_readi(cap->handle.pcm,
            dst + done * cap->hw_frame_bytes, frames - done);

        if (ret == -EAGAIN) {
            usleep(10 * 1000);
            continue;
        }

        if (ret == -EPIPE) {
            syslog(LOG_WARNING, "[%s] ALSA overrun, preparing capture\n",
                TAG);
            snd_vela_pcm_prepare(cap->handle.pcm);
            continue;
        }

        if (ret == -ESTRPIPE) {
            continue;
        }

        if (ret < 0) {
            return done > 0 ? (int)done : (int)ret;
        }

        done += (snd_pcm_uframes_t)ret;
    }

    return (int)done;
}

static void downmix_interleaved_s16_to_mono(audio_capture_t* cap,
    void* out_buf, const void* in_buf, size_t frames)
{
    int16_t* out = out_buf;
    const int16_t* in = in_buf;

    for (size_t i = 0; i < frames; i++) {
        int32_t mixed = 0;

        for (unsigned int ch = 0; ch < cap->hw_channels; ch++) {
            mixed += in[i * cap->hw_channels + ch];
        }

        out[i] = (int16_t)(mixed / (int32_t)cap->hw_channels);
    }
}

static int open_alsa_capture(audio_capture_t* cap, const char* dev_path,
    unsigned int sample_rate, unsigned int channels,
    unsigned int bits_per_sample)
{
    const char* alsa_name = resolve_alsa_name(dev_path);
    snd_pcm_format_t format = capture_format_from_bits(bits_per_sample);
    unsigned int attempts[2];
    int attempt_count = 0;
    int last_ret = -ENODEV;

    if ((int)format < 0) {
        return -ENOTSUP;
    }

    attempts[attempt_count++] = channels;
    if (channels == 1) {
        attempts[attempt_count++] = 3;
    }

    for (int i = 0; i < attempt_count; i++) {
        snd_pcm_t* pcm = NULL;
        unsigned int hw_channels = attempts[i];
        int ret = snd_vela_pcm_open(&pcm, alsa_name,
            SND_VELA_PCM_STREAM_CAPTURE, 0);

        if (ret < 0) {
            last_ret = ret;
            continue;
        }

        ret = configure_alsa_capture(pcm, format, sample_rate, hw_channels);
        if (ret < 0) {
            snd_vela_pcm_close(pcm);
            last_ret = ret;
            continue;
        }

        cap->backend = AUDIO_CAPTURE_BACKEND_ALSA;
        cap->handle.pcm = pcm;
        cap->bits_per_sample = bits_per_sample;
        cap->requested_channels = channels;
        cap->hw_channels = hw_channels;
        cap->out_frame_bytes = (bits_per_sample / 8) * channels;
        cap->hw_frame_bytes = snd_vela_pcm_frames_to_bytes(pcm, 1);

        syslog(LOG_INFO,
            "[%s] opened direct ALSA capture (%s, %uHz, req=%uch, hw=%uch, %ubit)\n",
            TAG, alsa_name, sample_rate, channels, hw_channels,
            bits_per_sample);
        return 0;
    }

    return last_ret;
}

#endif /* CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT */

static int open_media_recorder_capture(audio_capture_t* cap,
    unsigned int sample_rate, unsigned int channels,
    unsigned int bits_per_sample)
{
    char opts[CAP_OPTIONS_LEN];
    int ret;

    cap->handle.recorder = media_recorder_open(MEDIA_SOURCE_MIC);
    if (!cap->handle.recorder) {
        syslog(LOG_ERR, "[%s] media_recorder_open failed, errno=%d\n",
            TAG, errno);
        return -errno;
    }
    cap->backend = AUDIO_CAPTURE_BACKEND_MEDIA_RECORDER;

    ret = media_recorder_set_event_callback(cap->handle.recorder,
        cap, capture_event);
    if (ret < 0) {
        int close_ret = close_media_recorder(cap, CAP_CLOSE_TIMEOUT_MS);
        return close_ret < 0 ? close_ret : ret;
    }

    snprintf(opts, sizeof(opts),
        "format=s%ule:sample_rate=%u:ch_layout=%s",
        bits_per_sample, sample_rate,
        (channels == 1) ? "mono" : "stereo");

    ret = media_recorder_prepare(cap->handle.recorder, NULL, opts);
    if (ret < 0) {
        syslog(LOG_ERR, "[%s] prepare failed: %d\n", TAG, ret);
        int close_ret = close_media_recorder(cap, CAP_CLOSE_TIMEOUT_MS);
        return close_ret < 0 ? close_ret : ret;
    }

    cap->bits_per_sample = bits_per_sample;
    cap->requested_channels = channels;
    cap->hw_channels = channels;
    cap->out_frame_bytes = (bits_per_sample / 8) * channels;
    cap->hw_frame_bytes = cap->out_frame_bytes;

    syslog(LOG_INFO, "[%s] opened media recorder (%uHz %uch %ubit)\n",
        TAG, sample_rate, channels, bits_per_sample);
    return 0;
}

audio_capture_t* audio_capture_open(const char* dev_path,
    unsigned int sample_rate, unsigned int channels,
    unsigned int bits_per_sample)
{
    if (s_active_capture) {
        audio_capture_t *active = s_active_capture;
        if (active->ring) {
            pthread_mutex_lock(&active->lock);
            if (active->handoff && !active->stopping &&
                !active->stream_error && active->sample_rate == sample_rate &&
                active->requested_channels == channels &&
                active->bits_per_sample == bits_per_sample) {
                active->handoff = 0;
                active->local = 0;
                pthread_mutex_unlock(&active->lock);
                return active;
            }
            int error = active->stream_error;
            pthread_mutex_unlock(&active->lock);
            errno = error ? -error : EBUSY;
        } else {
            errno = EBUSY;
        }
        return NULL;
    }

    audio_capture_t* cap = calloc(1, sizeof(*cap));
    if (!cap) {
        return NULL;
    }
    if (sem_init(&cap->start_event, 0, 0) < 0) {
        free(cap);
        return NULL;
    }
    atomic_init(&cap->start_result, -EINPROGRESS);

    int ret = -ENOTSUP;
#ifdef CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT
    ret = open_alsa_capture(cap, dev_path, sample_rate, channels,
        bits_per_sample);
    if (ret < 0) {
        syslog(LOG_WARNING,
            "[%s] direct ALSA capture unavailable (%d), falling back to media recorder\n",
            TAG, ret);
    }
#else
    (void)dev_path;
#endif

    if (ret < 0) {
        ret = open_media_recorder_capture(cap, sample_rate, channels,
            bits_per_sample);
        if (ret < 0) {
            if (cap->backend == AUDIO_CAPTURE_BACKEND_MEDIA_RECORDER &&
                cap->handle.recorder) {
                s_active_capture = cap;
            } else {
                sem_destroy(&cap->start_event);
                free(cap);
            }
            errno = -ret;
            return NULL;
        }
    }

    s_active_capture = cap;
    cap->sample_rate = sample_rate;
    return cap;
}

audio_capture_t *audio_capture_open_local(const char *dev_path,
    unsigned int sample_rate, unsigned int channels, unsigned int bits)
{
    if (sample_rate != 16000 || channels != 1 || bits != 16) {
        errno = ENOTSUP;
        return NULL;
    }
    if (s_active_capture) {
        errno = EBUSY;
        return NULL;
    }
    audio_capture_t *cap = audio_capture_open(dev_path, sample_rate,
        channels, bits);
    if (!cap) return NULL;
    if (cap->backend != AUDIO_CAPTURE_BACKEND_MEDIA_RECORDER) {
        audio_capture_close(cap);
        errno = ENOTSUP;
        return NULL;
    }
    cap->ring_size = sample_rate * cap->out_frame_bytes * CAP_LOCAL_QUEUE_MS / 1000;
    cap->history_bytes = sample_rate * cap->out_frame_bytes * CAP_LOCAL_HISTORY_MS / 1000;
    unsigned char *ring = calloc(1, cap->ring_size);
    if (!ring) {
        audio_capture_close(cap);
        errno = ENOMEM;
        return NULL;
    }
    int ret = pthread_mutex_init(&cap->lock, NULL);
    if (!ret) {
        ret = pthread_cond_init(&cap->ready, NULL);
        if (ret) pthread_mutex_destroy(&cap->lock);
    }
    if (ret) {
        free(ring);
        audio_capture_close(cap);
        errno = ret;
        return NULL;
    }
    cap->ring = ring;
    cap->local = 1;
    return cap;
}

int audio_capture_start(audio_capture_t* cap)
{
    if (!cap) {
        return -EINVAL;
    }
    if (cap->started) return 0;

    switch (cap->backend) {
    case AUDIO_CAPTURE_BACKEND_ALSA:
#ifdef CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT
        if (!cap->handle.pcm) {
            return -EINVAL;
        }

        if (cap->started) {
            return 0;
        }

        if (snd_vela_pcm_prepare(cap->handle.pcm) < 0) {
            syslog(LOG_ERR, "[%s] ALSA prepare failed\n", TAG);
            return -EIO;
        }

        cap->started = 1;
        syslog(LOG_INFO, "[%s] ALSA capture prepared\n", TAG);
        return 0;
#else
        return -ENOTSUP;
#endif

    case AUDIO_CAPTURE_BACKEND_MEDIA_RECORDER: {
        int ret;

        if (!cap->handle.recorder) {
            return -EINVAL;
        }

        /* The RPC acknowledges enqueue, not encoder/graph readiness. A late
         * start event after timeout must not authorize a second start. Close
         * the owned recorder before retrying with a fresh handle. */
        if (cap->start_requested) return -EALREADY;
        cap->start_requested = 1;
        cap->route = s_route;
        if (s_route_prepare && cap->route) {
            /* A stopped source can spin on EOF as soon as a new sink asks
             * for frames, starving the server before its STARTED event.
             * Only the platform can establish warm format compatibility. */
            cap->route_active = 1;
            ret = s_route_prepare(cap->sample_rate,
                cap->requested_channels, cap->bits_per_sample);
            if (ret < 0) return ret;
            cap->route_active = ret > 0;
        }
        ret = media_recorder_start(cap->handle.recorder);
        if (!ret) ret = capture_wait_started(cap);
        if (ret < 0) {
            syslog(LOG_ERR, "[%s] start not ready: %d; route_active=%d cleanup required\n",
                TAG, ret, cap->route_active);
            return ret;
        }

        /* STARTED guarantees the format-bearing graph link was queued.
         * Platform routes must enqueue hardware activation behind that link,
         * not issue an immediate start before negotiation has run. */
        cap->route = s_route;
        if (cap->route && !cap->route_active) {
            cap->route_active = 1;
            ret = cap->route(1);
            if (ret < 0) return ret;
        }

        cap->started = 1;
        if (cap->ring) {
            pthread_attr_t attr;
            ret = pthread_attr_init(&attr);
            if (!ret) {
                size_t stack = CAP_PRODUCER_STACK;
#ifdef PTHREAD_STACK_MIN
                if (stack < (size_t)PTHREAD_STACK_MIN) stack = PTHREAD_STACK_MIN;
#endif
                ret = pthread_attr_setstacksize(&attr, stack);
                if (!ret)
                    ret = pthread_create(&cap->producer, &attr, capture_produce, cap);
                pthread_attr_destroy(&attr);
            }
            if (ret) {
                cap->started = 0;
                media_recorder_stop(cap->handle.recorder);
                return -ret;
            }
            cap->producer_valid = 1;
        }
        syslog(LOG_INFO, "[%s] media recorder capture started\n", TAG);
        return 0;
    }

    default:
        return -EINVAL;
    }
}

static int capture_read_device(audio_capture_t* cap, void* buf, size_t len)
{
    if (!cap || !buf || len == 0) {
        return -EINVAL;
    }

    switch (cap->backend) {
    case AUDIO_CAPTURE_BACKEND_ALSA: {
#ifdef CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT
        snd_pcm_uframes_t frames;
        int nframes;
        size_t out_len;

        if (!cap->handle.pcm || cap->out_frame_bytes == 0) {
            return -EINVAL;
        }

        frames = len / cap->out_frame_bytes;
        if (frames == 0) {
            return -EINVAL;
        }

        if (cap->hw_channels == cap->requested_channels) {
            nframes = alsa_read_frames(cap, buf, frames);
            if (nframes <= 0) {
                return nframes;
            }

            out_len = (size_t)nframes * cap->out_frame_bytes;
            if (cap->bits_per_sample == 16) {
                capture_stats_add(cap, buf, out_len);
                apply_capture_gain(buf, out_len);
            }
            return (int)out_len;
        }

        if (cap->bits_per_sample != 16 || cap->requested_channels != 1) {
            return -ENOTSUP;
        }

        if (ensure_scratch_capacity(cap, frames * cap->hw_frame_bytes) < 0) {
            return -ENOMEM;
        }

        nframes = alsa_read_frames(cap, cap->scratch, frames);
        if (nframes <= 0) {
            return nframes;
        }

        downmix_interleaved_s16_to_mono(cap, buf, cap->scratch,
            (size_t)nframes);
        out_len = (size_t)nframes * cap->out_frame_bytes;
        capture_stats_add(cap, buf, out_len);
        apply_capture_gain(buf, out_len);
        return (int)out_len;
#else
        return -ENOTSUP;
#endif
    }

    case AUDIO_CAPTURE_BACKEND_MEDIA_RECORDER: {
        ssize_t n;

        if (!cap->handle.recorder) {
            return -EINVAL;
        }

        /* The public recorder read is blocking. Poll its existing data socket
         * first so endpoint, cancellation and total-turn deadlines remain
         * observable without changing Media or its socket ownership. */
        int socket = media_recorder_get_socket(cap->handle.recorder);
        if (socket < 0) {
            return socket;
        }
        struct pollfd fd = { .fd = socket, .events = POLLIN };
        int ready;
        do {
            ready = poll(&fd, 1, 100);
        } while (ready < 0 && errno == EINTR);
        if (ready == 0) {
            return -EAGAIN;
        }
        if (ready < 0) {
            return -errno;
        }
        if (!(fd.revents & POLLIN)) {
            return fd.revents & (POLLERR | POLLHUP | POLLNVAL) ?
                -EPIPE : -EAGAIN;
        }

        n = media_recorder_read_data(cap->handle.recorder, buf, len);
        if (n > 0 && cap->bits_per_sample == 16 && !cap->ring) {
            capture_stats_add(cap, buf, (size_t)n);
            apply_capture_gain(buf, (size_t)n);
        }

        return (int)n;
    }

    default:
        return -EINVAL;
    }
}

static void ring_copy(audio_capture_t *cap, uint64_t offset, void *buf,
    size_t len, int writing)
{
    size_t index = offset % cap->ring_size;
    size_t first = cap->ring_size - index;
    if (first > len) first = len;
    if (writing) {
        memcpy(cap->ring + index, buf, first);
        memcpy(cap->ring, (unsigned char *)buf + first, len - first);
    } else {
        memcpy(buf, cap->ring + index, first);
        memcpy((unsigned char *)buf + first, cap->ring, len - first);
    }
}

static void ring_clear(audio_capture_t *cap, uint64_t offset, size_t len)
{
    size_t index = offset % cap->ring_size;
    size_t first = cap->ring_size - index;
    if (first > len) first = len;
    memset(cap->ring + index, 0, first);
    memset(cap->ring, 0, len - first);
}

static void *capture_produce(void *arg)
{
    audio_capture_t *cap = arg;
    unsigned char pcm[640];
    size_t partial = 0;
    unsigned int producer_epoch = 0;
    int partial_from_discard = 0;
    for (;;) {
        pthread_mutex_lock(&cap->lock);
        int stop = cap->stopping;
        unsigned int read_epoch = cap->discard_epoch;
        pthread_mutex_unlock(&cap->lock);
        if (stop) break;
        if (producer_epoch != read_epoch) {
            if (partial) partial_from_discard = 1;
            producer_epoch = read_epoch;
        }
        int n = capture_read_device(cap, pcm + partial, sizeof(pcm) - partial);
        if (n == -EAGAIN || n == -EINTR) continue;
        size_t bytes = n > 0 ? (size_t)n + partial : 0;
        size_t complete = bytes - bytes % cap->out_frame_bytes;
        pthread_mutex_lock(&cap->lock);
        if (cap->stopping) {
            pthread_mutex_unlock(&cap->lock);
            break;
        }
        uint64_t oldest = cap->local && !cap->handoff &&
            cap->consumed > cap->history_bytes ?
            cap->consumed - cap->history_bytes :
            cap->local && !cap->handoff ? 0 : cap->consumed;
        int discard_read = cap->discard || read_epoch != cap->discard_epoch;
        if (n <= 0 || (!discard_read &&
            cap->written - oldest + complete > cap->ring_size)) {
            cap->stream_error = n < 0 ? n : n == 0 ?
                (partial ? -EPROTO : -EPIPE) : -EOVERFLOW;
            cap->stopping = 1;
        } else if (discard_read) {
            cap->written += complete;
            cap->consumed = cap->written;
            cap->stats_samples = cap->written / cap->out_frame_bytes;
            memset(pcm, 0, complete);
            partial = bytes - complete;
            if (partial) memmove(pcm, pcm + complete, partial);
            partial_from_discard = partial != 0;
        } else {
            /* A byte-stream may split a PCM sample at the pause boundary.
             * Drop that whole cross-boundary frame; never combine a speaker
             * tail byte with the resumed conversation's first byte. */
            if (partial_from_discard && complete >= cap->out_frame_bytes) {
                size_t drop = cap->out_frame_bytes;
                cap->written += drop;
                cap->consumed = cap->written;
                cap->stats_samples = cap->written / cap->out_frame_bytes;
                memmove(pcm, pcm + drop, bytes - drop);
                bytes -= drop;
                complete -= drop;
                partial_from_discard = 0;
            }
            capture_stats_add(cap, pcm, complete);
            ring_copy(cap, cap->written, pcm, complete, 1);
            cap->written += complete;
            partial = bytes - complete;
            if (partial) memmove(pcm, pcm + complete, partial);
        }
        stop = cap->stopping;
        pthread_cond_broadcast(&cap->ready);
        pthread_mutex_unlock(&cap->lock);
        if (stop) {
            syslog(LOG_ERR, "[%s] input stopped error=%d sample=%llu\n",
                TAG, cap->stream_error,
                (unsigned long long)(cap->written / cap->out_frame_bytes));
            media_recorder_stop(cap->handle.recorder);
            break;
        }
    }
    memset(pcm, 0, sizeof(pcm));
    return NULL;
}

static int capture_read_ring(audio_capture_t *cap, void *buf, size_t len,
    uint64_t *first_sample, int local)
{
    if (!cap || !cap->ring || !buf || len < cap->out_frame_bytes)
        return -EINVAL;
    pthread_mutex_lock(&cap->lock);
    if (cap->local != local || cap->handoff) {
        pthread_mutex_unlock(&cap->lock);
        return -EBUSY;
    }
    if (!cap->stopping && cap->written == cap->consumed) {
        struct timespec until;
        clock_gettime(CLOCK_REALTIME, &until);
        until.tv_nsec += 100000000;
        if (until.tv_nsec >= 1000000000) {
            until.tv_sec++;
            until.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&cap->ready, &cap->lock, &until);
    }
    int ret = cap->stream_error ? cap->stream_error :
        cap->stopping ? -ECANCELED : -EAGAIN;
    if (!cap->stopping && !cap->discard &&
        cap->written > cap->consumed) {
        size_t available = cap->written - cap->consumed;
        if (len > available) len = available;
        len -= len % cap->out_frame_bytes;
        if (first_sample) *first_sample = cap->consumed / cap->out_frame_bytes;
        ring_copy(cap, cap->consumed, buf, len, 0);
        uint64_t old = cap->consumed;
        cap->consumed += len;
        if (local) {
            uint64_t first = old > cap->history_bytes ?
                old - cap->history_bytes : 0;
            uint64_t end = cap->consumed > cap->history_bytes ?
                cap->consumed - cap->history_bytes : 0;
            ring_clear(cap, first, end - first);
        } else {
            ring_clear(cap, old, len);
        }
        ret = (int)len;
    }
    pthread_mutex_unlock(&cap->lock);
    if (ret > 0 && !local) apply_capture_gain(buf, ret);
    return ret;
}

int audio_capture_read(audio_capture_t *cap, void *buf, size_t len)
{
    if (cap && cap->ring) return capture_read_ring(cap, buf, len, NULL, 0);
    int ret = capture_read_device(cap, buf, len);
    if (cap && ret < 0 && ret != -EAGAIN && ret != -EINTR)
        cap->stream_error = ret;
    return ret;
}

int audio_capture_read_at(audio_capture_t *cap, void *buf, size_t len,
    uint64_t *first_sample)
{
    if (!cap || !cap->ring || !first_sample) return -EINVAL;
    return capture_read_ring(cap, buf, len, first_sample, 0);
}

int audio_capture_read_local(audio_capture_t *cap, void *buf, size_t len,
    uint64_t *first_sample)
{
    return capture_read_ring(cap, buf, len, first_sample, 1);
}

int audio_capture_handoff_at(audio_capture_t *cap, uint64_t detected_sample)
{
    if (!cap || !cap->ring) return -EINVAL;
    pthread_mutex_lock(&cap->lock);
    int ret = cap->stream_error ? cap->stream_error :
        cap->stopping ? -ECANCELED : !cap->local || cap->handoff ? -EBUSY : 0;
    if (!ret && detected_sample != UINT64_MAX &&
        detected_sample != cap->consumed / cap->out_frame_bytes)
        ret = -EINVAL;
    if (!ret) {
        cap->detected_sample = detected_sample == UINT64_MAX ?
            cap->consumed / cap->out_frame_bytes : detected_sample;
        cap->consumed = cap->consumed > cap->history_bytes ?
            cap->consumed - cap->history_bytes : 0;
        cap->handoff = 1;
        syslog(LOG_INFO, "[%s] handoff first_sample=%llu queued_samples=%llu\n",
            TAG, (unsigned long long)(cap->consumed / cap->out_frame_bytes),
            (unsigned long long)((cap->written - cap->consumed) / cap->out_frame_bytes));
    }
    pthread_mutex_unlock(&cap->lock);
    if (!ret) capture_stats_log(cap, "handoff");
    return ret;
}

int audio_capture_handoff(audio_capture_t *cap)
{
    return audio_capture_handoff_at(cap, UINT64_MAX);
}

int audio_capture_get_handoff_sample(audio_capture_t *cap,
    uint64_t *detected_sample)
{
    if (!cap || !cap->ring || !detected_sample) return -EINVAL;
    pthread_mutex_lock(&cap->lock);
    int ret = cap->local || cap->handoff ? -EBUSY : 0;
    if (!ret) *detected_sample = cap->detected_sample;
    pthread_mutex_unlock(&cap->lock);
    return ret;
}

int audio_capture_set_discard(audio_capture_t *cap, int discard)
{
    if (!cap || !cap->ring) return -EINVAL;
    pthread_mutex_lock(&cap->lock);
    int ret = cap->local || cap->handoff ? -EBUSY :
        cap->stopping ? -ECANCELED : cap->stream_error;
    if (!ret) {
        if (discard) {
            ring_clear(cap, cap->consumed, cap->written - cap->consumed);
            cap->consumed = cap->written;
        }
        if (cap->discard != !!discard) {
            cap->discard = !!discard;
            cap->discard_epoch++;
        }
        pthread_cond_broadcast(&cap->ready);
    }
    pthread_mutex_unlock(&cap->lock);
    return ret;
}

int audio_capture_abort(audio_capture_t* cap)
{
    if (!cap) {
        return -EINVAL;
    }
    if (cap->ring) {
        pthread_mutex_lock(&cap->lock);
        cap->stopping = 1;
        pthread_cond_broadcast(&cap->ready);
        pthread_mutex_unlock(&cap->lock);
        /* This reader polls at 100 ms and is the socket's sole consumer.
         * Join before invoking recorder control or releasing its socket. */
        if (cap->producer_valid) {
            int joined = pthread_join(cap->producer, NULL);
            if (joined) return -joined;
            cap->producer_valid = 0;
        }
    }

    switch (cap->backend) {
    case AUDIO_CAPTURE_BACKEND_ALSA:
#ifdef CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT
        if (cap->handle.pcm && cap->started) {
            int ret = snd_vela_pcm_drop(cap->handle.pcm);
            cap->started = 0;
            return ret;
        }
        return 0;
#else
        return -ENOTSUP;
#endif

    case AUDIO_CAPTURE_BACKEND_MEDIA_RECORDER:
        if (cap->handle.recorder) {
#ifdef CONFIG_MEDIA_GRAPH
            media_recorder_close_socket(cap->handle.recorder);
#endif
            return media_recorder_stop(cap->handle.recorder);
        }
        return 0;

    default:
        return -EINVAL;
    }
}

static int capture_close(audio_capture_t* cap, unsigned int timeout_ms)
{
    if (!cap) {
        return 0;
    }
    if (!timeout_ms) return -EINVAL;
    if (cap->ring) {
        int ret = audio_capture_abort(cap);
        if (ret < 0 && cap->producer_valid) return ret;
    }

    switch (cap->backend) {
    case AUDIO_CAPTURE_BACKEND_ALSA:
#ifdef CONFIG_AI_AGENT_AUDIO_ALSA_DIRECT
        if (cap->handle.pcm) {
            if (cap->started) {
                snd_vela_pcm_drop(cap->handle.pcm);
            }
            snd_vela_pcm_close(cap->handle.pcm);
            cap->handle.pcm = NULL;
        }
        free(cap->scratch);
        cap->scratch = NULL;
        cap->scratch_size = 0;
#endif
        break;

    case AUDIO_CAPTURE_BACKEND_MEDIA_RECORDER:
        if (cap->handle.recorder) {
            media_recorder_stop(cap->handle.recorder);
            int ret = close_media_recorder(cap, timeout_ms);
            if (ret < 0) return ret;
        }
        break;

    default:
        break;
    }

    int ret = release_capture_route(cap, timeout_ms);
    if (ret < 0) return ret;
    if (s_active_capture == cap) {
        s_active_capture = NULL;
    }

    capture_stats_log(cap, "close");

    if (cap->ring) {
        memset(cap->ring, 0, cap->ring_size);
        free(cap->ring);
        pthread_cond_destroy(&cap->ready);
        pthread_mutex_destroy(&cap->lock);
    }
    /* Successful recorder close joins its event dispatcher before this
     * semaphore and callback cookie are released. */
    sem_destroy(&cap->start_event);
    free(cap);
    syslog(LOG_INFO, "[%s] closed\n", TAG);
    return 0;
}

int audio_capture_close(audio_capture_t *cap)
{
    return capture_close(cap, CAP_CLOSE_TIMEOUT_MS);
}

int audio_capture_cleanup(unsigned int timeout_ms)
{
    audio_capture_t* cap = s_active_capture;
    if (!cap) return 0;
    if (cap->ring) {
        pthread_mutex_lock(&cap->lock);
        int local_owned = cap->local && !cap->handoff && cap->producer_valid;
        pthread_mutex_unlock(&cap->lock);
        /* A failed second open has no authority to release a live local
         * reader. Its owner must first join it and explicitly close. */
        if (local_owned) return -EBUSY;
    }
    return capture_close(cap, timeout_ms);
}
