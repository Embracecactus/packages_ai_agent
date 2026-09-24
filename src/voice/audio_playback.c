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

/* audio_playback.c - Streaming audio playback via media_player buffer mode. */

#include "voice/audio_playback.h"
#include "agent_config.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdint.h>
#include <time.h>
#include <media_player.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>

static const char* TAG = "audio_pb";

#define PB_OPTIONS_LEN 128
#define PB_CLOSE_RETRY_US (50 * 1000)
#define PB_CLOSE_TIMEOUT_MS 2000u

static pthread_mutex_t s_player_lock = PTHREAD_MUTEX_INITIALIZER;
static int s_player_reserved;
static audio_playback_t* s_pending_close;

struct audio_playback {
    void* player;
    size_t total_written;
    int bytes_per_frame;
    atomic_int stopped;
    atomic_int result;
    atomic_int completed;
    sem_t completion;
};

static int64_t playback_now_ms(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

/* s_player_lock serializes Media handle teardown and reservation changes. */
static int playback_release_locked(audio_playback_t* pb,
    unsigned int timeout_ms)
{
    int ret;
    int last_ret = -EIO;
    int64_t deadline = playback_now_ms() + timeout_ms;

    do {
        ret = media_player_close(pb->player, 0);
        if (ret >= 0) {
            pb->player = NULL;
            return 0;
        }
        last_ret = ret;
        syslog(LOG_WARNING, "[%s] close failed: %d; retrying release\n",
            TAG, ret);

        /* A failed close leaves the client and its event cookie connected.
         * Reset the existing public Media session before retrying; do not
         * free or reuse the callback storage while that ownership remains. */
        media_player_stop(pb->player);
        media_player_reset(pb->player);
        if (playback_now_ms() >= deadline) break;
        usleep(PB_CLOSE_RETRY_US);
    } while (playback_now_ms() < deadline);

    return last_ret;
}

static void playback_free_locked(audio_playback_t* pb)
{
    sem_destroy(&pb->completion);
    free(pb);
    s_player_reserved = 0;
}

int audio_playback_cleanup(unsigned int timeout_ms)
{
    if (timeout_ms == 0) return -EINVAL;

    pthread_mutex_lock(&s_player_lock);
    audio_playback_t* pb = s_pending_close;
    if (!pb) {
        int ret = s_player_reserved ? -EBUSY : 0;
        pthread_mutex_unlock(&s_player_lock);
        return ret;
    }

    int ret = playback_release_locked(pb, timeout_ms);
    if (ret == 0) {
        s_pending_close = NULL;
        playback_free_locked(pb);
    }
    pthread_mutex_unlock(&s_player_lock);
    return ret;
}

static void playback_event(void* cookie, int event, int result, const char* extra)
{
    audio_playback_t* pb = cookie;
    (void)extra;
    if (result < 0) {
        int expected = 0;
        atomic_compare_exchange_strong(&pb->result, &expected, result);
        sem_post(&pb->completion);
    } else if (event == MEDIA_EVENT_COMPLETED) {
        atomic_store(&pb->completed, 1);
        sem_post(&pb->completion);
    }
}

audio_playback_t* audio_playback_open(const char* dev_path,
    unsigned int sample_rate, unsigned int channels,
    unsigned int bits_per_sample)
{
    (void)dev_path;
    if (sample_rate == 0 || (channels != 1 && channels != 2) ||
        bits_per_sample != 16) {
        errno = EINVAL;
        return NULL;
    }

    int cleanup = audio_playback_cleanup(PB_CLOSE_TIMEOUT_MS);
    if (cleanup < 0 && cleanup != -EBUSY) {
        errno = -cleanup;
        return NULL;
    }

    pthread_mutex_lock(&s_player_lock);
    if (s_player_reserved) {
        pthread_mutex_unlock(&s_player_lock);
        errno = EBUSY;
        return NULL;
    }
    s_player_reserved = 1;
    pthread_mutex_unlock(&s_player_lock);

    audio_playback_t* pb = calloc(1, sizeof(*pb));
    if (!pb) {
        goto fail_reservation;
    }
    if (sem_init(&pb->completion, 0, 0) < 0) {
        free(pb);
        goto fail_reservation;
    }
    void* player = media_player_open(MEDIA_STREAM_MUSIC);
    if (!player) {
        syslog(LOG_ERR, "[%s] media_player_open failed\n", TAG);
        goto fail;
    }
    pb->player = player;
    if (media_player_set_event_callback(player, pb, playback_event) < 0) {
        goto fail;
    }

    char opts[PB_OPTIONS_LEN];
    /* The 16-kHz mono PCM demuxer emits 2048-byte frames. Match the
     * voice queue's 8192-byte prefill with four Media frames; the default
     * six-frame resume threshold would wait for another network chunk
     * even after the application has released a full prefill. Leave other
     * formats on the Media default until their buffering is coordinated. */
    snprintf(opts, sizeof(opts),
        "format=s%ule:sample_rate=%u:ch_layout=%s%s",
        bits_per_sample, sample_rate,
        (channels == 1) ? "mono" : "stereo",
        (sample_rate == 16000 && channels == 1) ? ":datqmax=4" : "");

    int ret = media_player_prepare(player, NULL, opts);
    if (ret < 0) {
        syslog(LOG_ERR, "[%s] prepare failed: %d\n", TAG, ret);
        goto fail;
    }

    ret = media_player_start(player);
    if (ret < 0) {
        syslog(LOG_ERR, "[%s] start failed: %d\n", TAG, ret);
        goto fail;
    }

    int fd = media_player_get_socket(player);
    int flags = fd < 0 ? -1 : fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        goto fail;
    }
    pb->bytes_per_frame = (bits_per_sample / 8) * channels;

    syslog(LOG_INFO, "[%s] opened (%uHz %uch %ubit)\n",
        TAG, sample_rate, channels, bits_per_sample);
    return pb;

fail:
    if (pb->player) {
        media_player_stop(pb->player);
        pthread_mutex_lock(&s_player_lock);
        int close_result = playback_release_locked(pb,
            PB_CLOSE_TIMEOUT_MS);
        if (close_result < 0) {
            s_pending_close = pb;
            pthread_mutex_unlock(&s_player_lock);
            errno = -close_result;
            return NULL;
        }
        pthread_mutex_unlock(&s_player_lock);
    }
    sem_destroy(&pb->completion);
    free(pb);
fail_reservation:
    pthread_mutex_lock(&s_player_lock);
    s_player_reserved = 0;
    pthread_mutex_unlock(&s_player_lock);
    return NULL;
}

int audio_playback_write(audio_playback_t* pb, const void* buf, size_t len)
{
    if (!pb || !pb->player || !buf || len == 0) {
        return -EINVAL;
    }

    /* Buffer-mode Media explicitly exposes its data socket for callers that
     * need normal stream semantics.  Use it here because a nonblocking stream
     * write may complete partially; media_player_write_data() treats that as
     * a terminal transfer error instead of returning the partial byte count. */
    int socket_fd = media_player_get_socket(pb->player);
    if (socket_fd < 0) {
        return socket_fd;
    }

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t deadline = (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000 + 1000;
    size_t written = 0;
    int ret = 0;
    while (written < len) {
        if (atomic_load(&pb->stopped)) {
            ret = -ECANCELED;
            break;
        }
        ret = atomic_load(&pb->result);
        if (ret < 0) {
            break;
        }
        ssize_t n = write(socket_fd,
            (const unsigned char*)buf + written, len - written);
        if (n > 0) {
            written += (size_t)n;
            continue;
        }
        if (n < 0 && n != -EAGAIN && n != -EWOULDBLOCK &&
            n != -EINTR && !(n == -1 &&
            (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))) {
            ret = n == -1 ? -errno : (int)n;
            break;
        }
        if (n == 0) {
            ret = -EPIPE;
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &now);
        int64_t remaining = deadline - ((int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000);
        if (remaining <= 0) {
            ret = -ETIMEDOUT;
            break;
        }
        struct pollfd fd = { .fd = socket_fd, .events = POLLOUT };
        int ready = poll(&fd, 1, remaining < 100 ? (int)remaining : 100);
        if (ready < 0 && errno != EINTR) {
            ret = -errno;
            break;
        }
        if (ready > 0 && (fd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
            ret = -EPIPE;
            break;
        }
    }
    pb->total_written += written;
    if (ret < 0) {
        int expected = 0;
        atomic_compare_exchange_strong(&pb->result, &expected, ret);
        sem_post(&pb->completion);
        return ret;
    }
    return (int)written;
}

int audio_playback_drain(audio_playback_t* pb, unsigned int timeout_ms)
{
    if (!pb || !pb->player || timeout_ms == 0) {
        return -EINVAL;
    }
    /* EOF closes only the data stream. The Media graph retains ownership
     * until its actual completion event, including downstream resampling. */
    media_player_close_socket(pb->player);
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += timeout_ms / 1000;
    deadline.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    while (!atomic_load(&pb->completed) && !atomic_load(&pb->result) &&
           !atomic_load(&pb->stopped)) {
        if (sem_clockwait(&pb->completion, CLOCK_MONOTONIC, &deadline) < 0 && errno != EINTR) {
            return -errno;
        }
    }
    if (atomic_load(&pb->stopped)) {
        return -ECANCELED;
    }
    return atomic_load(&pb->result);
}

void audio_playback_stop(audio_playback_t* pb)
{
    if (pb) {
        atomic_store(&pb->stopped, 1);
        sem_post(&pb->completion);
        if (pb->player) {
            media_player_stop(pb->player);
        }
    }
}

int audio_playback_close(audio_playback_t* pb)
{
    if (!pb) {
        return 0;
    }

    if (pb->player) {
        syslog(LOG_INFO, "[%s] closing (%zu bytes written)\n",
            TAG, pb->total_written);
        media_player_stop(pb->player);
        pthread_mutex_lock(&s_player_lock);
        int ret = playback_release_locked(pb, PB_CLOSE_TIMEOUT_MS);
        if (ret < 0) {
            /* Media has not disconnected its event thread. Keep its cookie
             * alive for explicit cleanup; never free under it. */
            s_pending_close = pb;
            pthread_mutex_unlock(&s_player_lock);
            return ret;
        }
        pthread_mutex_unlock(&s_player_lock);
    }

    pthread_mutex_lock(&s_player_lock);
    playback_free_locked(pb);
    pthread_mutex_unlock(&s_player_lock);
    return 0;
}
