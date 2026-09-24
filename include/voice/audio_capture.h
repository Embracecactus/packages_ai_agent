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

#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Audio capture session handle (opaque). */
typedef struct audio_capture audio_capture_t;

/* Recent raw PCM16 mono input only (at most eight one-second buckets).
 * Receive intervals are scheduling/transport observations, not lost frames.
 * Hardware drop counts are unavailable from this interface. No PCM is saved. */
typedef struct audio_capture_stats {
    uint64_t first_sample;
    uint64_t end_sample; /* exclusive */
    uint64_t samples;
    uint32_t rms;
    uint32_t peak;
    int32_t dc;
    uint32_t clipped_permyriad;
    uint32_t max_receive_interval_ms;
    int stream_error;
} audio_capture_stats_t;

/* Optional platform route/policy mapping. Install before opening capture.
 * The capture owner enables it after Media STARTED and disables it after close.
 * STARTED only confirms the format-bearing graph link was queued; a Media
 * hardware route must enqueue activation behind that link, not start inline.
 * An error prevents capture or is retained for cleanup before the next open. */
int audio_capture_set_route(int (*route)(int active));

/* Optional warm-route preparation before queuing a new recorder graph link.
 * Return 1 only when a previously negotiated, compatible route was enabled,
 * 0 to defer a cold route until STARTED, or a negative error. Installed with
 * no active capture. route(0) must undo any partial preparation on failure. */
int audio_capture_set_route_prepare(int (*prepare)(unsigned int rate,
    unsigned int channels, unsigned int bits));

/* Local wake consumer of the same recorder used by the next conversation.
 * PCM stays local until handoff and a matching audio_capture_open claims it.
 * The queue is bounded to two seconds; only 400 ms is retained as history.
 * Lifecycle operations are serialized by the product owner. Stop/join the
 * local reader before close; handoff is called by that reader after detection.
 */
audio_capture_t *audio_capture_open_local(const char *dev_path,
    unsigned int sample_rate, unsigned int channels,
    unsigned int bits_per_sample);
int audio_capture_read_local(audio_capture_t *cap, void *buf, size_t len,
    uint64_t *first_sample);
int audio_capture_handoff(audio_capture_t *cap);
/* Detection frame end (absolute mono sample index), not a guessed word end. */
int audio_capture_handoff_at(audio_capture_t *cap, uint64_t detected_sample);
int audio_capture_read_at(audio_capture_t *cap, void *buf, size_t len,
    uint64_t *first_sample);
int audio_capture_get_handoff_sample(audio_capture_t *cap,
    uint64_t *detected_sample);
/* Keep the sole Media producer draining while dropping and clearing PCM.
 * The conversational reader must be quiescent before enabling discard. */
int audio_capture_set_discard(audio_capture_t *cap, int discard);

int audio_capture_get_stats(audio_capture_t *cap,
    audio_capture_stats_t *stats);

/* Open the capture device and configure for PCM recording.
 * Returns NULL on failure. */
audio_capture_t *audio_capture_open(const char *dev_path,
    unsigned int sample_rate, unsigned int channels,
    unsigned int bits_per_sample);

/* Start the audio hardware. Must be called after open. */
int audio_capture_start(audio_capture_t *cap);

/* Read PCM data from the capture device (blocking).
 * Returns bytes read, or negative errno on error. */
int audio_capture_read(audio_capture_t *cap,
    void *buf, size_t len);

/* Interrupt any in-flight blocking read without freeing the handle. */
int audio_capture_abort(audio_capture_t* cap);

/* Stop recording and release all resources. A negative result means the
 * handle remains owned and must not be replaced by another capture. */
int audio_capture_close(audio_capture_t *cap);

/* Retry release of a capture retained after an open/close failure. */
int audio_capture_cleanup(unsigned int timeout_ms);

#ifdef __cplusplus
}
#endif
