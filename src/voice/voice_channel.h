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
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize voice channel (load config keys). */
int voice_channel_init(void);

/* Start voice recording, ASR, then push text to agent via message_bus. */
int voice_channel_start(void);
int voice_channel_start_auto(void);
/* First local wake only: keep the same recorder through the acknowledgement. */
int voice_channel_start_auto_wake(void);
/* Submit an already completed input through the same cancellable turn owner.
 * PCM is signed 16-bit little-endian, 16 kHz mono, bounded to 15 seconds.
 * These calls bypass microphone capture and endpoint detection. PCM remains
 * caller-owned and immutable until synchronous ASR returns; text is copied
 * before return. Success means accepted for Agent dispatch, not playback
 * completion. Observe TURN_COMPLETE and is_idle for terminal cleanup.
 * An active capture, reply, or retained cleanup rejects input with -EBUSY.
 */
int voice_channel_submit_pcm(const unsigned char *pcm, size_t size);
int voice_channel_submit_text(const char *text);
/* Optional admission lease, checked under the voice owner lock at claim,
 * request-status and commit. It must not call channel APIs. Context remains
 * valid until TURN_COMPLETE; NULL selects the normal channel cancellation. */
int voice_channel_submit_text_checked(const char *text,
    int (*check)(void *), void *context);
int voice_channel_submit_pcm_checked(const unsigned char *pcm, size_t size,
    int (*check)(void *), void *context);
typedef void (*voice_channel_event_cb)(int event, int result);
/* Optional local text consumer. It is called outside the channel owner lock
 * after ASR succeeds: 0 leaves normal Agent dispatch intact, 1 consumes the
 * text, and a negative errno completes the current request as an error. */
typedef int (*voice_channel_local_text_handler)(uint64_t request_id,
    const char *text, int (*request_status)(uint64_t));
#define VOICE_CHANNEL_EVENT_TTS_COMPLETE 1
#define VOICE_CHANNEL_EVENT_CAPTURE_COMPLETE 2
#define VOICE_CHANNEL_EVENT_TURN_COMPLETE 3
#define VOICE_CHANNEL_EVENT_INITIALIZED 4
#define VOICE_CHANNEL_EVENT_SERVICE_READY 5
/* Synchronous notification after the recording worker has joined and the
 * capture lease is released, before ASR finalization. A local acknowledgement
 * here cannot enter this turn's captured PCM. Never emitted on capture error.
 */
#define VOICE_CHANNEL_EVENT_CAPTURE_QUIESCENT 6
/* Synchronous: capture's sole reader is paused and the producer discards PCM.
 * Product must call voice_channel_wake_ack_result() before returning. */
#define VOICE_CHANNEL_EVENT_WAKE_ACK_REQUEST 7
#define VOICE_CHANNEL_EVENT_WAKE_ACK_SKIP 8
#define VOICE_CHANNEL_EVENT_WAKE_ACK_CANCEL 9
/* Media accepted the first answer PCM / the output owner finished cleanup.
 * These are playback lifecycle signals, not acoustic timestamps. */
#define VOICE_CHANNEL_EVENT_OUTPUT_STARTED 10
#define VOICE_CHANNEL_EVENT_OUTPUT_FINISHED 11
void voice_channel_wake_ack_result(int result);
void voice_channel_service_ready(int result);
int voice_channel_set_event_callback(voice_channel_event_cb callback);
/* Registration is accepted only while idle, so an active turn sees one
 * stable handler without extending the owner lock across callback code. */
int voice_channel_set_local_text_handler(voice_channel_local_text_handler callback);
int voice_channel_is_idle(void);
/* Last allocated turn identity for lifecycle logging, including after idle. */
uint64_t voice_channel_request_id(void);
int voice_channel_cancel(void);

/* Retry teardown retained after a terminal Media close failure. The product
 * may call this from its existing service task; no new session is started. */
int voice_channel_recover(void);
int voice_channel_speak_reply(uint64_t request_id, const char *text);

/* Platform-owned storage for the bounded TTS PCM reservoir. The weak Agent
 * implementation uses the normal heap and falls back to minimum_size if the
 * preferred allocation fails. A target with external memory may override all
 * three functions; it reports the actual capacity through allocated_size.
 * Allocation and release must be a matched pair and the memory must remain
 * CPU-readable until release. Control locks, the consumer stack and each
 * Media transfer frame stay in normal memory. */
#define VOICE_CHANNEL_TTS_STORAGE_NORMAL   0
#define VOICE_CHANNEL_TTS_STORAGE_EXTERNAL 1
void *voice_channel_tts_reservoir_alloc(size_t preferred_size,
    size_t minimum_size, size_t *allocated_size);
void voice_channel_tts_reservoir_free(void *memory);
int voice_channel_tts_reservoir_storage(const void *memory);

/* Stop an active voice session. */
int voice_channel_stop(void);

/* Stop recording and return ASR text to caller (does NOT push inbound).
 * text_out: buffer to receive ASR text, text_cap: buffer capacity.
 * Returns 0 on success, negative errno on failure.
 * Empty ASR is -ENODATA; cancellation is -ECANCELED. */
int voice_channel_stop_with_text(char *text_out, size_t text_cap);

/* Synthesize text and play back (called from outbound dispatcher). */
int voice_channel_speak(const char *text);

/* Test TTS: synthesize text, save PCM to file. */
int voice_channel_test_tts(const char *text, const char *out_path);

/* Test ASR: read PCM file, recognize, print result. */
int voice_channel_test_asr(const char *pcm_path);

#ifdef __cplusplus
}
#endif
