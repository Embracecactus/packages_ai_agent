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
typedef void (*voice_channel_event_cb)(int event, int result);
#define VOICE_CHANNEL_EVENT_TTS_COMPLETE 1
#define VOICE_CHANNEL_EVENT_CAPTURE_COMPLETE 2
#define VOICE_CHANNEL_EVENT_TURN_COMPLETE 3
#define VOICE_CHANNEL_EVENT_INITIALIZED 4
#define VOICE_CHANNEL_EVENT_SERVICE_READY 5
void voice_channel_service_ready(int result);
int voice_channel_set_event_callback(voice_channel_event_cb callback);
int voice_channel_is_idle(void);
int voice_channel_cancel(void);

/* Retry teardown retained after a terminal Media close failure. The product
 * may call this from its existing service task; no new session is started. */
int voice_channel_recover(void);
int voice_channel_speak_reply(uint64_t request_id, const char *text);

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
