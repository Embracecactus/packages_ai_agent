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

#ifdef __cplusplus
extern "C" {
#endif

/* Generic TTS backend interface. Each provider implements this struct
 * and registers via voice_tts_register(). */

typedef void (*voice_tts_chunk_cb)(const unsigned char *pcm_data,
                                   size_t pcm_len,
                                   int is_last,
                                   void *user_data);

typedef enum {
    VOICE_TTS_LOCATION_UNSPECIFIED = 0,
    VOICE_TTS_LOCATION_DEVICE,
    VOICE_TTS_LOCATION_REMOTE
} voice_tts_location_t;

typedef struct {
    voice_tts_location_t location;
    bool needs_network;
    unsigned int sample_rate;
    unsigned int batch_sample_rate;
    unsigned int channels;
    unsigned int bits;
} voice_tts_capabilities_t;

typedef struct voice_tts_ops {
    const char *name;

    /* Prepare the configured backend once per activation: load a model or
     * snapshot service settings. No network/credential requirement is imposed
     * by this interface. On failure, release all partial resources. Backends
     * must not call registry functions from init/deinit. */
    int (*init)(void);
    /* Optional short per-request reset, serialized before cancel is enabled.
     * Do not load models here or call the registry from this operation. */
    int (*prepare_request)(void);

    /* Synthesize complete UTF-8 text into PCM16 LE. Legacy implementations
     * without capabilities retain the 16kHz/mono contract. */
    int (*synthesize)(const char *text,
                      unsigned char *pcm_out,
                      size_t pcm_cap,
                      size_t *pcm_len);

    /* Optional streaming synthesis and its native PCM sample rate. */
    int (*synthesize_stream)(const char *text, voice_tts_chunk_cb cb,
                             void *user_data);
    /* Describe the prepared model/backend, including its actual PCM format.
     * A single-voice engine validates its model/voice binding during init. */
    int (*get_capabilities)(voice_tts_capabilities_t *caps);
    /* Interrupt work only; synthesize remains its resource owner and returns
     * after cleanup. Optional: absence reports -ENOTSUP, never fake success. */
    int (*cancel)(void);

    /* Release resources held by the backend. */
    void (*deinit)(void);
} voice_tts_ops_t;

/* Register a TTS backend. Multiple backends can be registered. */
int voice_tts_register(const voice_tts_ops_t *ops);

/* Select/reload a backend at an idle boundary. Returns -EBUSY in flight.
 * The old model is released before loading the new one; on init failure no
 * backend remains active. There is no implicit fallback or dual residency. */
int voice_tts_set_backend(const char *name);

/* Get the name of the current active backend, or NULL. */
const char *voice_tts_get_backend(void);

/* Synthesize text using the active backend. */
int voice_tts_speak(const char *text,
                    unsigned char *pcm_out,
                    size_t pcm_cap,
                    size_t *pcm_len);

/* Native-format batch call. Returns the format snapshot with the PCM so a
 * later backend switch cannot relabel already synthesized audio. The legacy
 * voice_tts_speak entry accepts only 16kHz/mono, before making a request. */
int voice_tts_speak_pcm(const char *text, unsigned char *pcm_out,
                      size_t pcm_cap, size_t *pcm_len,
                      voice_tts_capabilities_t *format);

/* ── Streaming TTS interface ─────────────────────────────────── */

/* Callback invoked for each PCM chunk received from TTS.
 * pcm_data/pcm_len: decoded PCM; format is supplied by backend capabilities.
 * is_last: exactly one terminal callback on success (possibly zero bytes).
 * No callbacks may outlive the call. Errors need not emit a terminal callback.
 * user_data: opaque pointer passed to voice_tts_speak_stream(). */
/* Synthesize text with streaming callback. Each decoded PCM chunk
 * is delivered via cb by the selected engine or service.
 * Returns 0 on success, negative errno on error. */
int voice_tts_speak_stream(const char *text,
                           voice_tts_chunk_cb cb,
                           void *user_data);

int voice_tts_get_capabilities(voice_tts_capabilities_t *caps);
bool voice_tts_stream_supported(void);
bool voice_tts_is_busy(void);
int voice_tts_cancel(void);
/* Caller request checks run during the backend claim, after request reset.
 * They must not call this registry. Existing untracked calls are unchanged. */
int voice_tts_speak_pcm_checked(const char *text, unsigned char *pcm,
    size_t capacity, size_t *size, voice_tts_capabilities_t *format,
    int (*check)(void *), void *request);
int voice_tts_speak_stream_checked(const char *text, voice_tts_chunk_cb cb,
    void *context, int (*check)(void *), void *request);

#ifdef __cplusplus
}
#endif
