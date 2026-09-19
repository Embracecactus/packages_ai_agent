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

/* Generic ASR backend interface. Each provider implements this struct
 * and registers via voice_asr_register(). */

typedef void *voice_asr_stream_handle_t;

typedef struct voice_asr_ops {
    const char *name;

    /* Prepare resources and freeze configured model/service settings once per
     * activation. On failure release partial resources. init/deinit must not
     * call registry functions. Network and credentials are backend concerns. */
    int (*init)(void);
    /* Optional short per-request reset before concurrent cancellation becomes
     * possible. This is not model loading and must not call the registry. */
    int (*prepare_request)(void);

    /* Recognize PCM audio (16-bit LE, 16kHz, mono) into text. */
    int (*recognize)(const unsigned char *pcm_data,
                     size_t pcm_len,
                     char *text_out,
                     size_t text_cap);
    /* Interrupt only; recognize retains ownership until it returns. */
    int (*cancel)(void);

    /* Optional streaming operations.  All four must be provided together. */
    voice_asr_stream_handle_t (*stream_open)(void);
    int (*stream_send)(voice_asr_stream_handle_t stream,
                       const unsigned char *pcm, size_t len);
    int (*stream_finish)(voice_asr_stream_handle_t stream,
                         char *text_out, size_t text_cap);
    void (*stream_abort)(voice_asr_stream_handle_t stream);
    /* Release resources held by the backend. */
    void (*deinit)(void);
} voice_asr_ops_t;

/* Register an ASR backend. Multiple backends can be registered. */
int voice_asr_register(const voice_asr_ops_t *ops);

/* Select/reload only when idle (-EBUSY otherwise). Release the old resources
 * before init; failed init leaves no active backend and never falls back. */
int voice_asr_set_backend(const char *name);

/* Get the name of the current active backend, or NULL. */
const char *voice_asr_get_backend(void);

/* Recognize speech using the active backend. */
int voice_asr_recognize(const unsigned char *pcm_data,
                        size_t pcm_len,
                        char *text_out,
                        size_t text_cap);
int voice_asr_cancel(void);
/* Optional caller request check, evaluated while claiming the backend and
 * after its per-request reset. It closes the cancel-before-start window.
 * The callback must not call the backend registry. */
int voice_asr_recognize_checked(const unsigned char *pcm, size_t size,
    char *text, size_t capacity, int (*check)(void *), void *request);

/* ── Streaming ASR interface ─────────────────────────────────── */

/* Opaque handle for a streaming ASR session. */
typedef struct voice_asr_stream voice_asr_stream_t;

/* Open a streaming ASR session using the active backend.
 * Binds the selected backend/configuration until finish/abort. Returns NULL
 * with errno set (including ENOTSUP and EBUSY). */
voice_asr_stream_t *voice_asr_stream_open(void);

/* Send one PCM chunk to the streaming session. */
int voice_asr_stream_send(voice_asr_stream_t *s,
                          const unsigned char *pcm, size_t len);

/* Finish streaming, get final text, and free the session. */
int voice_asr_stream_finish(voice_asr_stream_t *s,
                            char *text_out, size_t text_cap);

/* Destroy the stream after its send/finish/cancel callers have stopped. */
void voice_asr_stream_abort(voice_asr_stream_t *s);
bool voice_asr_stream_supported(void);
bool voice_asr_is_busy(void);

#ifdef __cplusplus
}
#endif
