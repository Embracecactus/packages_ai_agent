/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The deployment owns context and its verified TLS identity until the backend
 * is idle. open_verified MUST authenticate CA, hostname and trusted time.
 * send/recv obey absolute monotonic deadlines; interrupt may race with either,
 * but close is called only after the receive worker and sender have quiesced.
 * No plaintext or certificate-verification bypass is permitted. */
typedef struct {
    /* Called by ASR registry before cancellation becomes visible for a turn.
     * open_verified must never clear a cancellation issued after this point. */
    int (*prepare_request)(void* context);
    int (*open_verified)(void* context, const char* host, uint16_t port,
                         uint64_t deadline_ms);
    ssize_t (*send)(void* context, const uint8_t* bytes, size_t len,
                    uint64_t deadline_ms);
    ssize_t (*recv)(void* context, uint8_t* bytes, size_t len,
                    uint64_t deadline_ms);
    int (*interrupt)(void* context);
    int (*close)(void* context);
    int (*random)(void* context, uint8_t* bytes, size_t len);
    int (*sha1)(void* context, const uint8_t* bytes, size_t len,
                uint8_t digest[20]);
    uint64_t (*now_ms)(void* context);
} funasr_asr_transport_t;

/* Configure an explicit WSS endpoint before selecting the "funasr" backend.
 * Returns -EBUSY during a session. The transport is borrowed, not copied
 * across reconfiguration. A missing provider leaves backend activation closed.
 */
int funasr_asr_configure(const funasr_asr_transport_t* transport,
    void* context, const char* host, uint16_t port, const char* path);
/* Retry a retained close failure before the deployment releases context.
 * Until this returns 0, configuration and new sessions fail closed. */
int funasr_asr_recover(void);
int funasr_asr_register(void);

#ifdef __cplusplus
}
#endif
