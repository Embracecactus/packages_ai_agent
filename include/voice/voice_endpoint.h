/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* Sample-clock endpoint candidate. It consumes PCM, not scheduling intervals.
 * Noise adaptation is frozen on speech; no gain is applied to ASR samples.
 * This is an energy VAD, not a claim of neural speech/noise discrimination. */
typedef struct {
    unsigned int rate, tail_ms, frame_samples, frame_count;
    int64_t sum;
    uint64_t absolute, samples, speech_samples, quiet_samples;
    uint64_t attack_samples;
    unsigned int noise_q8, threshold, peak;
    int started, ended;
} voice_endpoint_t;

int voice_endpoint_init(voice_endpoint_t *state, unsigned int rate,
                        unsigned int tail_ms);
/* 0 waiting, 1 speech started, 2 endpoint. Input must be complete PCM16 frames. */
int voice_endpoint_feed(voice_endpoint_t *state, const int16_t *pcm,
                        size_t samples);
