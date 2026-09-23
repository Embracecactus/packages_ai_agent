/* SPDX-License-Identifier: Apache-2.0 */
#include "voice/voice_endpoint.h"
#include <errno.h>
#include <string.h>

int voice_endpoint_init(voice_endpoint_t *s, unsigned int rate,
                        unsigned int tail_ms)
{
    if (!s || rate < 8000 || rate > 48000 || rate % 50 ||
        tail_ms < 500 || tail_ms > 1500) return -EINVAL;
    memset(s, 0, sizeof(*s));
    s->rate = rate;
    s->tail_ms = tail_ms;
    s->frame_samples = rate / 50;
    s->noise_q8 = 12u << 8;
    return 0;
}

int voice_endpoint_feed(voice_endpoint_t *s, const int16_t *pcm, size_t count)
{
    if (!s || !s->rate || (!pcm && count)) return -EINVAL;
    if (s->ended) return 2;
    for (size_t i = 0; i < count; i++) {
        int v = pcm[i];
        unsigned int a = v < 0 ? -v : v;
        s->sum += v;
        s->absolute += a;
        if (a > s->peak) s->peak = a;
        s->samples++;
        if (++s->frame_count < s->frame_samples) continue;
        unsigned int mean = s->absolute / s->frame_count;
        /* Remove only the conservative DC contribution from the detector;
         * the original waveform remains unchanged for recognition. */
        int64_t dc = s->sum / s->frame_count;
        unsigned int bias = dc < 0 ? -dc : dc;
        unsigned int energy = mean > bias ? mean - bias : 0;
        unsigned int noise = s->noise_q8 >> 8;
        unsigned int threshold = noise * (s->started ? 2 : 3);
        unsigned int floor = s->started ? 24 : 40;
        if (threshold < floor) threshold = floor;
        s->threshold = threshold;
        if (energy >= threshold) {
            s->attack_samples += s->frame_samples;
            s->speech_samples += s->frame_samples;
            s->quiet_samples = 0;
            if (s->attack_samples >= s->rate * 60u / 1000u) s->started = 1;
        } else {
            s->attack_samples = 0;
            if (s->started) s->quiet_samples += s->frame_samples;
            /* Fast downward, slow upward tracking cannot instantly absorb a
             * quiet syllable into a new noise floor. */
            unsigned int target = energy << 8;
            if (target < s->noise_q8)
                s->noise_q8 -= (s->noise_q8 - target) / 8;
            else if (!s->started)
                s->noise_q8 += (target - s->noise_q8) / 64;
            if (s->noise_q8 < (3u << 8)) s->noise_q8 = 3u << 8;
        }
        s->frame_count = 0;
        s->sum = 0;
        s->absolute = 0;
        if (s->started && s->speech_samples >= s->rate / 5u &&
            s->quiet_samples * 1000u >= (uint64_t)s->tail_ms * s->rate) {
            s->ended = 1;
            return 2;
        }
    }
    return s->started ? 1 : 0;
}
