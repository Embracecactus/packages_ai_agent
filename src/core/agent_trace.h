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

/**
 * agent_trace.h — Structured execution tracing for AI Agent Agent
 *
 * Provides run_id-based structured logging so that each agent
 * execution can be traced end-to-end via syslog. Every log line
 * within a single ask/task is tagged with the same run_id.
 *
 * Trace status codes:
 *   0 = running
 *   1 = ok (completed successfully)
 *   2 = fail (LLM or tool error)
 *   3 = timeout (iteration limit reached)
 */

#include <stdint.h>
#include <inttypes.h>
#include <syslog.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Trace status codes */
#define AGENT_TRACE_RUNNING 0
#define AGENT_TRACE_OK 1
#define AGENT_TRACE_FAIL 2
#define AGENT_TRACE_TIMEOUT 3

typedef struct {
    char run_id[17]; /* 16 hex chars + NUL */
    char chat_id[64];
    char channel[16];
    uint32_t start_ts; /* epoch seconds */
    int iteration; /* current iteration number */
    int backend_idx; /* current router backend index */
    uint32_t total_latency_ms; /* cumulative LLM latency */
    int total_tool_calls; /* cumulative tool call count */
    int status; /* AGENT_TRACE_* */
} agent_trace_t;

/* A bounded, stateless timing ledger shared by Agent and voice ownership.
 * Stages deliberately carry only a request ID, monotonic timestamp, numeric
 * ordinal and result; no prompt, tool arguments, audio or credentials. */
/* Optional numeric observer supplied by an engineering product adapter.
 * It must be bounded and must not call back into the voice owner. No audio,
 * text or backend configuration crosses this interface. Times share the AP
 * monotonic clock. */
extern void agent_audio_observe(uint64_t request, uint32_t kind,
    uint64_t begin, uint64_t end, uint32_t bytes, uint32_t level)
    __attribute__((weak));

static inline void agent_trace_audio(uint64_t request, uint32_t kind,
    uint64_t begin, uint64_t end, uint32_t bytes, uint32_t level)
{
    if (agent_audio_observe)
        agent_audio_observe(request, kind, begin, end, bytes, level);
}

static inline void agent_trace_stage(uint64_t request_id, const char *stage,
    uint64_t value, int result)
{
    struct timespec now = {0};
    uint64_t monotonic_ms = 0;

    if (clock_gettime(CLOCK_MONOTONIC, &now) == 0) {
        monotonic_ms = (uint64_t)now.tv_sec * 1000u +
            (uint64_t)now.tv_nsec / 1000000u;
    }
    syslog(LOG_INFO, "[voice-stage] request=%" PRIu64
        " mono_ms=%" PRIu64 " stage=%s value=%" PRIu64 " result=%d\n",
        request_id, monotonic_ms, stage ? stage : "unknown", value, result);
}

/**
 * Begin a new trace. Generates a unique run_id and logs the start.
 */
void agent_trace_begin(agent_trace_t* t, const char* chat_id,
    const char* channel);

/**
 * Log one iteration step (after each LLM call + tool execution).
 */
void agent_trace_step(agent_trace_t* t, int iteration,
    const char* tool_name, uint32_t latency_ms,
    int llm_ok);

/**
 * End the trace and log the summary.
 */
void agent_trace_end(agent_trace_t* t, int status);

#ifdef __cplusplus
}
#endif
