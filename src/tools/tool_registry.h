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

/*
 * This file contains code derived from MimiClaw (https://github.com/memovai/mimiclaw)
 * Copyright (c) 2026 Ziboyan Wang, licensed under the MIT License.
 * See NOTICE file for the original MIT License terms.
 */

#pragma once

#include "agent_compat.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* name;
    const char* description;
    const char* input_schema_json;
    int (*execute)(const char* input_json, char* output, size_t output_size);
} agent_tool_t;

/* ── Schema builder macros (compile-time JSON Schema generation) ─── */

#define TOOL_SCHEMA_BEGIN() "{\"type\":\"object\",\"properties\":{"

#define TOOL_SCHEMA_END() "},\"required\":[]}"

#define TOOL_SCHEMA_END_REQUIRED(required_list) \
    "},\"required\":[" required_list "]}"

#define TOOL_NO_PARAMS "{\"type\":\"object\",\"properties\":{},\"required\":[]}"

#define TOOL_PARAM_STR(name, desc) \
    "\"" name "\":{\"type\":\"string\",\"description\":\"" desc "\"}"

#define TOOL_PARAM_NUM(name, desc) \
    "\"" name "\":{\"type\":\"number\",\"description\":\"" desc "\"}"

#define TOOL_PARAM_BOOL(name, desc) \
    "\"" name "\":{\"type\":\"boolean\",\"description\":\"" desc "\"}"

#define TOOL_PARAM_ENUM(name, desc, enum_values)                      \
    "\"" name "\":{\"type\":\"string\",\"description\":\"" desc "\"," \
    "\"enum\":[" enum_values "]}"

/* ── Registration macros ───────────────────────────────────────────── */

#define REGISTER_TOOL(tname, desc, schema, fn) \
    do {                                       \
        agent_tool_t _tool_##fn = {         \
            .name = tname,                     \
            .description = desc,               \
            .input_schema_json = schema,       \
            .execute = fn,                     \
        };                                     \
        register_tool(&_tool_##fn);            \
    } while (0)

#define REGISTER_TOOL_NO_PARAMS(tname, desc, fn) \
    REGISTER_TOOL(tname, desc, TOOL_NO_PARAMS, fn)

/* ── External tool provider callback (breaks circular dependency) ──── */

/**
 * Callback type for external tool providers.
 * Returns a JSON array string of tools (caller must free), or NULL if none.
 */
typedef char* (*tool_provider_fn)(void);

/**
 * Callback type for external tool executors.
 * Returns OK if tool was found and executed, ERROR otherwise.
 */
typedef int (*tool_executor_fn)(const char* name, const char* input_json,
                                char* output, size_t output_size);

/* 请求上下文只在同步执行期间借用；检查函数仍由原请求的 owner 提供。
 * 长耗时工具应在各阶段及子请求中复用检查，不另建取消状态。
 */
typedef int (*tool_executor_checked_fn)(const char* name, const char* input_json,
    char* output, size_t output_size, int (*check)(void*), void* request_context);

/**
 * Register an external tool provider (e.g., node_manager, mcp_client).
 * The provider's get_tools_json callback will be called during tools JSON build.
 * The executor callback will be called as fallback during tool execution.
 * Max 4 providers supported.
 */
void tool_registry_register_provider(const char* name,
                                     tool_provider_fn get_tools,
                                     tool_executor_fn execute);
void tool_registry_register_provider_checked(const char* name,
    tool_provider_fn get_tools, tool_executor_checked_fn execute);

int   tool_registry_init(void);
char *tool_registry_get_tools_json(void);  /* caller must free() */
void  tool_registry_rebuild_json(void);    /* force rebuild tools JSON now */
void  tool_registry_invalidate(void);      /* mark tools dirty for rebuild */
void  tool_registry_cleanup(void);         /* free tools JSON cache */
bool  tool_registry_has_tool(const char *name);
int   tool_registry_execute(const char *name, const char *input_json,
                                   char *output, size_t output_size);
int   tool_registry_execute_checked(const char *name, const char *input_json,
    char *output, size_t output_size, int (*check)(void*), void *request_context);

#ifdef __cplusplus
}
#endif
