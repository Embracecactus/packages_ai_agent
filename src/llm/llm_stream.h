/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <stddef.h>

/* Final-body-only OpenAI SSE parser. The caller must have committed to a
 * tool-free final phase before feeding bytes. Any tool delta is an error.
 * Callback text is borrowed for the call; negative returns abort consumption.
 */
typedef int (*llm_text_delta_t)(void *context, const char *text, size_t length);
typedef struct llm_final_stream llm_final_stream_t;
llm_final_stream_t *llm_final_stream_new(llm_text_delta_t emit, void *context,
                                      size_t text_limit);
int llm_final_stream_feed(void *parser, const char *bytes, size_t length);
int llm_final_stream_complete(const llm_final_stream_t *parser);
/* Requires stop + DONE. Transfers complete text on success, never partial. */
int llm_final_stream_finish(llm_final_stream_t *parser, char **text);
void llm_final_stream_free(llm_final_stream_t *parser);
