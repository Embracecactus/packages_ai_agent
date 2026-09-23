/* SPDX-License-Identifier: Apache-2.0 */
#include "llm/llm_stream.h"
#include "cJSON.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define EVENT_LIMIT 16384u
struct llm_final_stream {
    char line[EVENT_LIMIT + 1];
    char event[EVENT_LIMIT + 1];
    size_t line_size, event_size, size, limit, capacity;
    char *text;
    llm_text_delta_t emit;
    void *context;
    int stopped, done, error;
};

static int valid_utf8(const unsigned char *s)
{
    while (*s) {
        unsigned int cp = *s++, n;
        if (cp < 0x80) continue;
        if (cp >= 0xc2 && cp <= 0xdf) { n = 1; cp &= 31; }
        else if (cp >= 0xe0 && cp <= 0xef) { n = 2; cp &= 15; }
        else if (cp >= 0xf0 && cp <= 0xf4) { n = 3; cp &= 7; }
        else return 0;
        unsigned int count = n;
        while (n--) {
            if ((*s & 0xc0) != 0x80) return 0;
            cp = (cp << 6) | (*s++ & 63);
        }
        if ((count == 2 && cp < 0x800) || (count == 3 && cp < 0x10000) ||
            cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return 0;
    }
    return 1;
}

static int dispatch(struct llm_final_stream *p)
{
    if (!p->event_size) return 0;
    p->event[p->event_size] = 0;
    if (p->done) return -EPROTO;
    if (!strcmp(p->event, "[DONE]")) {
        if (!p->stopped || !p->size) return -EPROTO;
        p->done = 1;
        return 0;
    }
    /* Embedded NUL escapes could otherwise silently truncate a JSON string. */
    if (strstr(p->event, "\\u0000")) return -EBADMSG;
    cJSON *root = cJSON_ParseWithOpts(p->event, NULL, 1);
    if (!root) return -EBADMSG;
    int ret = -EPROTO;
    cJSON *choices = cJSON_GetObjectItemCaseSensitive(root, "choices");
    if (!cJSON_IsObject(root) || cJSON_GetObjectItemCaseSensitive(root, "error") ||
        !cJSON_IsArray(choices)) goto out;
    /* A usage-only event may follow the final choice. */
    if (!cJSON_GetArraySize(choices)) { ret = 0; goto out; }
    if (p->stopped || cJSON_GetArraySize(choices) != 1) goto out;
    cJSON *choice = cJSON_GetArrayItem(choices, 0);
    cJSON *index = cJSON_GetObjectItemCaseSensitive(choice, "index");
    cJSON *delta = cJSON_GetObjectItemCaseSensitive(choice, "delta");
    cJSON *finish = cJSON_GetObjectItemCaseSensitive(choice, "finish_reason");
    if (!cJSON_IsNumber(index) || index->valuedouble != 0 || !cJSON_IsObject(delta)) goto out;
    if (cJSON_GetObjectItemCaseSensitive(delta, "tool_calls") ||
        cJSON_GetObjectItemCaseSensitive(delta, "function_call")) goto out;
    cJSON *role = cJSON_GetObjectItemCaseSensitive(delta, "role");
    if (role && (!cJSON_IsString(role) || strcmp(role->valuestring, "assistant"))) goto out;
    if (finish && !cJSON_IsNull(finish) &&
        (!cJSON_IsString(finish) || strcmp(finish->valuestring, "stop"))) goto out;
    cJSON *content = cJSON_GetObjectItemCaseSensitive(delta, "content");
    if (content && !cJSON_IsNull(content)) {
        if (!cJSON_IsString(content) || !valid_utf8((unsigned char *)content->valuestring)) goto out;
        size_t n = strlen(content->valuestring);
        if (n > p->limit - p->size) { ret = -E2BIG; goto out; }
        if (n) {
            size_t needed = p->size + n + 1;
            if (needed > p->capacity) {
                size_t capacity = p->capacity;
                while (capacity < needed && capacity < p->limit + 1)
                    capacity = capacity > (p->limit + 1) / 2 ?
                        p->limit + 1 : capacity * 2;
                char *text = realloc(p->text, capacity);
                if (!text) { ret = -ENOMEM; goto out; }
                p->text = text;
                p->capacity = capacity;
            }
            memcpy(p->text + p->size, content->valuestring, n);
            p->size += n;
            p->text[p->size] = 0;
            ret = p->emit ? p->emit(p->context, content->valuestring, n) : 0;
            if (ret) goto out;
        }
    }
    /* reasoning_content is intentionally never emitted. */
    if (cJSON_IsString(finish)) p->stopped = 1;
    ret = 0;
out:
    cJSON_Delete(root);
    return ret;
}

llm_final_stream_t *llm_final_stream_new(llm_text_delta_t emit, void *context,
                                      size_t limit)
{
    if (!limit || limit > 1024u * 1024u) return NULL;
    struct llm_final_stream *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->capacity = limit < 4095 ? limit + 1 : 4096;
    p->text = calloc(1, p->capacity);
    if (!p->text) { free(p); return NULL; }
    p->limit = limit; p->emit = emit; p->context = context;
    return p;
}

int llm_final_stream_feed(void *parser, const char *bytes, size_t length)
{
    struct llm_final_stream *p = parser;
    if (!p || (!bytes && length)) return -EINVAL;
    if (p->error) return p->error;
    for (size_t i = 0; i < length; i++) {
        unsigned char c = bytes[i];
        if (!c) { p->error = -EBADMSG; break; }
        if (c != '\n') {
            if (p->line_size == EVENT_LIMIT) { p->error = -E2BIG; break; }
            p->line[p->line_size++] = c;
            continue;
        }
        if (p->line_size && p->line[p->line_size - 1] == '\r') p->line_size--;
        p->line[p->line_size] = 0;
        if (!p->line_size) {
            p->error = dispatch(p);
            p->event_size = 0;
        } else if (!strncmp(p->line, "data:", 5)) {
            size_t start = p->line[5] == ' ' ? 6 : 5;
            size_t n = p->line_size - start;
            size_t separator = p->event_size != 0;
            if (n + separator > EVENT_LIMIT - p->event_size) p->error = -E2BIG;
            else {
                if (separator) p->event[p->event_size++] = '\n';
                memcpy(p->event + p->event_size, p->line + start, n);
                p->event_size += n;
            }
        }
        p->line_size = 0;
        if (p->error) break;
    }
    return p->error;
}

int llm_final_stream_complete(const llm_final_stream_t *p)
{
    return p && p->done && !p->error && !p->line_size && !p->event_size;
}

int llm_final_stream_finish(llm_final_stream_t *p, char **text)
{
    if (!p || !text) return -EINVAL;
    *text = NULL;
    if (p->error) return p->error;
    if (!p->done || p->line_size || p->event_size) return -EPROTO;
    *text = p->text;
    p->text = NULL;
    return 0;
}

void llm_final_stream_free(llm_final_stream_t *p)
{
    if (!p) return;
    if (p->text) { memset(p->text, 0, p->capacity); free(p->text); }
    memset(p, 0, sizeof(*p));
    free(p);
}
