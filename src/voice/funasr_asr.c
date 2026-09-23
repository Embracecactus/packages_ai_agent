/* SPDX-License-Identifier: Apache-2.0 */
/* FunASR public 2pass WebSocket protocol over an injected verified byte stream. */
#include "voice/funasr_asr.h"
#include "voice/voice_asr.h"
#include "agent_config.h"
#include "cJSON.h"
#include "mbedtls/base64.h"

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define FUNASR_HOST_MAX 127
#define FUNASR_PATH_MAX 255
#define FUNASR_HEADER_MAX 2048
#define FUNASR_MESSAGE_MAX 4096
#define FUNASR_PCM_FRAME_MAX 4096
#define FUNASR_PCM_QUEUE_FRAMES 2
#define FUNASR_RESULT_MAX 512
#define FUNASR_CONNECT_MS 10000u
#define FUNASR_TURN_MS 30000u

typedef struct {
    funasr_asr_transport_t io;
    void* context;
    char host[FUNASR_HOST_MAX + 1];
    char path[FUNASR_PATH_MAX + 1];
    uint16_t port;
    int configured;
} funasr_config_t;

typedef struct {
    funasr_config_t cfg;
    pthread_mutex_t tx_lock;
    pthread_mutex_t queue_lock;
    pthread_cond_t queue_changed;
    pthread_t receiver;
    pthread_t sender;
    int receiver_started;
    int sender_started;
    int opened;
    uint8_t pcm_queue[FUNASR_PCM_QUEUE_FRAMES][FUNASR_PCM_FRAME_MAX];
    size_t pcm_sizes[FUNASR_PCM_QUEUE_FRAMES];
    unsigned queue_head;
    unsigned queue_count;
    int producer_done;
    volatile int canceled;
    volatile int error;
    volatile int final_ready;
    volatile int finish_requested;
    char final_text[FUNASR_RESULT_MAX];
    uint64_t deadline;
} funasr_session_t;

static pthread_mutex_t s_lock = PTHREAD_MUTEX_INITIALIZER;
static funasr_config_t s_config;
static funasr_session_t* s_active;
static funasr_session_t* s_retained;
static int s_canceled;

static int canceled(const funasr_session_t* s)
{ return __atomic_load_n(&s->canceled, __ATOMIC_ACQUIRE); }

static int remaining(const funasr_session_t* s)
{
    if (canceled(s)) return -ECANCELED;
    return s->cfg.io.now_ms(s->cfg.context) >= s->deadline ? -ETIMEDOUT : 0;
}

static int send_all(funasr_session_t* s, const uint8_t* bytes, size_t len)
{
    for (size_t off = 0; off < len;) {
        int ret = remaining(s);
        if (ret) return ret;
        ssize_t n = s->cfg.io.send(s->cfg.context, bytes + off, len - off,
            s->deadline);
        if (n <= 0 || (size_t)n > len - off) return n < 0 ? (int)n : -EPIPE;
        off += (size_t)n;
    }
    return 0;
}

static int recv_exact(funasr_session_t* s, uint8_t* bytes, size_t len)
{
    for (size_t off = 0; off < len;) {
        int ret = remaining(s);
        if (ret) return ret;
        ssize_t n = s->cfg.io.recv(s->cfg.context, bytes + off, len - off,
            s->deadline);
        if (n <= 0 || (size_t)n > len - off) return n < 0 ? (int)n : -ECONNRESET;
        off += (size_t)n;
    }
    return 0;
}

static int token_header(const char* value, const char* token)
{
    size_t target = strlen(token);
    while (*value) {
        while (*value == ' ' || *value == '\t' || *value == ',') value++;
        const char* end = value;
        while (*end && *end != ',') end++;
        const char* trimmed = end;
        while (trimmed > value && (trimmed[-1] == ' ' || trimmed[-1] == '\t')) trimmed--;
        if ((size_t)(trimmed - value) == target &&
            strncasecmp(value, token, target) == 0) return 1;
        value = end;
        if (*value) value++;
    }
    return 0;
}

static int ws_upgrade(funasr_session_t* s)
{
    uint8_t nonce[16], digest[20], key[32], accept[32];
    size_t key_len = 0, accept_len = 0;
    static const char guid[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    if (s->cfg.io.random(s->cfg.context, nonce, sizeof(nonce)) != 0) return -EIO;
    if (mbedtls_base64_encode(key, sizeof(key), &key_len, nonce,
        sizeof(nonce)) != 0) return -EIO;
    char challenge[sizeof(key) + sizeof(guid)];
    memcpy(challenge, key, key_len);
    memcpy(challenge + key_len, guid, sizeof(guid) - 1);
    if (s->cfg.io.sha1(s->cfg.context, (const uint8_t*)challenge,
        key_len + sizeof(guid) - 1, digest) != 0) return -EIO;
    if (mbedtls_base64_encode(accept, sizeof(accept), &accept_len, digest,
        sizeof(digest)) != 0) return -EIO;

    char req[FUNASR_HEADER_MAX];
    int n = snprintf(req, sizeof(req),
        "GET %s HTTP/1.1\r\nHost: %s:%u\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Key: %.*s\r\n"
        "Sec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: binary\r\n\r\n",
        s->cfg.path, s->cfg.host, (unsigned)s->cfg.port, (int)key_len, key);
    if (n <= 0 || n >= (int)sizeof(req)) return -EOVERFLOW;
    int ret = send_all(s, (const uint8_t*)req, (size_t)n);
    if (ret) return ret;

    char resp[FUNASR_HEADER_MAX + 1];
    size_t used = 0;
    do {
        if (used == FUNASR_HEADER_MAX) return -EOVERFLOW;
        ret = recv_exact(s, (uint8_t*)&resp[used], 1);
        if (ret) return ret;
        used++;
    } while (used < 4 || memcmp(resp + used - 4, "\r\n\r\n", 4) != 0);
    resp[used] = 0;
    char* status_end = strstr(resp, "\r\n");
    if (!status_end || status_end - resp < 12 ||
        memcmp(resp, "HTTP/1.1 101", 12) != 0 ||
        (status_end != resp + 12 && resp[12] != ' ')) return -EPROTO;
    for (char* p = resp + 12; p < status_end; p++)
        if ((unsigned char)*p < 0x20 || *p == 0x7f) return -EPROTO;

    int upgrade = 0, connection = 0, accepted = 0, protocol = 1;
    char* line = strstr(resp, "\r\n");
    while (line && line[2] != '\r') {
        line += 2;
        char* end = strstr(line, "\r\n");
        if (!end) return -EPROTO;
        *end = 0;
        char* colon = strchr(line, ':');
        if (!colon) return -EPROTO;
        *colon++ = 0;
        while (*colon == ' ' || *colon == '\t') colon++;
        if (strcasecmp(line, "Upgrade") == 0)
            upgrade = token_header(colon, "websocket");
        else if (strcasecmp(line, "Connection") == 0)
            connection = token_header(colon, "Upgrade");
        else if (strcasecmp(line, "Sec-WebSocket-Accept") == 0)
            accepted = strlen(colon) == accept_len &&
                memcmp(colon, accept, accept_len) == 0;
        else if (strcasecmp(line, "Sec-WebSocket-Protocol") == 0)
            protocol = strcmp(colon, "binary") == 0;
        line = end;
    }
    return upgrade && connection && accepted && protocol ? 0 : -EPROTO;
}

/* The bounded PCM sender and receiver-side pong share the frame write lock. */
static int ws_send(funasr_session_t* s, uint8_t opcode,
    const uint8_t* data, size_t len)
{
    if (len > FUNASR_PCM_FRAME_MAX || (opcode & 8 && len > 125)) return -EOVERFLOW;
    uint8_t header[8], mask[4], scratch[1024];
    size_t hlen = 2;
    header[0] = 0x80 | opcode;
    if (len < 126) header[1] = 0x80 | (uint8_t)len;
    else {
        header[1] = 0x80 | 126;
        header[2] = (uint8_t)(len >> 8);
        header[3] = (uint8_t)len;
        hlen = 4;
    }
    int ret = s->cfg.io.random(s->cfg.context, mask, sizeof(mask));
    if (ret) return -EIO;
    memcpy(header + hlen, mask, sizeof(mask));
    hlen += sizeof(mask);
    pthread_mutex_lock(&s->tx_lock);
    ret = send_all(s, header, hlen);
    for (size_t off = 0; !ret && off < len;) {
        size_t count = len - off;
        if (count > sizeof(scratch)) count = sizeof(scratch);
        for (size_t i = 0; i < count; i++)
            scratch[i] = data[off + i] ^ mask[(off + i) & 3];
        ret = send_all(s, scratch, count);
        off += count;
    }
    pthread_mutex_unlock(&s->tx_lock);
    return ret;
}

static int valid_utf8(const uint8_t* p, size_t n)
{
    for (size_t i = 0; i < n;) {
        uint8_t c = p[i++];
        if (c == 0) return 0;
        if (c < 0x80) continue;
        unsigned count = c >= 0xc2 && c <= 0xdf ? 1 :
            c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : 0;
        if (!count || i + count > n) return 0;
        if ((c == 0xe0 && p[i] < 0xa0) || (c == 0xed && p[i] >= 0xa0) ||
            (c == 0xf0 && p[i] < 0x90) || (c == 0xf4 && p[i] >= 0x90)) return 0;
        while (count--) if ((p[i++] & 0xc0) != 0x80) return 0;
    }
    return 1;
}

static int parse_result(funasr_session_t* s, uint8_t* data, size_t len)
{
    if (!valid_utf8(data, len)) return -EPROTO;
    data[len] = 0;
    if (strstr((char*)data, "\\u0000")) return -EPROTO;
    cJSON* root = cJSON_ParseWithOpts((char*)data, NULL, 1);
    if (!root) return -EPROTO;
    cJSON* mode = cJSON_GetObjectItemCaseSensitive(root, "mode");
    cJSON* text = cJSON_GetObjectItemCaseSensitive(root, "text");
    cJSON* end = cJSON_GetObjectItemCaseSensitive(root, "is_end");
    cJSON* error = cJSON_GetObjectItemCaseSensitive(root, "error");
    cJSON* code = cJSON_GetObjectItemCaseSensitive(root, "code");
    int ret = 0;
    if ((error && !cJSON_IsNull(error)) ||
        (cJSON_IsNumber(code) && code->valueint != 0)) ret = -EPROTO;
    else if (cJSON_IsString(mode) && strcmp(mode->valuestring, "2pass-offline") == 0) {
        if (!cJSON_IsString(text)) ret = -EPROTO;
        else {
            size_t previous = strlen(s->final_text);
            size_t added = strlen(text->valuestring);
            if (added > FUNASR_RESULT_MAX - previous - 1)
                ret = -EOVERFLOW;
            else memcpy(s->final_text + previous, text->valuestring, added + 1);
        }
        if (!ret) {
            __atomic_store_n(&s->final_ready, 1, __ATOMIC_RELEASE);
        }
    } else if (cJSON_IsString(mode) &&
               strcmp(mode->valuestring, "2pass-online") == 0) {
        if (!cJSON_IsString(text)) ret = -EPROTO;
    } else if (!cJSON_IsTrue(end)) ret = -EPROTO;
    /* A VAD segment's is_final is not the end of the recording. Only the
     * explicit is_end acknowledgement completes this protocol variant. */
    if (!ret && cJSON_IsTrue(end)) {
        if (!__atomic_load_n(&s->finish_requested, __ATOMIC_ACQUIRE)) ret = -EPROTO;
        else ret = __atomic_load_n(&s->final_ready, __ATOMIC_ACQUIRE) &&
                   s->final_text[0] ? 1 : -ENODATA;
    }
    cJSON_Delete(root);
    return ret;
}

static int recv_message(funasr_session_t* s, uint8_t* message, size_t* used,
    int* fragmented)
{
    uint8_t header[2];
    int ret = recv_exact(s, header, sizeof(header));
    if (ret) return ret;
    if (header[0] & 0x70 || header[1] & 0x80) return -EPROTO;
    int fin = !!(header[0] & 0x80), opcode = header[0] & 0x0f;
    size_t len = header[1] & 0x7f;
    if (len == 126) {
        uint8_t ext[2];
        ret = recv_exact(s, ext, sizeof(ext));
        if (ret) return ret;
        len = ((size_t)ext[0] << 8) | ext[1];
    } else if (len == 127) return -EOVERFLOW;
    if (opcode & 8) {
        uint8_t control[125];
        if (!fin || len > sizeof(control)) return -EPROTO;
        ret = recv_exact(s, control, len);
        if (ret) return ret;
        if (opcode == 8) {
            if (len == 1) return -EPROTO;
            (void)ws_send(s, 8, control, len);
            return -ECONNRESET;
        }
        if (opcode == 9) return ws_send(s, 10, control, len);
        return opcode == 10 ? 0 : -EPROTO;
    }
    if (opcode == 1) {
        if (*fragmented) return -EPROTO;
        *used = 0;
    } else if (opcode != 0 || !*fragmented) return -EPROTO;
    if (len > FUNASR_MESSAGE_MAX - *used) return -EOVERFLOW;
    ret = recv_exact(s, message + *used, len);
    if (ret) return ret;
    *used += len;
    *fragmented = !fin;
    return fin ? 1 : 0;
}

static void* receiver_main(void* context)
{
    funasr_session_t* s = context;
    uint8_t message[FUNASR_MESSAGE_MAX + 1];
    size_t used = 0;
    int fragmented = 0, ret = 0;
    while (!ret) {
        ret = recv_message(s, message, &used, &fragmented);
        if (ret == 1) {
            ret = parse_result(s, message, used);
            used = 0;
            if (ret == 1) break;
        }
    }
    __atomic_store_n(&s->error, ret == 1 ? 0 : ret, __ATOMIC_RELEASE);
    return NULL;
}

static void* sender_main(void* context)
{
    funasr_session_t* s = context;
    uint8_t pcm[FUNASR_PCM_FRAME_MAX];
    for (;;) {
        pthread_mutex_lock(&s->queue_lock);
        while (!s->queue_count && !s->producer_done && !canceled(s))
            pthread_cond_wait(&s->queue_changed, &s->queue_lock);
        if (canceled(s) || (!s->queue_count && s->producer_done)) {
            pthread_mutex_unlock(&s->queue_lock);
            return NULL;
        }
        unsigned index = s->queue_head;
        size_t count = s->pcm_sizes[index];
        memcpy(pcm, s->pcm_queue[index], count);
        s->queue_head = (index + 1) % FUNASR_PCM_QUEUE_FRAMES;
        s->queue_count--;
        pthread_cond_broadcast(&s->queue_changed);
        pthread_mutex_unlock(&s->queue_lock);
        int ret = ws_send(s, 2, pcm, count);
        if (ret) {
            __atomic_store_n(&s->error, ret, __ATOMIC_RELEASE);
            pthread_mutex_lock(&s->queue_lock);
            pthread_cond_broadcast(&s->queue_changed);
            pthread_mutex_unlock(&s->queue_lock);
            return NULL;
        }
    }
}

static int start_worker(pthread_t* thread, void* (*entry)(void*),
    funasr_session_t* s)
{
    pthread_attr_t attr;
    int ret = pthread_attr_init(&attr);
    if (ret) return -ret;
    ret = pthread_attr_setstacksize(&attr, 24 * 1024);
    if (!ret) ret = pthread_create(thread, &attr, entry, s);
    pthread_attr_destroy(&attr);
    return ret ? -ret : 0;
}

static int close_session(funasr_session_t* s, int interrupt)
{
    if (interrupt) {
        __atomic_store_n(&s->canceled, 1, __ATOMIC_RELEASE);
        s->cfg.io.interrupt(s->cfg.context);
    }
    pthread_mutex_lock(&s->queue_lock);
    s->producer_done = 1;
    pthread_cond_broadcast(&s->queue_changed);
    pthread_mutex_unlock(&s->queue_lock);
    if (s->sender_started) pthread_join(s->sender, NULL);
    if (s->receiver_started) pthread_join(s->receiver, NULL);
    pthread_mutex_lock(&s_lock);
    if (s_active == s) s_active = NULL;
    int ret = s->opened ? s->cfg.io.close(s->cfg.context) : 0;
    if (ret < 0) s_retained = s;
    pthread_mutex_unlock(&s_lock);
    if (ret < 0) return ret;
    pthread_cond_destroy(&s->queue_changed);
    pthread_mutex_destroy(&s->queue_lock);
    pthread_mutex_destroy(&s->tx_lock);
    free(s);
    return 0;
}

static int backend_init(void)
{
    if (AGENT_VOICE_SAMPLE_RATE != 16000 || AGENT_VOICE_CHANNELS != 1 ||
        AGENT_VOICE_BITS != 16) return -ENOTSUP;
    pthread_mutex_lock(&s_lock);
    int ret = s_retained ? -EBUSY : s_config.configured ? 0 : -ENOKEY;
    pthread_mutex_unlock(&s_lock);
    return ret;
}

static int backend_prepare_request(void)
{
    pthread_mutex_lock(&s_lock);
    int ret = (s_active || s_retained) ? -EBUSY :
              !s_config.configured ? -ENOKEY : 0;
    if (!ret && s_config.io.prepare_request)
        ret = s_config.io.prepare_request(s_config.context);
    if (!ret) s_canceled = 0;
    pthread_mutex_unlock(&s_lock);
    return ret;
}

static voice_asr_stream_handle_t stream_open(void)
{
    funasr_session_t* s = calloc(1, sizeof(*s));
    if (!s) { errno = ENOMEM; return NULL; }
    pthread_mutex_init(&s->tx_lock, NULL);
    pthread_mutex_init(&s->queue_lock, NULL);
    pthread_cond_init(&s->queue_changed, NULL);
    pthread_mutex_lock(&s_lock);
    if (!s_config.configured || s_active || s_retained || s_canceled) {
        int error = (s_active || s_retained) ? EBUSY :
                    s_canceled ? ECANCELED : ENOKEY;
        pthread_mutex_unlock(&s_lock);
        pthread_cond_destroy(&s->queue_changed);
        pthread_mutex_destroy(&s->queue_lock);
        pthread_mutex_destroy(&s->tx_lock);
        free(s);
        errno = error;
        return NULL;
    }
    s->cfg = s_config;
    s_active = s;
    pthread_mutex_unlock(&s_lock);
    uint64_t now = s->cfg.io.now_ms(s->cfg.context);
    s->deadline = now + FUNASR_TURN_MS;
    uint64_t connect_deadline = now + FUNASR_CONNECT_MS;
    if (connect_deadline < s->deadline) s->deadline = connect_deadline;
    int ret = s->cfg.io.open_verified(s->cfg.context, s->cfg.host,
        s->cfg.port, s->deadline);
    if (ret == 0) {
        s->opened = 1;
        ret = remaining(s);
    }
    if (!ret) ret = ws_upgrade(s);
    static const char hello[] =
        "{\"mode\":\"2pass\",\"wav_name\":\"voice\",\"wav_format\":\"pcm\","
        "\"audio_fs\":16000,\"chunk_size\":[5,10,5],\"is_speaking\":true}";
    if (!ret) ret = ws_send(s, 1, (const uint8_t*)hello, sizeof(hello) - 1);
    if (!ret) {
        s->deadline = now + FUNASR_TURN_MS;
        ret = start_worker(&s->receiver, receiver_main, s);
        if (!ret) s->receiver_started = 1;
    }
    if (!ret) {
        ret = start_worker(&s->sender, sender_main, s);
        if (!ret) s->sender_started = 1;
    }
    if (ret) {
        int close_error = close_session(s, 1);
        errno = close_error < 0 ? -close_error : -ret;
        return NULL;
    }
    return s;
}

static int stream_send(voice_asr_stream_handle_t handle,
    const unsigned char* pcm, size_t len)
{
    funasr_session_t* s = handle;
    if (!s || (!pcm && len) || (len & 1)) return -EINVAL;
    int error = __atomic_load_n(&s->error, __ATOMIC_ACQUIRE);
    if (error) return error;
    for (size_t off = 0; off < len;) {
        size_t count = len - off;
        if (count > FUNASR_PCM_FRAME_MAX) count = FUNASR_PCM_FRAME_MAX;
        pthread_mutex_lock(&s->queue_lock);
        while (s->queue_count == FUNASR_PCM_QUEUE_FRAMES && !canceled(s) &&
               !__atomic_load_n(&s->error, __ATOMIC_ACQUIRE)) {
            if (remaining(s) != 0) break;
            struct timespec now;
            clock_gettime(CLOCK_REALTIME, &now);
            now.tv_nsec += 20000000;
            if (now.tv_nsec >= 1000000000) { now.tv_sec++; now.tv_nsec -= 1000000000; }
            (void)pthread_cond_timedwait(&s->queue_changed, &s->queue_lock, &now);
        }
        int ret = remaining(s);
        if (!ret) ret = __atomic_load_n(&s->error, __ATOMIC_ACQUIRE);
        if (!ret && s->producer_done) ret = -EPIPE;
        if (!ret) {
            unsigned tail = (s->queue_head + s->queue_count) % FUNASR_PCM_QUEUE_FRAMES;
            memcpy(s->pcm_queue[tail], pcm + off, count);
            s->pcm_sizes[tail] = count;
            s->queue_count++;
            pthread_cond_broadcast(&s->queue_changed);
        }
        pthread_mutex_unlock(&s->queue_lock);
        if (ret) return ret;
        off += count;
    }
    return 0;
}

static int stream_finish(voice_asr_stream_handle_t handle,
    char* text, size_t cap)
{
    funasr_session_t* s = handle;
    if (!s) return -EINVAL;
    if (!text || !cap) {
        int close_error = close_session(s, 1);
        return close_error < 0 ? close_error : -EINVAL;
    }
    text[0] = 0;
    pthread_mutex_lock(&s->queue_lock);
    s->producer_done = 1;
    pthread_cond_broadcast(&s->queue_changed);
    pthread_mutex_unlock(&s->queue_lock);
    pthread_join(s->sender, NULL);
    s->sender_started = 0;
    int upstream_error = __atomic_load_n(&s->error, __ATOMIC_ACQUIRE);
    static const char end[] = "{\"is_speaking\":false,\"is_end\":true}";
    __atomic_store_n(&s->finish_requested, 1, __ATOMIC_RELEASE);
    int ret = upstream_error ? upstream_error :
        ws_send(s, 1, (const uint8_t*)end, sizeof(end) - 1);
    if (!ret) {
        pthread_join(s->receiver, NULL);
        s->receiver_started = 0;
        ret = __atomic_load_n(&s->error, __ATOMIC_ACQUIRE);
        if (!ret && !__atomic_load_n(&s->final_ready, __ATOMIC_ACQUIRE))
            ret = -ENODATA;
        if (!ret && strlen(s->final_text) >= cap) ret = -EOVERFLOW;
        if (!ret) strcpy(text, s->final_text);
    }
    int close_error = close_session(s, ret != 0);
    return close_error < 0 ? close_error : ret;
}

static void stream_abort(voice_asr_stream_handle_t handle)
{ if (handle) (void)close_session(handle, 1); }

static int backend_cancel(void)
{
    pthread_mutex_lock(&s_lock);
    s_canceled = 1;
    funasr_session_t* s = s_active;
    if (s) __atomic_store_n(&s->canceled, 1, __ATOMIC_RELEASE);
    if (s) {
        pthread_mutex_lock(&s->queue_lock);
        pthread_cond_broadcast(&s->queue_changed);
        pthread_mutex_unlock(&s->queue_lock);
    }
    int ret = s_config.configured ?
        s_config.io.interrupt(s_config.context) : 0;
    pthread_mutex_unlock(&s_lock);
    return ret;
}

static int recognize(const unsigned char* pcm, size_t len,
    char* text, size_t cap)
{
    voice_asr_stream_handle_t s = stream_open();
    if (!s) return -errno;
    int ret = stream_send(s, pcm, len);
    if (ret) { stream_abort(s); return ret; }
    return stream_finish(s, text, cap);
}

static const voice_asr_ops_t s_ops = {
    .name = "funasr", .init = backend_init, .recognize = recognize,
    .prepare_request = backend_prepare_request,
    .cancel = backend_cancel, .stream_open = stream_open,
    .stream_send = stream_send, .stream_finish = stream_finish,
    .stream_abort = stream_abort,
};

int funasr_asr_configure(const funasr_asr_transport_t* io, void* context,
    const char* host, uint16_t port, const char* path)
{
    if (!io || !io->open_verified || !io->send || !io->recv ||
        !io->interrupt || !io->close || !io->random || !io->sha1 ||
        !io->now_ms || !host || !path || !port) return -EINVAL;
    size_t hn = strnlen(host, FUNASR_HOST_MAX + 1);
    size_t pn = strnlen(path, FUNASR_PATH_MAX + 1);
    if (!hn || hn > FUNASR_HOST_MAX || pn < 1 || pn > FUNASR_PATH_MAX ||
        path[0] != '/') return -EINVAL;
    for (size_t i = 0; i < hn; i++) {
        char c = host[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '.' || c == '-')) return -EINVAL;
    }
    for (size_t i = 0; i < pn; i++) {
        char c = path[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || strchr("/-._~?&=%+", c))) return -EINVAL;
    }
    pthread_mutex_lock(&s_lock);
    if (s_active || s_retained) { pthread_mutex_unlock(&s_lock); return -EBUSY; }
    s_config = (funasr_config_t){ .io = *io, .context = context, .port = port,
        .configured = 1 };
    memcpy(s_config.host, host, hn + 1);
    memcpy(s_config.path, path, pn + 1);
    pthread_mutex_unlock(&s_lock);
    return 0;
}

int funasr_asr_register(void)
{ return voice_asr_register(&s_ops); }

int funasr_asr_recover(void)
{
    pthread_mutex_lock(&s_lock);
    if (s_active) { pthread_mutex_unlock(&s_lock); return -EBUSY; }
    funasr_session_t* s = s_retained;
    if (!s) { pthread_mutex_unlock(&s_lock); return 0; }
    int ret = s->cfg.io.close(s->cfg.context);
    if (ret == 0) s_retained = NULL;
    pthread_mutex_unlock(&s_lock);
    if (ret < 0) return ret;
    pthread_cond_destroy(&s->queue_changed);
    pthread_mutex_destroy(&s->queue_lock);
    pthread_mutex_destroy(&s->tx_lock);
    free(s);
    return 0;
}
