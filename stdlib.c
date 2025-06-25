/*
 * This file is licensed under the GPL-3 License (or AGPL-3 if applicable)
 * Copyright (C) 2025  MikaSukie (old user), MikaLorielle (alt user), EmikaMai (current user), JaydenFreeman (legal name)

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <inttypes.h>
#include <ctype.h>

#define BUF_ALLOC(size) (char*)malloc(size)
#define FORMAT_INT_FUNC(name, type, fmt, bufsize) \
    char* name(type val) { \
        char* buf = BUF_ALLOC(bufsize); \
        if (!buf) return NULL; \
        snprintf(buf, bufsize, fmt, val); \
        return buf; \
    }

extern int64_t orcat_argc_global;
extern char **orcat_argv_global;

uintptr_t Cmalloc(int size) {
    return (uintptr_t)malloc((size_t)size);
}

void Cfree(uintptr_t ptr) {
    free((void*)ptr);
}

FORMAT_INT_FUNC(i64tostr, int64_t, "%" PRId64, 32)
FORMAT_INT_FUNC(i32tostr, int32_t, "%" PRId32, 16)
FORMAT_INT_FUNC(i16tostr, int16_t, "%" PRId16, 8)
FORMAT_INT_FUNC(i8tostr,  int8_t,  "%" PRId8,  8)

char* ftostr(double f) {
    char* buf = BUF_ALLOC(64);
    if (!buf) return NULL;
    snprintf(buf, 64, "%f", f);
    return buf;
}

char* btostr(bool b) {
    return strdup(b ? "true" : "false");
}

char* tostr(const char* s) {
    return strdup(s);
}

void free_str(char* s) {
    free(s);
}

void print(const char* s) {
    fputs(s, stdout);
}

void println(const char* s) {
    puts(s);
}

void eprint(const char* s) {
    fputs(s, stderr);
}

static char* concat_and_free(char* a, char* b) {
    size_t la = strlen(a);
    size_t lb = strlen(b);
    char* out = BUF_ALLOC(la + lb + 1);
    if (!out) {
        free(a);
        free(b);
        return NULL;
    }
    memcpy(out, a, la);
    memcpy(out + la, b, lb + 1);
    free(a);
    free(b);
    return out;
}

char* sb_create() {
    return strdup("");
}

char* sb_append_str(char* builder, const char* s) {
    return concat_and_free(builder, strdup(s));
}

char* itostr(int i) {
    return i64tostr((int64_t)i);
}

char* sb_append_int(char* builder, int x) {
    char* num = i64tostr(x);
    return concat_and_free(builder, num);
}

char* sb_append_float(char* builder, double f) {
    char* num = ftostr(f);
    return concat_and_free(builder, num);
}

char* sb_append_bool(char* builder, bool bb) {
    char* boo = btostr(bb);
    return concat_and_free(builder, boo);
}

char* sb_finish(char* builder) {
    return builder;
}

char* input(const char* prompt) {
    if (prompt) {
        fputs(prompt, stdout);
        fflush(stdout);
    }

    size_t cap = 128;
    char* buf = BUF_ALLOC(cap);
    if (!buf) return NULL;

    if (!fgets(buf, (int)cap, stdin)) {
        free(buf);
        return NULL;
    }

    size_t len = strlen(buf);
    if (len && buf[len - 1] == '\n') buf[len - 1] = '\0';

    return buf;
}

int64_t iinput(const char* prompt) {
    char* s = input(prompt);
    if (!s) return 0;
    int64_t val = strtoll(s, NULL, 10);
    free(s);
    return val;
}

double finput(const char* prompt) {
    char* s = input(prompt);
    if (!s) return 0.0;
    double val = strtod(s, NULL);
    free(s);
    return val;
}

bool binput(const char* prompt) {
    char* s = input(prompt);
    if (!s) return false;
    bool result = (strcmp(s, "true") == 0 || strcmp(s, "1") == 0);
    free(s);
    return result;
}

char* sinput(const char* prompt) {
    return input(prompt);
}

int ilength(int x) {
    if (x == 0) return 1;
    int len = (x < 0) ? 1 : 0;
    while (x) {
        len++;
        x /= 10;
    }
    return len;
}

int flength(double f) {
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%f", f);
    if (len < 0) return 0;

    while (len > 0 && buf[len - 1] == '0') len--;
    if (len > 0 && buf[len - 1] == '.') len--;
    return len;
}

int slength(const char* s) {
    return s ? (int)strlen(s) : 0;
}

char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return strdup("");

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);

    char* buf = BUF_ALLOC(len + 1);
    if (!buf) {
        fclose(f);
        return strdup("");
    }

    size_t r = fread(buf, 1, len, f);
    buf[r] = '\0';
    fclose(f);
    return buf;
}

bool write_file(const char* path, const char* content) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    size_t len = strlen(content);
    bool success = fwrite(content, 1, len, f) == len;
    fclose(f);
    return success;
}

bool append_file(const char* path, const char* content) {
    FILE* f = fopen(path, "ab");
    if (!f) return false;
    size_t len = strlen(content);
    bool success = fwrite(content, 1, len, f) == len;
    fclose(f);
    return success;
}

bool file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

char* read_lines(const char* path) {
    return read_file(path);
}

bool streq(const char* a, const char* b) {
    return strcmp(a, b) == 0;
}

double todouble(int x) {
    return (double)x;
}

int toint(double x) {
    return (int)x;
}

int rtoint(double x) {
    return (int)(x + (x >= 0 ? 0.5 : -0.5));
}

char* rmtrz(double val) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.15g", val);

    char* dot = strchr(buf, '.');
    if (dot) {
        char* end = buf + strlen(buf) - 1;
        while (end > dot && *end == '0') *end-- = '\0';
        if (end == dot) *end = '\0';
    }

    return strdup(buf);
}
bool contains(const char* str, const char* substr) {
    if (!str || !substr) return false;
    return strstr(str, substr) != NULL;
}
int countcontain(const char* str, const char* substr) {
    if (!str || !substr || !*substr) return 0;

    int count = 0;
    const char* temp = str;

    while ((temp = strstr(temp, substr)) != NULL) {
        count++;
        temp += strlen(substr);
    }

    return count;
}

char* tac(const char* s) {
    if (!s) return NULL;
    char* result = strdup(s);
    if (!result) return NULL;

    for (char* p = result; *p; ++p) {
        *p = (char)toupper((unsigned char)*p);
    }

    return result;
}

char* tal(const char* s) {
    if (!s) return NULL;
    char* result = strdup(s);
    if (!result) return NULL;

    for (char* p = result; *p; ++p) {
        *p = (char)tolower((unsigned char)*p);
    }

    return result;
}

int64_t orcat_argc(void) {
    return orcat_argc_global;
}

char *orcat_argv(int64_t idx) {
    if (idx < 0 || idx >= orcat_argc_global) return "null";
    return orcat_argv_global[idx] ? orcat_argv_global[idx] : "null";
}

const char* get_os() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#elif defined(__unix__)
    return "unix";
#else
    return "unknown";
#endif
}
