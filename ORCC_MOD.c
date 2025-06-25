/*
 * This file is licensed under the GPL-3 License (or AGPL-3 if applicable)
 * 
 * Copyright (c) 2025 MikaSukie, MikaLorielle, EmikaMai
 * 
 * This project is maintained by MikaSukie (old username), MikaLorielle (recent username),
 * EmikaMai (most recent username), and Jayden Freeman (legal name). 
 * All contributions are made under the terms of the GPL-3 License. 
 * See LICENSE file for more details.
 */
// now deprecated. will be removed.
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>

uintptr_t Cmalloc(int size) {
    return (uintptr_t)malloc((size_t)size);
}

void Cfree(uintptr_t ptr) {
    free((void*)ptr);
}

char* itostr(int64_t x) {
    char* buf = malloc(32);
    if (!buf) return NULL;
    snprintf(buf, 32, "%" PRId64, x);
    return buf;
}

char* i64tostr(int64_t val) {
    char* buf = malloc(32);
    if (!buf) return NULL;
    snprintf(buf, 32, "%" PRId64, val);
    return buf;
}

char* i32tostr(int32_t val) {
    char* buf = malloc(16);
    if (!buf) return NULL;
    snprintf(buf, 16, "%" PRId32, val);
    return buf;
}

char* i16tostr(int16_t val) {
    char* buf = malloc(8);
    if (!buf) return NULL;
    snprintf(buf, 8, "%" PRId16, val);
    return buf;
}

char* i8tostr(int8_t val) {
    char* buf = malloc(8);
    if (!buf) return NULL;
    snprintf(buf, 8, "%" PRId8, val);
    return buf;
}

char* ftostr(double f) {
    char* buf = (char*)malloc(64);
    if (!buf) return NULL;
    sprintf(buf, "%f", f);
    return buf;
}

char* btostr(bool b) {
    return _strdup(b ? "true" : "false");
}

char* tostr(const char* s) {
    return _strdup(s);
}

void free_str(char* s) {
    free(s);
}

void print(const char* s) {
    printf("%s", s);
}

void println(const char* s) {
    printf("%s\n", s);
}

void eprint(const char* s) {
    fprintf(stderr, "%s", s);
}

static char* concat_and_free(char* a, char* b) {
    size_t la = strlen(a);
    size_t lb = strlen(b);
    char* out = (char*)malloc(la + lb + 1);
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
    return _strdup("");
}

char* sb_append_str(char* builder, char* s) {
    return concat_and_free(builder, _strdup(s));
}

char* sb_append_int(char* builder, int x) {
    char* num = itostr(x);
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
    printf("%s", prompt);
    fflush(stdout);
    char buf[sizeof(prompt)];
    if (fgets(buf, sizeof(prompt), stdin) == NULL) return NULL;
    size_t len = strlen(buf);
    if (len && buf[len - 1] == '\n') buf[len - 1] = '\0';
    return _strdup(buf);
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
    double val = atof(s);
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
    int length = 0;
    if (x < 0) {
        length++;
        x = -x;
    }
    while (x > 0) {
        length++;
        x /= 10;
    }
    return length;
}

int flength(double f) {
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%f", f);
    if (len < 0) return 0;
    while (len > 0 && buf[len-1] == '0') len--;
    if (len > 0 && buf[len-1] == '.') len--;
    return len;
}

int slength(const char* s) {
    if (!s) return 0;
    return (int)strlen(s);
}

char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return _strdup("");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);
    char* buf = (char*)malloc(len + 1);
    if (!buf) {
        fclose(f);
        return _strdup("");
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
    size_t w = fwrite(content, 1, len, f);
    fclose(f);
    return w == len;
}

bool append_file(const char* path, const char* content) {
    FILE* f = fopen(path, "ab");
    if (!f) return false;
    size_t len = strlen(content);
    size_t w = fwrite(content, 1, len, f);
    fclose(f);
    return w == len;
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
    if (x >= 0)
        return (int)(x + 0.5);
    else
        return (int)(x - 0.5);
}
char* rmtrz(double val) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.15g", val);
    char* dot = strchr(buf, '.');
    if (dot) {
        char* end = buf + strlen(buf) - 1;
        while (end > dot && *end == '0') {
            *end = '\0';
            end--;
        }
        if (end == dot) {
            *end = '\0';
        }
    }

    return _strdup(buf);
}
