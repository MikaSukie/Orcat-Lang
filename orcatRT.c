#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

char* itostr(int x) {
    char* buf = (char*)malloc(32);
    if (!buf) return NULL;
    sprintf(buf, "%d", x);
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

char* tostr(char* s) {
    return _strdup(s);
}

void print(char* s) {
    printf("%s", s);
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
    return concat_and_free(builder, s);
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
