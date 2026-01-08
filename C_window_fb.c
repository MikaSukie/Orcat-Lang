#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static GLFWwindow* g_window = NULL;
static int fb_w = 0, fb_h = 0;
static uint8_t* fb_pixels = NULL;
static GLuint fb_tex = 0;

static void ensure_texture() {
    if (!fb_tex) {
        glGenTextures(1, &fb_tex);
        glBindTexture(GL_TEXTURE_2D, fb_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fb_w, fb_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    }
}

static void framebuffer_size_cb(GLFWwindow* w, int width, int height) {
    (void)w;
    if (width <= 0 || height <= 0) return;
    fb_w = width;
    fb_h = height;
    free(fb_pixels);
    fb_pixels = malloc((size_t)fb_w * fb_h * 4);
    memset(fb_pixels, 0, (size_t)fb_w * fb_h * 4);
    if (fb_tex) {
        glBindTexture(GL_TEXTURE_2D, fb_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fb_w, fb_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, fb_pixels);
    }
}

bool win_create(int w, int h, const char* title) {
    if (g_window) return true;
    if (!glfwInit()) return false;
    g_window = glfwCreateWindow(w, h, title, NULL, NULL);
    if (!g_window) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(g_window);
    glfwSetFramebufferSizeCallback(g_window, framebuffer_size_cb);
    fb_w = w; fb_h = h;
    fb_pixels = malloc((size_t)fb_w * fb_h * 4);
    memset(fb_pixels, 0, (size_t)fb_w * fb_h * 4);
    ensure_texture();
    return true;
}

bool win_is_open() { return g_window && !glfwWindowShouldClose(g_window); }
void win_poll_events() { glfwPollEvents(); }
void win_get_size(int* w, int* h) { if (!g_window) { *w=*h=0; return; } glfwGetFramebufferSize(g_window, w, h); }
int win_get_width() { int w,h; win_get_size(&w,&h); return w; }
int win_get_height(){ int w,h; win_get_size(&w,&h); return h; }

void fb_clear_color(int r,int g,int b) {
    if (!fb_pixels) return;
    for (int y=0;y<fb_h;y++) for (int x=0;x<fb_w;x++){
        size_t i=((size_t)y*fb_w+x)*4;
        fb_pixels[i+0]=(uint8_t)r;
        fb_pixels[i+1]=(uint8_t)g;
        fb_pixels[i+2]=(uint8_t)b;
        fb_pixels[i+3]=255;
    }
}

void fb_set_pixel(int x,int y,int r,int g,int b){
    if(!fb_pixels||x<0||x>=fb_w||y<0||y>=fb_h)return;
    size_t i=((size_t)y*fb_w+x)*4;
    fb_pixels[i+0]=(uint8_t)r;
    fb_pixels[i+1]=(uint8_t)g;
    fb_pixels[i+2]=(uint8_t)b;
    fb_pixels[i+3]=255;
}

void fb_fill_rect(int x,int y,int w,int h,int r,int g,int b){
    if(!fb_pixels)return;
    int x0=x<0?0:x;
    int y0=y<0?0:y;
    int x1=x+w>fb_w?fb_w:x+w;
    int y1=y+h>fb_h?fb_h:y+h;
    for(int yy=y0;yy<y1;yy++){
        size_t base=(size_t)yy*fb_w;
        for(int xx=x0;xx<x1;xx++){
            size_t i=(base+xx)*4;
            fb_pixels[i+0]=(uint8_t)r;
            fb_pixels[i+1]=(uint8_t)g;
            fb_pixels[i+2]=(uint8_t)b;
            fb_pixels[i+3]=255;
        }
    }
}

void fb_present(){
    if(!g_window||!fb_pixels) return;
    glBindTexture(GL_TEXTURE_2D, fb_tex);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,fb_w,fb_h,GL_RGBA,GL_UNSIGNED_BYTE,fb_pixels);
    glViewport(0,0,fb_w,fb_h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, fb_w, fb_h, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fb_tex);
    glBegin(GL_QUADS);
        glTexCoord2f(0,0); glVertex2f(0,0);
        glTexCoord2f(1,0); glVertex2f(fb_w,0);
        glTexCoord2f(1,1); glVertex2f(fb_w,fb_h);
        glTexCoord2f(0,1); glVertex2f(0,fb_h);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glfwSwapBuffers(g_window);
}

int win_get_mouse_x(){if(!g_window)return 0;double mx,my;glfwGetCursorPos(g_window,&mx,&my);int ww,hh;glfwGetFramebufferSize(g_window,&ww,&hh);return ww?mx*fb_w/ww:(int)mx;}
int win_get_mouse_y(){if(!g_window)return 0;double mx,my;glfwGetCursorPos(g_window,&mx,&my);int ww,hh;glfwGetFramebufferSize(g_window,&ww,&hh);return hh?my*fb_h/hh:(int)my;}
bool win_mouse_down(int btn){return g_window&&glfwGetMouseButton(g_window,btn)==GLFW_PRESS;}

void win_destroy(){
    if(!g_window)return;
    if(fb_pixels){free(fb_pixels);fb_pixels=NULL;}
    if(fb_tex){glDeleteTextures(1,&fb_tex); fb_tex=0;}
    glfwDestroyWindow(g_window);
    g_window=NULL;
    glfwTerminate();
}
