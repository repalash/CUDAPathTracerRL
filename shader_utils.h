#ifndef _SHADER_UTILS_H_
#define _SHADER_UTILS_H_

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GL/glew.h>
#endif
GLuint createProgram(const char *vshader_filename, const char* fshader_filename);
#endif
