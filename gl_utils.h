#ifndef _GL_UTILS_H_
#define _GL_UTILS_H_

#include <GL/glew.h>
#include <cstdio>

#define printOpenGLError() printOglError(__FILE__, __LINE__)
#define degreeToRadians(X) ((X)*M_PI/180.0f)

int printOglError(char *file, int line) {
    //
    // Returns 1 if an OpenGL error occurred, 0 otherwise.
    //
    GLenum glErr;
    int    retCode = 0;

    glErr = glGetError();
    while (glErr != GL_NO_ERROR)
    {
        printf("glError in file %s @ line %d: %d, %s\n", file, line, glErr, gluErrorString(glErr));
        retCode = 1;
        glErr = glGetError();
    }
    return retCode;
}

#endif
