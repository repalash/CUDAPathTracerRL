//Credits: http://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_02
#include "shader_utils.h"
#include <stdlib.h>
#include <stdio.h>

char* getShaderCode(const char* filename);
void printLog(GLuint object);
GLuint createShader(const char* filename, GLenum type);


GLuint createProgram(const char *vshader_filename, const char* fshader_filename)
{
	//Create shader objects
	GLuint vs, fs;
	if ((vs = createShader(vshader_filename, GL_VERTEX_SHADER))   == 0) return 0;
	if ((fs = createShader(fshader_filename, GL_FRAGMENT_SHADER)) == 0) return 0;

	//Creare program object and link shader objects
	GLuint program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	GLint link_ok;
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
	if (!link_ok) {
		fprintf(stderr, "glLinkProgram error:");
		printLog(program);
		glDeleteShader(vs);
		glDeleteShader(fs);
		glDeleteProgram(program);
		return 0;
	}

	return program;
}

//Read shader source as a string
char* getShaderCode(const char* filename)
{
	FILE* input = fopen(filename, "rb");
	if(input == NULL) return NULL;

	if(fseek(input, 0, SEEK_END) == -1) return NULL;
	long size = ftell(input);
	if(size == -1) return NULL;
	if(fseek(input, 0, SEEK_SET) == -1) return NULL;

	/*if using c-compiler: dont cast malloc's return value*/
	char *content = (char*) malloc( (size_t) size +1  ); 
	if(content == NULL) return NULL;

	fread(content, 1, (size_t)size, input);
	if(ferror(input)) {
		free(content);
		return NULL;
	}

	fclose(input);
	content[size] = '\0';
	return content;
}

//Print error log
void printLog(GLuint object)
{
	GLint log_length = 0;
	if (glIsShader(object))
		glGetShaderiv(object, GL_INFO_LOG_LENGTH, &log_length);
	else if (glIsProgram(object))
		glGetProgramiv(object, GL_INFO_LOG_LENGTH, &log_length);
	else {
		fprintf(stderr, "printlog: Not a shader or a program\n");
		return;
	}

	char* log = (char*)malloc(log_length);

	if (glIsShader(object))
		glGetShaderInfoLog(object, log_length, NULL, log);
	else if (glIsProgram(object))
		glGetProgramInfoLog(object, log_length, NULL, log);

	fprintf(stderr, "%s", log);
	free(log);
}

//Create shader object
GLuint createShader(const char* filename, GLenum type)
{
	const GLchar* source = getShaderCode(filename);
	if (source == NULL) {
		fprintf(stderr, "Error opening %s: ", filename); perror("");
		return 0;
	}
	GLuint res = glCreateShader(type);
	glShaderSource(res, 1, &source, NULL);
	free((void*)source);

	glCompileShader(res);
	GLint compile_ok = GL_FALSE;
	glGetShaderiv(res, GL_COMPILE_STATUS, &compile_ok);
	if (compile_ok == GL_FALSE) {
		fprintf(stderr, "%s:", filename);
		printLog(res);
		glDeleteShader(res);
		return 0;
	}

	return res;
}
