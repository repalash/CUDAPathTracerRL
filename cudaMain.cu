//
// Created by Palash on 19-02-2018.
//

#include "cudaMain.h"

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
#include <time.h>

using namespace glm;

#include "shader_utils.h"
#include "gl_utils.h"
#include "camera.h"
#include "renderengine.h"
#include "world.h"
#include "material.h"
#include "object.h"
#include "sphere.h"
#include "lightsource.h"
#include "pointlightsource.h"
#include "triangle.h"
#include "quadric.h"
#include "renderengine_gpu.h"

//Globals
GLuint program;
GLint attribute_coord2d;
int screen_width = 360, screen_height = 240; //Both even numbers
float quadVertices[] = {-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1}; //2D screen space coordinates
GLuint texImage;
GLint uniform_texImage;
GLuint vao, vbo;

Camera *camera;
RenderEngine *engine;

int init_resources(void)
{
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    //Create program
    program = createProgram("vshader.vs", "fshader.fs");
    attribute_coord2d = glGetAttribLocation(program, "coord2d");
    if(attribute_coord2d == -1)
    {
        fprintf(stderr, "Could not bind location: coord2d\n");
        return 0;
    }

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    //Scene Setup...

    //Initialize raytracer objects
    Vector3D camera_position(0, 3, 16);
    Vector3D camera_target(0,0,0); //Looking down -Z axis
    Vector3D camera_up(0, 1, 0);
    float camera_fovy =  45;
    camera = new Camera(camera_position, camera_target, camera_up, camera_fovy, screen_width, screen_height);
    //Create a world
    World *world = new World;
    world->setAmbient(Color(1));
    world->setBackground(Color(0, 0, 0));

    Material *m = new Material(world); //glossy
    m->color = Color(1, 1, 0.23);  m->n = 20;

    Material *mPlane = new Material(world);
    mPlane->color = Color(0.9,0.9,0.9); mPlane->kr = 0;
    Material *mPlane2 = new Material(world);
    mPlane2->color = Color(0.9,0.2,0.3); mPlane2->kr = 0;
    Material *mPlane3 = new Material(world);
    mPlane3->color = Color(0.1,0.3,0.9); mPlane3->kr = 0;
    Material *mPlane4 = new Material(world);
    mPlane4->color = Color(0.1,0.9,0.2); mPlane4->kr = 0;

    Object *sphere5 = new Sphere(Vector3D(0, -4, 0), 1.5, m);
    world->addObject(sphere5);

    Object *sphere6 = new Sphere(Vector3D(0, -2006, 0), 2000, mPlane);
    world->addObject(sphere6);

    world->addObject(new Sphere(Vector3D(2009, 0, 0), 2000, mPlane3));
    world->addObject(new Sphere(Vector3D(-2009, 0, 0), 2000, mPlane2));
    world->addObject(new Sphere(Vector3D(0, 0, -2012), 2000, mPlane4));
    world->addObject(new Sphere(Vector3D(0, 0, 2016), 2000, mPlane2));
//    world->addObject(new Sphere(Vector3D(0, 2016, 0), 2000, mPlane));

    LightSource *light = new PointLightSource(world, Vector3D(0, 16, 0), Color(3,3,3));
    world->addLight(light);

    engine = new RenderEngine_GPU(world, camera);

    //Initialise texture
    glGenTextures(1, &texImage);
    glBindTexture(GL_TEXTURE_2D, texImage);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screen_width, screen_height, 0, GL_RGB, GL_UNSIGNED_BYTE, camera->getBitmap());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //Show pixels when zoomed
    uniform_texImage = glGetUniformLocation(program, "texImage");
    if(uniform_texImage == -1)
    {
        fprintf(stderr, "Could not bind uniform: texImage\n");
        return 0;
    }
    return 1;
}

void onDisplay()
{
    /* Clear the background as white */
    glClearColor(1.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    printOpenGLError();

    glUseProgram(program);
    printOpenGLError();

    glEnableVertexAttribArray(attribute_coord2d);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    printOpenGLError();
    glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    printOpenGLError();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texImage);
    glUniform1i(uniform_texImage, 0);

    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDisableVertexAttribArray(attribute_coord2d);
    printOpenGLError();

    /* Display the result */
    glfwSwapBuffers(window);
    glfwPollEvents();
    printOpenGLError();
}

void free_resources()
{
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glDeleteTextures(1, &texImage);
}

void onReshape(int width, int height) { //TODO: port for GLFW
    screen_width = width;
    screen_height = height;
    glViewport(0, 0, screen_width, screen_height);
}

void onIdle(void)
{
    static bool done = false;
    //Generate a pretty picture
    if(!done)
    {
//        for(int i=0; i < screen_width/10; i++)
            if(engine->renderLoop())
            {
                done = true;
                fprintf(stderr, "Rendering complete.\n");
            }

        //Update texture on GPU
        glBindTexture(GL_TEXTURE_2D, texImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screen_width, screen_height, GL_RGB, GL_UNSIGNED_BYTE, camera->getBitmap());
    }
}

int cudaMain(int argc, char **argv) {

    srand((unsigned int) time(0));
    if(argc > 1)
    {
        screen_width = atoi(argv[1]);
        screen_height = atoi(argv[2]);
        screen_width -= (screen_width % 2); //Make it even
        screen_height -= (screen_height % 2); //Make it even
    }
    fprintf(stderr, "Welcome to Blend raytracer and editor.\nFull command: %s [width] [height]\nPress 's' to save framebufer to disk.\n", argv[0]);


    // Initialise GLFW
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow( screen_width, screen_height, "Path Tracer", NULL, NULL);
    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    if (1 == init_resources()) {
        do {
            onIdle();
            onDisplay();
        } // Check if the ESC key was pressed or the window was closed
        while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
               glfwWindowShouldClose(window) == 0);
    }
    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    free_resources();
    return EXIT_SUCCESS;
}
