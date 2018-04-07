//structs.h
#ifndef _STRUCTS_H_
#define _STRUCTS_H_

#include "vector3D.h"
#include "color.h"
#include "material.h"

#define MAX_LIGHTS 10
struct renderViewInfo
{
    Vector3D eye;
    double screenWidth;
    double screenHeight;
    double screenZ;
    int windowWidth;
    int windowHeight;
 
    Vector3D light[MAX_LIGHTS];
    Color lightColor[MAX_LIGHTS];
    Color lightAmbient;
    int levelOfRecursion;
    bool doJitter;
    bool disableSpecular;    
    int numLights;
};

#endif
