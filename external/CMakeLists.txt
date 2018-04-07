#if(MSVC AND NOT "${MSVC_VERSION}" LESS 1400)
#	add_definitions( "/MP" )
#endif()
#
#

# create a C static library
IF(NOT CMAKE_C_CREATE_STATIC_LIBRARY)
	SET(CMAKE_C_CREATE_STATIC_LIBRARY
			"<CMAKE_AR> cr <TARGET> <LINK_FLAGS> <OBJECTS> "
			"<CMAKE_RANLIB> <TARGET> ")
ENDIF(NOT CMAKE_C_CREATE_STATIC_LIBRARY)

# compile a C file into an object file
IF(NOT CMAKE_C_COMPILE_OBJECT)
	SET(CMAKE_C_COMPILE_OBJECT
			"<CMAKE_C_COMPILER> -o <OBJECT>  <FLAGS> -c <SOURCE>")
ENDIF(NOT CMAKE_C_COMPILE_OBJECT)


add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)

### GLFW ###

add_subdirectory (glfw-3.1.2)

include_directories(
	glfw-3.1.2/include/GLFW/
	glew-1.13.0/include/
)

### GLEW ###

#set(GLEW_SOURCE
#	glew-1.13.0/src/glew.c
#)
#
#set(GLEW_INCLUDE glew-1.13.0/include/)
#
#
#add_library( GLEW_1130 STATIC
#	${GLEW_SOURCE}
#	${GLEW_INCLUDE}
#)
#
#target_include_directories(GLEW_1130 PUBLIC
#		${GLEW_INCLUDE})
#
#target_link_libraries(GLEW_1130
#	${OPENGL_LIBRARY}
#	${EXTRA_LIBS}
#)

#### ANTTWEAKBAR ###
#
#set(ANTTWEAKBAR_SOURCE
#	AntTweakBar-1.16/src/LoadOGL.cpp
#	AntTweakBar-1.16/src/LoadOGLCore.cpp
#	AntTweakBar-1.16/src/TwColors.cpp
#	AntTweakBar-1.16/src/TwBar.cpp
#	AntTweakBar-1.16/src/TwEventGLFW.c
#	AntTweakBar-1.16/src/TwFonts.cpp
#	AntTweakBar-1.16/src/TwMgr.cpp
#	AntTweakBar-1.16/src/TwOpenGL.cpp
#	AntTweakBar-1.16/src/TwOpenGLCore.cpp
#	AntTweakBar-1.16/src/TwPrecomp.cpp
#)
#
#add_definitions(
#	-D_WINDOWS
#)
#
#set(ANTTWEAKBAR_HEADERS
#	AntTweakBar-1.16/src/AntPerfTimer.h
#	AntTweakBar-1.16/src/LoadOGL.h
#	AntTweakBar-1.16/src/LoadOGLCore.h
#	AntTweakBar-1.16/src/MiniGLFW.h
#	AntTweakBar-1.16/src/resource.h
#	AntTweakBar-1.16/src/TwBar.h
#	AntTweakBar-1.16/src/TwColors.h
#	AntTweakBar-1.16/src/TwFonts.h
#	AntTweakBar-1.16/src/TwGraph.h
#	AntTweakBar-1.16/src/TwMgr.h
#	AntTweakBar-1.16/src/TwOpenGL.h
#	AntTweakBar-1.16/src/TwOpenGLCore.h
#	AntTweakBar-1.16/src/TwPrecomp.h
#)
#
#include_directories(
#	AntTweakBar-1.16/include/
#)
#
#add_library( ANTTWEAKBAR_116_OGLCORE_GLFW STATIC
#	${ANTTWEAKBAR_SOURCE}
#	${ANTTWEAKBAR_HEADERS}
#)
#
#target_link_libraries(ANTTWEAKBAR_116_OGLCORE_GLFW
#	${OPENGL_LIBRARY}
#	${EXTRA_LIBS}
#)
#
#### ASSIMP ###
## AssImp already has a CMakeLists.txt so let's use these
#
## Compile built-in, modified version of Zlib
#include(CheckIncludeFile)
#include(CheckTypeSize)
#include(CheckFunctionExists)
#add_subdirectory( assimp-3.0.1270/contrib/zlib )
#

## Compile without Boost
#include_directories( assimp-3.0.1270/code/BoostWorkaround )
#add_definitions( -DASSIMP_BUILD_BOOST_WORKAROUND )

#
## Compile AssImp
#set( LIB_INSTALL_DIR "lib")
#set(LIBASSIMP_COMPONENT libassimp3.0-r1270-OGLtuts)
#set(ZLIB_LIBRARIES zlib)
#set(BUILD_STATIC_LIB ON)
##set(ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/zlib)
#add_subdirectory( assimp-3.0.1270/code )
#
#
#### BULLET ###
## Bullet already has a CMakeLists.txt so let's use these
#
#set(BULLET_VERSION 2.81)
#include_directories(
#	bullet-2.81-rev2613/src
#)
#add_subdirectory( bullet-2.81-rev2613/src/BulletSoftBody )
#add_subdirectory( bullet-2.81-rev2613/src/BulletCollision )
#add_subdirectory( bullet-2.81-rev2613/src/BulletDynamics )
#add_subdirectory( bullet-2.81-rev2613/src/LinearMath )
#
