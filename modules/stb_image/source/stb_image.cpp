#define STB_IMAGE_IMPLEMENTATION

#ifdef WIN32
#define __EXPORT __declspec(dllexport)
#else
#define __EXPORT
#endif

#define STBIDEF extern __EXPORT

#include "../include/stb_image.h"