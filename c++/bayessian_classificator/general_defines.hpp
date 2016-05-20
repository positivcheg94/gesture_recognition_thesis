#ifndef GENERAL_DEFINES
#define GENERAL_DEFINES

#include <opencv2/opencv.hpp>
#include <librealsense/rs.hpp>

typedef cv::Vec3b vec3b;

namespace gray_pixel{
    const uint8_t black = 0;
    const uint8_t white = 255;
};

namespace rgb_pixel{
    const vec3b black(0,0,0);
    const vec3b white(255,255,255);
};

namespace hsv_pixel{
    const vec3b black(0,0,0);
    const vec3b white(0,0,255);
};

#endif //GENERAL_DEFINES
