#ifndef GENERAL_DEFINES
#define GENERAL_DEFINES

#include <opencv2/opencv.hpp>
#include <librealsense/rs.hpp>

typedef cv::Vec3b vec3b;

template<class UIntegerType>
struct point{
    UIntegerType x,y;
    point() = default;
    point(const point& p) = default;
    point(point&& p) = default;
    point& operator=(const point&) = default;
    point(const UIntegerType& x_, const UIntegerType& y_):x(x_),y(y_) {};
};

typedef point<size_t> spoint;

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
