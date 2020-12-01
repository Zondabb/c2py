#ifndef C2PY_HPP
#define C2PY_HPP

#define CV_EXPORTS_W
#define CV_WRAP
#define InputArray
#define OutputArray

#define C2PY_VERSION "0.1.0"

#include <string>

#define INFO_LOG(format, ...) \
do { \
    fprintf(stderr, "[%s] " format "\n",    \
                   timeString(),          \
                   ##__VA_ARGS__);        \
} while(0)

char * timeString();

#endif