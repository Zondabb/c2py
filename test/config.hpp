#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include <string>

#define INFO_LOG(format, ...) \
do { \
  fprintf(stderr, "[%s] " format "\n",   \
                  ##__VA_ARGS__);        \
} while(0)
namespace c2py {
char * timeString();
}	// namespace c2py
#endif // __CONFIG_HPP__