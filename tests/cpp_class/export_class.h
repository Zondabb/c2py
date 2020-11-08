#ifndef __EXPORT_CLASS_H__
#define __EXPORT_CLASS_H__

#include <string>
#include <vector>

#include "c2py.hpp"

namespace c2py {

class CV_EXPORTS_W Mod {
public:
  CV_WRAP Mod() {}
  virtual ~Mod() {}

  CV_WRAP virtual void Print(const std::string &str);
  CV_WRAP virtual int Sum(const std::vector<size_t> &num_list);
  // CV_WRAP void face_detect(InputArray src, CV_OUT std::vector<cv::Rect>& faces);
  // CV_WRAP virtual bool landmark(
  //     InputArray src, std::vector<Rect>& objects, OutputArray points);
};

} // namespace c2py

#endif