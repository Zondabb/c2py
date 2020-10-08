#ifndef C2PY_MODEL_HPP
#define C2PY_MODEL_HPP

#include <vector>
#include <string>
#include "c2py.hpp"

namespace c2py {

// class CV_EXPORTS_W Algorithm {
// };

namespace dnn_inference {

namespace sub {
CV_WRAP int sub_add(int a, int b);

class CV_EXPORTS_W SubModel {
public:
  CV_WRAP SubModel() {}
  virtual ~SubModel() {}
};
}

class CV_EXPORTS_W Model {
public:
    CV_WRAP Model() {}
    virtual ~Model() {}

    CV_WRAP virtual bool open(const std::string& model_file, const std::string& tmp_file);
    // CV_WRAP void face_detect(InputArray src, CV_OUT std::vector<cv::Rect>& faces);
    // CV_WRAP virtual bool landmark(
    //     InputArray src, std::vector<Rect>& objects, OutputArray points);
};

}}

#endif