#ifndef C2PY_MODEL_HPP
#define C2PY_MODEL_HPP

#include <vector>
#include "c2py.hpp"

namespace c2py { namespace dnn_inference {

class CV_EXPORTS_W Model {
public:
    CV_WRAP Model() {}
    virtual ~Model() {}

    CV_WRAP virtual bool open(const std::string& model_file);
    // CV_WRAP void face_detect(InputArray src, CV_OUT std::vector<cv::Rect>& faces);
    // CV_WRAP virtual bool landmark(
    //     InputArray src, std::vector<Rect>& objects, OutputArray points);
};

}}

#endif