Tensor::Tensor() : _data(nullptr) {}

Tensor::Tensor(std::vector<size_t> shape, TensorType type) : _data(nullptr), _shape(shape), _step(shape), _type(type) {
  int _dim = shape.size();
  size_t size = 1;
  for (int i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  _data = std::make_shared<uint8_t>(size * type.SizeInBytes());
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<size_t> step, TensorType type) :
  _data(nullptr), _shape(shape), _step(step), _type(type) {
    int _dim = shape.size();
    size_t size = 1;
    for (int i = 0; i < step.size(); i++) {
      size *= step[i];
    }
    _data = std::make_shared<uint8_t>(size * type.SizeInBytes());
}

// template<typename T> Tensor::Tensor(const std::vector<T>& vec) : dims(2), data(nullptr) {
//   size = std::vector<size_t>({vec.size(), 1});
//   step = std::vector<size_t>({vec.size(), 1});
//   data.reset(vec.data());
// }