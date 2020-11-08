#include "export_class.h"

namespace c2py {

void Mod::Print(const std::string &str) {
  INFO_LOG("Print from Mod: %s", str.c_str());
}

int Mod::Sum(const std::vector<size_t> &num_list) {
  int sum = 0;
  for (int i = 0; i < num_list.size(); i++) {
    sum += num_list[i];
  }
  return sum;
}

} // cv