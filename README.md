# C2PY
C2PY 是一个将 C++ 代码转换成 Python 接口的工具，参考自 OpenCV。希望能为数值计算和深度学习领域打造一个好用的，模块化的解决方案。具体用法请看 tests 里面的测试用例，由于目前包含有较多的 OpenCV 源代码，所以该工程仅用于学习和参考。

# for c2py project
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/home/jzw/workspace/c2py_install ..

# for c2py test project
cd tests/single_function
mkdir build && cd build
cmake -DC2py_DIR=/home/jzw/workspace/c2py/build ..
