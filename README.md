# for c2py project
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/home/jzw/workspace/c2py_install ..

# for c2py test project
cd tests/single_function
mkdir build && cd build
cmake -DC2py_DIR=/home/jzw/workspace/c2py/build ..
