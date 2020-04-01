#ifndef C2PY_TEST_HPP
#define C2PY_TEST_HPP

#define CV_EXPORTS

namespace c2py { namespace test {

class CV_EXPORTS Test
{
public:
    Test();
    ~Test();

    // enum
    // {
    //     TYPE_DEFAULT     = (1 << 0),
    //     TYPE_CPU         = (1 << 1),
    //     TYPE_GPU         = (1 << 2),
    //     TYPE_ACCELERATOR = (1 << 3),
    //     TYPE_DGPU        = TYPE_GPU + (1 << 16),
    //     TYPE_IGPU        = TYPE_GPU + (1 << 17),
    //     TYPE_ALL         = 0xFFFFFFFF
    // };

    enum
    {
        UNKNOWN_VENDOR=0,
        VENDOR_AMD=1,
        VENDOR_INTEL=2,
        VENDOR_NVIDIA=3
    };

    
};

}}
#endif