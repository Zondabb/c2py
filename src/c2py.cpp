#include <math.h>
#include <Python.h>
#include <memory>
#include <string>
#include <vector>

using namespace std;
#define Ptr std::shared_ptr
#define makePtr std::make_shared
#define MODULESTR "c2py"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #include <numpy/ndarrayobject.h>

#  define CV_PYTHON_TYPE_HEAD_INIT() PyVarObject_HEAD_INIT(&PyType_Type, 0)

#include "c2py_generated_include.h"

static PyObject* c2py_error = 0;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

char * timeString() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm * timeinfo = localtime(&ts.tv_sec);
    static char timeStr[40];
    sprintf(timeStr, "%.4d-%.2d-%.2dT%.2d:%.2d:%.2d.%.3ld+08:00",
            timeinfo->tm_year + 1900,
            timeinfo->tm_mon + 1,
            timeinfo->tm_mday,
            timeinfo->tm_hour,
            timeinfo->tm_min,
            timeinfo->tm_sec,
            ts.tv_nsec / 1000000);
    return timeStr;
}

struct ArgInfo
{
    const char * name;
    bool outputarg;
    // more fields may be added if necessary

    ArgInfo(const char * name_, bool outputarg_)
        : name(name_)
        , outputarg(outputarg_) {}

    // to match with older c2py_to function signature
    operator const char *() const { return name; }
};

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (std::exception& e) \
{ \
    PyErr_SetString(c2py_error, e.what()); \
    return 0; \
}

static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

template<typename T> static
bool c2py_to(PyObject* obj, T& p, const char* name = "<unknown>");

template<typename T> static
PyObject* c2py_from(const T& src);

template <typename T>
bool c2py_to(PyObject *o, Ptr<T>& p, const char *name)
{
    if (!o || o == Py_None)
        return true;
    p = makePtr<T>();
    return c2py_to(o, *p, name);
}

template<>
bool c2py_to(PyObject* obj, std::string& value, const char* name)
{
  (void)name;
  if(!obj || obj == Py_None)
    return true;
  const char* str = PyUnicode_AsUTF8(obj);
  if(!str)
    return false;
  value = str;
  return true;
}

template<>
bool c2py_to(PyObject* o, std::vector<size_t>& value, const char* name) {
  if(!o || o == Py_None) {
    return true;
  }

  if(PyTuple_Check(o)) {
    size_t sz = (size_t)PyTuple_Size((PyObject*)o);
    value.resize(sz);
    for (size_t i = 0; i < sz; i++) {
      PyObject* oi = PyTuple_GetItem(o, i);
      if(PyLong_Check(oi)) {
        value[i] = (size_t)PyLong_AsLong(oi);
      } else {
        INFO_LOG("Not a numerical tuple.");
        return false;
      }
    }
    return true;
  }

  return false;
}

template<>
PyObject* c2py_from(const bool& value)
{
    return PyBool_FromLong(value);
}

template<>
bool c2py_to(PyObject* obj, bool& value, const char* name)
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    int _val = PyObject_IsTrue(obj);
    if(_val < 0)
        return false;
    value = _val > 0;
    return true;
}

template<>
PyObject* c2py_from(const size_t& value)
{
    return PyLong_FromSize_t(value);
}

template<>
bool c2py_to(PyObject* obj, size_t& value, const char* name)
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    value = (int)PyLong_AsUnsignedLong(obj);
    return value != (size_t)-1 || !PyErr_Occurred();
}

template<>
PyObject* c2py_from(const int& value)
{
    return PyLong_FromLong(value);
}

template<>
bool c2py_to(PyObject* obj, int& value, const char* name)
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(PyLong_Check(obj))
        value = (int)PyLong_AsLong(obj);
    else
        return false;
    return value != -1 || !PyErr_Occurred();
}

template<>
PyObject* c2py_from(const uint8_t& value)
{
    return PyLong_FromLong(value);
}

template<>
bool c2py_to(PyObject* obj, uint8_t& value, const char* name)
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    int ivalue = (int)PyLong_AsLong(obj);
    value = static_cast<uint8_t>(ivalue);
    return ivalue != -1 || !PyErr_Occurred();
}

template<>
PyObject* c2py_from(const double& value)
{
    return PyFloat_FromDouble(value);
}

template<>
bool c2py_to(PyObject* obj, double& value, const char* name)
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(!!PyLong_CheckExact(obj))
        value = (double)PyLong_AS_LONG(obj);
    else
        value = PyFloat_AsDouble(obj);
    return !PyErr_Occurred();
}

template<>
PyObject* c2py_from(const float& value)
{
    return PyFloat_FromDouble(value);
}

template<>
bool c2py_to(PyObject* obj, float& value, const char* name)
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(!!PyLong_CheckExact(obj))
        value = (float)PyLong_AS_LONG(obj);
    else
        value = (float)PyFloat_AsDouble(obj);
    return !PyErr_Occurred();
}

template<>
PyObject* c2py_from(const int64_t& value)
{
    return PyLong_FromLongLong(value);
}

#define MKTYPE2(NAME) c2py_##NAME##_specials(); if (!to_ok(&c2py_##NAME##_Type)) return NULL;

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#include "c2py_generated_types.h"
#include "c2py_generated_funcs.h"

static PyMethodDef special_methods[] = {
  {NULL, NULL},
};

/* Module init */

struct ConstDef
{
    const char * name;
    long val;
};

static void init_submodule(PyObject * root, const char * name, PyMethodDef * methods, ConstDef * consts)
{
  // traverse and create nested submodules
  std::string s = name;
  size_t i = s.find('.');
  while (i < s.length() && i != std::string::npos)
  {
    size_t j = s.find('.', i);
    if (j == std::string::npos)
        j = s.length();
    std::string short_name = s.substr(i, j-i);
    std::string full_name = s.substr(0, j);
    i = j+1;

    PyObject * d = PyModule_GetDict(root);
    PyObject * submod = PyDict_GetItemString(d, short_name.c_str());
    if (submod == NULL)
    {
        submod = PyImport_AddModule(full_name.c_str());
        PyDict_SetItemString(d, short_name.c_str(), submod);
    }

    if (short_name != "")
        root = submod;
  }

  // populate module's dict
  PyObject * d = PyModule_GetDict(root);
  for (PyMethodDef * m = methods; m->ml_name != NULL; ++m)
  {
    PyObject * method_obj = PyCFunction_NewEx(m, NULL, NULL);
    PyDict_SetItemString(d, m->ml_name, method_obj);
    Py_DECREF(method_obj);
  }
  for (ConstDef * c = consts; c->name != NULL; ++c)
  {
    PyDict_SetItemString(d, c->name, PyLong_FromLong(c->val));
  }

}

#include "c2py_generated_ns_reg.h"

static int to_ok(PyTypeObject *to)
{
  to->tp_alloc = PyType_GenericAlloc;
  to->tp_new = PyType_GenericNew;
  to->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  return (PyType_Ready(to) == 0);
}

static struct PyModuleDef c2py_moduledef = {
    PyModuleDef_HEAD_INIT,
    MODULESTR,
    "Python wrapper for c2py.",
    -1,     /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
    special_methods
};

bool publish_to_module(PyObject *root, const std::string &name, std::string ns, PyObject *type) {
  PyObject *d = PyModule_GetDict(root);
  size_t s = ns.find(".");
  std::string root_name = s == std::string::npos ? ns : ns.substr(0, s);
  PyObject *sub = root_name == "" ? root : PyDict_GetItemString(d, root_name.c_str());

  if (sub == NULL) {
    return false;
  }

  if (s != std::string::npos) {
    return publish_to_module(sub, name, ns.substr(s + 1), type);
  }

  PyModule_AddObject(sub, name.c_str(), type);
  return true;
}

PyMODINIT_FUNC PyInit_c2py() {
//   import_array();

#include "c2py_generated_type_reg.h"

  PyObject* m = PyModule_Create(&c2py_moduledef);
  init_submodules(m); // from "c2py_generated_ns_reg.h"

  PyObject* d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "__version__", PyUnicode_FromString(C2PY_VERSION));

  c2py_error = PyErr_NewException((char*)MODULESTR".error", NULL, NULL);
  PyDict_SetItemString(d, "error", c2py_error);

#define PUBLISH_OBJECT(name, ns, type) Py_INCREF(&type);\
  publish_to_module(m, name, ns, (PyObject *)&type);

#include "c2py_generated_type_publish.h"

#define PUBLISH(I) PyDict_SetItemString(d, #I, PyInt_FromLong(I))
#define PUBLISHU(I) PyDict_SetItemString(d, #I, PyLong_FromUnsignedLong(I))
#define PUBLISH2(I, value) PyDict_SetItemString(d, #I, PyLong_FromLong(value))

  return m;
}
