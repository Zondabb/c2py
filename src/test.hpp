#include "config.hpp"

#define CV_EXPORTS_W
#define CV_WRAP
#define InputArray
#define OutputArray
#define CV_OUT

#define C2PY_VERSION "0.1.0"

#include "mat.h"
#include "model.hpp"

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (std::exception& e) \
{ \
    return 0; \
}

struct ArgInfo
{
  const char * name;
  bool outputarg;
  // more fields may be added if necessary

  ArgInfo(const char * name_, bool outputarg_)
    : name(name_)
    , outputarg(outputarg_) {}

  // to match with older pyopencv_to function signature
  operator const char *() const { return name; }
};

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

struct c2py_Model_t {
  PyObject_HEAD
  std::shared_ptr<c2py::dnn_inference::Model> v;
};

static PyTypeObject c2py_Model_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "c2py.Model",
  sizeof(c2py_Model_t),
};

static void c2py_Model_dealloc(PyObject* self) {
  INFO_LOG("call c2py_Model_dealloc");
  ((c2py_Model_t*)self)->v.reset();
  PyObject_Del(self);
}

static PyObject* c2py_Model_repr(PyObject* self) {
  char str[1000];
  sprintf(str, "<c2py_Model %p>", self);
  return PyUnicode_FromString(str);
}

static PyGetSetDef c2py_Model_getseters[] = {
  {NULL}  /* Sentinel */
};

static int c2py_Model_Model(c2py_Model_t* self, PyObject* args, PyObject* kw)
{
  INFO_LOG("123 my c2py_Model_Model...");
  // return 0;
  if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
  {
    new (&(self->v)) std::shared_ptr<c2py::dnn_inference::Model>(); // init Ptr with placement new
    if(self) {
      INFO_LOG("model construct to self...");
      ERRWRAP2(self->v.reset(new c2py::dnn_inference::Model()));
      if (self->v.get()) {
        INFO_LOG("run 1");
      } else {
        INFO_LOG("run 2");
      }
    }
    INFO_LOG("model construct success");
    return 0;
  }
  INFO_LOG("model construct failed");

  return -1;
}

bool pyopencv_to(PyObject* obj, std::string& value, const char* name)
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

bool pyopencv_to(PyObject* o, Mat& m) {
  if(!o || o == Py_None) {
    return true;
  }

  if(PyTuple_Check(o)) {
    failmsgp("Not a numpy array, neither a scalar");
    return false;
  }

  PyArrayObject* oarr = (PyArrayObject*) o;
}

PyObject* pyopencv_from(const bool& value)
{
    return PyBool_FromLong(value);
}

static PyObject* c2py_Model_open(PyObject* self, PyObject* args, PyObject* kw)
{
  INFO_LOG("run c2py model open");
  // using namespace c2py::dnn_inference;

  c2py::dnn_inference::Model* _self_ = NULL;
  if(PyObject_TypeCheck(self, &c2py_Model_Type)) {
    _self_ = ((c2py_Model_t*)self)->v.get();
  }
  
  if (_self_ == NULL)
    return failmsgp("Incorrect type of self (must be 'c2py_dnn_inference_Model' or its derivative)");
  PyObject* pyobj_model_file = NULL;
  std::string model_file;
  PyObject* pyobj_tmp_file = NULL;
  std::string tmp_file;
  bool retval;

  const char* keywords[] = { "model_file", "tmp_file", NULL };
  if( PyArg_ParseTupleAndKeywords(args, kw, "OO:c2py_dnn_inference_Model.open", (char**)keywords, &pyobj_model_file, &pyobj_tmp_file) &&
      pyopencv_to(pyobj_model_file, model_file, ArgInfo("model_file", 0)) &&
      pyopencv_to(pyobj_tmp_file, tmp_file, ArgInfo("tmp_file", 0)) )
  {
      ERRWRAP2(retval = _self_->open(model_file, tmp_file));
      return pyopencv_from(retval);
  }

  Py_RETURN_NONE;
}

static PyMethodDef c2py_Model_methods[] =
{
    {"open", (PyCFunction)c2py_Model_open, METH_VARARGS | METH_KEYWORDS, "open(model_file) -> retval\n."},

    {NULL,          NULL}
};

static void c2py_Model_specials(void)
{
    c2py_Model_Type.tp_base = NULL;
    c2py_Model_Type.tp_dealloc = c2py_Model_dealloc;
    c2py_Model_Type.tp_repr = c2py_Model_repr;
    c2py_Model_Type.tp_getset = c2py_Model_getseters;
    c2py_Model_Type.tp_init = (initproc)c2py_Model_Model;
    c2py_Model_Type.tp_methods = c2py_Model_methods;
}

static int to_ok(PyTypeObject *to)
{
  to->tp_alloc = PyType_GenericAlloc;
  to->tp_new = PyType_GenericNew;
  to->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  return (PyType_Ready(to) == 0);
}
