#include "config.hpp"

#define CV_EXPORTS_W
#define CV_WRAP
#define InputArray
#define OutputArray
#define CV_OUT

#define C2PY_VERSION "0.1.0"

#include "tensor.h"
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
  INFO_LOG("model dealloc...");
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
  if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
  {
    new (&(self->v)) std::shared_ptr<c2py::dnn_inference::Model>(); // init Ptr with placement new
    if(self) {
      ERRWRAP2(self->v.reset(new c2py::dnn_inference::Model()));
      // if (self->v.get()) {
      //   TODO
      // } else {
      //   TODO
      // }
    }
    INFO_LOG("model construct success...");
    return 0;
  }
  INFO_LOG("model construct failed...");

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

bool pyopencv_to(PyObject* o, Tensor& t) {
  if(!o || o == Py_None) {
    return true;
  }

  if(PyTuple_Check(o)) {
    size_t i, sz = (size_t)PyTuple_Size((PyObject*)o);
    // t = Mat(sz, 1, CV_64F);
    t = Tensor({sz, 1}, TensorType::FLOAT32);
    for( i = 0; i < sz; i++ ) {
      PyObject* oi = PyTuple_GetItem(o, i);
      if(PyLong_Check(oi)) {
        t.at<float>(i) = (float)PyLong_AsLong(oi);
      } else if (PyFloat_Check(oi)) {
        t.at<float>(i) = (float)PyFloat_AsDouble(oi);
      } else {
        INFO_LOG("Not a numerical tuple.");
        return false;
      }
    }
    return true;
  }

  if(!PyArray_Check(o)) {
    INFO_LOG("Not a numpy array, neither a scalar");
    return false;
  }

  // PyArrayObject* oarr = (PyArrayObject*) o;
  // bool needcopy = false, needcast = false;
  // int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
  // int type = typenum == NPY_UBYTE ? CV_8U :
  //             typenum == NPY_BYTE ? CV_8S :
  //             typenum == NPY_USHORT ? CV_16U :
  //             typenum == NPY_SHORT ? CV_16S :
  //             typenum == NPY_INT ? CV_32S :
  //             typenum == NPY_INT32 ? CV_32S :
  //             typenum == NPY_FLOAT ? CV_32F :
  //             typenum == NPY_DOUBLE ? CV_64F : -1;
}

PyObject* pyopencv_from(const bool& value) {
  return PyBool_FromLong(value);
}

PyObject* pyopencv_from(const Tensor& t) {

}

static PyObject* c2py_Model_open(PyObject* self, PyObject* args, PyObject* kw)
{
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

static PyObject* c2py_Model_compute(PyObject* self, PyObject* args, PyObject* kw) {
  c2py::dnn_inference::Model* _self_ = NULL;
  if(PyObject_TypeCheck(self, &c2py_Model_Type)) {
    _self_ = ((c2py_Model_t*)self)->v.get();
  }
  if (_self_ == NULL)
    return failmsgp("Incorrect type of self (must be 'c2py_dnn_inference_Model' or its derivative)");
  PyObject* pyobj_mat_a = NULL;
  Tensor ta;
  PyObject* pyobj_mat_b = NULL;
  Tensor tb;
  bool retval;

  const char* keywords[] = { "mat_a", "mat_b", NULL };
  if( PyArg_ParseTupleAndKeywords(args, kw, "OO:c2py_dnn_inference_Model.compute", (char**)keywords, &pyobj_mat_a, &pyobj_mat_b) &&
      pyopencv_to(pyobj_mat_a, ta) &&
      pyopencv_to(pyobj_mat_b, tb) )
  {
      // ERRWRAP2(retval = _self_->open(model_file, tmp_file));
      return pyopencv_from(retval);
  }

  Py_RETURN_NONE;
}

static PyMethodDef c2py_Model_methods[] =
{
  {"open", (PyCFunction)c2py_Model_open, METH_VARARGS | METH_KEYWORDS, "open(model_file) -> retval\n."},
  {"compute", (PyCFunction)c2py_Model_compute, METH_VARARGS | METH_KEYWORDS, "compute(mat_a, mat_b -> retaval)\n."},
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
