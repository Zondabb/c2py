#include "config.hpp"

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
  // Ptr<c2py::dnn_inference::Model> v;
};

static PyTypeObject c2py_Model_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "c2py.Model",
  sizeof(c2py_Model_t),
};

static void c2py_Model_dealloc(PyObject* self) {
  // ((pyopencv_c2py_dnn_inference_Model_t*)self)->v.release();
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
  INFO_LOG("my c2py_Model_Model...");
  return 0;
}

static PyObject* c2py_Model_open(PyObject* self, PyObject* args, PyObject* kw)
{
  INFO_LOG("my open...");
  // using namespace c2py::dnn_inference;

  // c2py::dnn_inference::Model* _self_ = NULL;
  if(!PyObject_TypeCheck(self, &c2py_Model_Type))
    Py_RETURN_NONE;
  
  return failmsgp("Incorrect type of self (must be 'c2py_dnn_inference_Model' or its derivative)");
  // PyObject* pyobj_model_file = NULL;
  // string model_file;
  // bool retval;

  // const char* keywords[] = { "model_file", NULL };
  // if( PyArg_ParseTupleAndKeywords(args, kw, "O:c2py_dnn_inference_Model.open", (char**)keywords, &pyobj_model_file) &&
  //     pyopencv_to(pyobj_model_file, model_file, ArgInfo("model_file", 0)) )
  // {
  //     ERRWRAP2(retval = _self_->open(model_file));
  //     return pyopencv_from(retval);
  // }
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
