#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string>
#include <vector>
#include <memory>
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#define M 10000
#define K 10000
#define N 10000
using namespace std;

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

#define PUBLISH_OBJECT(name, type) Py_INCREF(&type);\
  PyModule_AddObject(m, name, (PyObject *)&type);

#include "test.hpp"

int fac(int n){
  if (n<2)
    return 1;
  else
    return (n)*fac(n-1);
}

static PyObject * Extest_fac(PyObject *self,PyObject *args){
  int num;
  if(!PyArg_ParseTuple(args,"i",&num)) return NULL;
	return (PyObject*)Py_BuildValue("i",fac(num));
}

static PyObject * Extest_keyword(PyObject *self, PyObject *args, PyObject *keywds) {
  int voltage;
  char *state = "a stiff";
  char *action = "voom";
  char *type = "Norwegian Blue";

  static char *kwlist[] = {"voltage", "state", "action", "type", NULL};
  if (!PyArg_ParseTupleAndKeywords(
      args, keywds, "i|sss:keyword", kwlist,
      &voltage, &state, &action, &type)) {
      return NULL;
  }
  printf("-- This parrot wouldn't %s if you put %i Volts through it.\n", 
      action, voltage);
  printf("-- Lovely plumage, the %s -- It's %s!\n", type, state);

  Py_INCREF(Py_None);
  return Py_None;
}

vector< vector<int> > matrix_multiply(vector< vector<int> > arrA, vector< vector<int> > arrB) {
	int rowA = arrA.size();
    int colA = arrA[0].size();
	int rowB = arrB.size();
	int colB = arrB[0].size();
	vector< vector<int> > res;
	if (colA != rowB) {
		return res;
	}

	res.resize(rowA);
	for (int i = 0; i < rowA; i++) {
		res[i].resize(colB);
	}

	for (int i = 0; i < rowA; i++) {
		for (int j = 0; j < colB; j++) {
			for (int k = 0; k < colA; k++) {
				res[i][j] += arrA[i][k] * arrB[k][j];
			}
		}
	}
	return res;
}

static PyObject * long_running_test(PyObject *self, PyObject *args, PyObject *kw) {
	PyObject *pyobj_model_file = NULL;
    std::string model_file;
	const char* keywords[] = { "model_file", NULL };

	if( PyArg_ParseTupleAndKeywords(
		args, kw, "O:Extest.long_running_test", 
		(char**)keywords, &pyobj_model_file)) {
		
		PyAllowThreads allowThreads;
		{
			PyEnsureGIL gil;
		}
		// PyEnsureGIL gil;
		vector< vector<int> > A;
		vector< vector<int> > B;
		A.resize(M);
		for (int i = 0; i < M; i++) {
			A[i].resize(K);
			for (int j = 0; j < K; j++) {
				A[i][j] = 1;
			}
		}
		B.resize(K);
		for (int i = 0; i < K; i++) {
			B[i].resize(N);
			for (int j = 0; j < N; j++) {
				B[i][j] = 1;
			}
		}
		printf("start computing...\n");
		vector< vector<int> > C = matrix_multiply(A, B);
		for (int i = 0; i < 10; i++) {
			printf("%i, ", C[0][i]);
		}
		printf("\n");
		const char* str = PyUnicode_AsUTF8(pyobj_model_file);
		printf("%s.\n", str);
	}
	Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ExtestMethods[] = {
	{"fac", Extest_fac, METH_VARARGS},
    {"keyword", (PyCFunction)Extest_keyword, METH_VARARGS | METH_KEYWORDS},
	{"long_running_test", (PyCFunction)long_running_test, METH_VARARGS | METH_KEYWORDS},
	{NULL, NULL}
};

static struct PyModuleDef extestmodule = {
	PyModuleDef_HEAD_INIT,
	"Extest",
	NULL,
	-1,
	ExtestMethods
};

PyMODINIT_FUNC
PyInit_c2py(void){
	import_array();
	c2py_Model_specials(); 
	if (!to_ok(&c2py_Model_Type)) {
    return NULL;
	}

	PyObject *m;
	m = PyModule_Create(&extestmodule);
	if(m == NULL) {
    return NULL;
  }

	PyObject* d = PyModule_GetDict(m);
	PyDict_SetItemString(d, "__version__", PyUnicode_FromString("0.1.0"));

	static PyObject* c2py_error = 
		PyErr_NewException((char*)"Extest.error", NULL, NULL);
	PyDict_SetItemString(d, "error", c2py_error);

  PUBLISH_OBJECT("c2py_Model", c2py_Model_Type)
	// Tensor _mat;
	// INFO_LOG("mat rows: %d, cols: %d", _mat.size[0], _mat.size[1]);
	return m;
}