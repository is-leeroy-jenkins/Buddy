#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once
#include <ATen/Dimname.h>
#include <torch/csrc/python_headers.h>

at::Dimname THPDimname_parse(PyObject* obj);
bool THPUtils_checkDimname(PyObject* obj);
bool THPUtils_checkDimnameList(PyObject* obj);

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
