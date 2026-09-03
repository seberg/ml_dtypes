/* Copyright 2024 The ml_dtypes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ML_DTYPES_FLOATS_H_
#define ML_DTYPES_FLOATS_H_

#include <cstring>
#include <limits>
#include <type_traits>

#include "Eigen/Core"
#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/include/float8.h"
#include "ml_dtypes/include/mxfloat.h"

namespace ml_dtypes {

template <typename T>
struct CustomFloatTraits {};

template <>
struct CustomFloatTraits<bfloat16> {
  static constexpr const char* kTypeName = "bfloat16";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.bfloat16";
  static constexpr const char* kTpDoc = "bfloat16 floating-point values";
  static constexpr char kNumPy1DescrType = 'E';
};

template <>
struct CustomFloatTraits<float8_e3m4> {
  static constexpr const char* kTypeName = "float8_e3m4";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e3m4";
  static constexpr const char* kTpDoc = "float8_e3m4 floating-point values";
  static constexpr char kNumPy1DescrType = '3';
};

template <>
struct CustomFloatTraits<float8_e4m3> {
  static constexpr const char* kTypeName = "float8_e4m3";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3";
  static constexpr const char* kTpDoc = "float8_e4m3 floating-point values";
  static constexpr char kNumPy1DescrType = '7';
};

template <>
struct CustomFloatTraits<float8_e4m3b11fnuz> {
  static constexpr const char* kTypeName = "float8_e4m3b11fnuz";
  static constexpr const char* kQualifiedTypeName =
      "ml_dtypes.float8_e4m3b11fnuz";
  static constexpr const char* kTpDoc =
      "float8_e4m3b11fnuz floating-point values";
  static constexpr char kNumPy1DescrType = 'L';
};

template <>
struct CustomFloatTraits<float8_e4m3fn> {
  static constexpr const char* kTypeName = "float8_e4m3fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3fn";
  static constexpr const char* kTpDoc = "float8_e4m3fn floating-point values";
  static constexpr char kNumPy1DescrType = '4';
};

template <>
struct CustomFloatTraits<float8_e4m3fnuz> {
  static constexpr const char* kTypeName = "float8_e4m3fnuz";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3fnuz";
  static constexpr const char* kTpDoc = "float8_e4m3fnuz floating-point values";
  static constexpr char kNumPy1DescrType = 'G';
};

template <>
struct CustomFloatTraits<float8_e5m2> {
  static constexpr const char* kTypeName = "float8_e5m2";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e5m2";
  static constexpr const char* kTpDoc = "float8_e5m2 floating-point values";
  static constexpr char kNumPy1DescrType = '5';
};

template <>
struct CustomFloatTraits<float8_e5m2fnuz> {
  static constexpr const char* kTypeName = "float8_e5m2fnuz";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e5m2fnuz";
  static constexpr const char* kTpDoc = "float8_e5m2fnuz floating-point values";
  static constexpr char kNumPy1DescrType = 'C';
};

template <>
struct CustomFloatTraits<float6_e2m3fn> {
  static constexpr const char* kTypeName = "float6_e2m3fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float6_e2m3fn";
  static constexpr const char* kTpDoc = "float6_e2m3fn floating-point values";
  static constexpr char kNumPy1DescrType = '8';
};

template <>
struct CustomFloatTraits<float6_e3m2fn> {
  static constexpr const char* kTypeName = "float6_e3m2fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float6_e3m2fn";
  static constexpr const char* kTpDoc = "float6_e3m2fn floating-point values";
  static constexpr char kNumPy1DescrType = '9';
};

template <>
struct CustomFloatTraits<float4_e2m1fn> {
  static constexpr const char* kTypeName = "float4_e2m1fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float4_e2m1fn";
  static constexpr const char* kTpDoc = "float4_e2m1fn floating-point values";
  static constexpr char kNumPy1DescrType = '0';
};

template <>
struct CustomFloatTraits<float8_e8m0fnu> {
  static constexpr const char* kTypeName = "float8_e8m0fnu";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e8m0fnu";
  static constexpr const char* kTpDoc = "float8_e8m0fnu floating-point values";
  static constexpr char kNumPy1DescrType = 'W';
};

template <typename T, typename = void>
struct is_custom_float : std::false_type {};

template <typename T>
struct is_custom_float<T,
                       std::void_t<decltype(CustomFloatTraits<T>::kTypeName)>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_custom_float_v = is_custom_float<T>::value;

// True if every value of Src is exactly representable in Dst.
// Ignore has_signaling_NaN since NumPy doesn't use it.
template <typename Src, typename Dst>
inline constexpr bool CustomFloatSafeTo() {
  return (!std::numeric_limits<Src>::is_signed ||
          std::numeric_limits<Dst>::is_signed) &&
         std::numeric_limits<Dst>::digits >= std::numeric_limits<Src>::digits &&
         std::numeric_limits<Dst>::min_exponent -
                 std::numeric_limits<Dst>::digits <=
             std::numeric_limits<Src>::min_exponent -
                 std::numeric_limits<Src>::digits &&
         std::numeric_limits<Dst>::max_exponent >=
             std::numeric_limits<Src>::max_exponent &&
         (!std::numeric_limits<Src>::has_infinity ||
          std::numeric_limits<Dst>::has_infinity) &&
         (!std::numeric_limits<Src>::has_quiet_NaN ||
          std::numeric_limits<Dst>::has_quiet_NaN);
}

template <typename T>
struct CustomFloatType {
  static int Dtype() { return npy_type; }

  // Registered numpy type ID. Global variable populated by the registration
  // code. Protected by the GIL.
  static int npy_type;

  // Pointer to the python type object we are using. This is either a pointer
  // to type, if we choose to register it, or to the python type
  // registered by another system into NumPy.
  static PyObject* type_ptr;

  static PyMethodDef methods[];
  static PyType_Spec type_spec;
  static PyType_Slot type_slots[];
  static PyArray_ArrFuncs arr_funcs;
  static PyArray_DescrProto npy_descr_proto;
  static PyArray_Descr* npy_descr;

  // New-style DType metaclass object.  Zero-initialized; fields are filled in
  // at registration time before PyType_Ready is called.
  static PyArray_DTypeMeta dtype_meta;
};

// True if `meta` is one of our custom floating-point DTypes.
inline bool IsCustomFloatDType(const PyArray_DTypeMeta* meta) {
  return meta == &CustomFloatType<bfloat16>::dtype_meta ||
         meta == &CustomFloatType<float8_e3m4>::dtype_meta ||
         meta == &CustomFloatType<float8_e4m3>::dtype_meta ||
         meta == &CustomFloatType<float8_e4m3b11fnuz>::dtype_meta ||
         meta == &CustomFloatType<float8_e4m3fn>::dtype_meta ||
         meta == &CustomFloatType<float8_e4m3fnuz>::dtype_meta ||
         meta == &CustomFloatType<float8_e5m2>::dtype_meta ||
         meta == &CustomFloatType<float8_e5m2fnuz>::dtype_meta ||
         meta == &CustomFloatType<float6_e2m3fn>::dtype_meta ||
         meta == &CustomFloatType<float6_e3m2fn>::dtype_meta ||
         meta == &CustomFloatType<float4_e2m1fn>::dtype_meta ||
         meta == &CustomFloatType<float8_e8m0fnu>::dtype_meta;
}

template <typename T>
struct DtypeTraits<T, std::enable_if_t<is_custom_float_v<T>>> {
  static int Dtype() { return CustomFloatType<T>::Dtype(); }
};

bool RegisterCustomFloats(PyObject* numpy);

}  // namespace ml_dtypes

#endif  // ML_DTYPES_FLOATS_H_
