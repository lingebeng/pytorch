#pragma once
#include <c10/metal/common.h>

// Parameters for adaptive max pooling
// Supports 2D adaptive pooling with variable window sizes per output element
template <typename idx_type_t = int32_t>
struct AdaptiveMaxPoolParams {
  int32_t dims;           // Total number of dimensions (3 or 4)
  idx_type_t channels;    // Number of channels (N*C combined)
  idx_type_t input_height;
  idx_type_t input_width;
  idx_type_t output_height;
  idx_type_t output_width;
  // Input and output strides for non-contiguous tensors
  idx_type_t input_stride_n;   // Stride for batch*channel dimension
  idx_type_t input_stride_h;   // Stride for height
  idx_type_t input_stride_w;   // Stride for width
  idx_type_t output_stride_n;
  idx_type_t output_stride_h;
  idx_type_t output_stride_w;
  idx_type_t indices_stride_n;
  idx_type_t indices_stride_h;
  idx_type_t indices_stride_w;
};

template <typename idx_type_t = int32_t>
struct AdaptiveMaxPoolBackwardParams {
  int32_t dims;
  idx_type_t channels;
  idx_type_t input_height;
  idx_type_t input_width;
  idx_type_t output_height;
  idx_type_t output_width;
  idx_type_t grad_input_stride_n;
  idx_type_t grad_input_stride_h;
  idx_type_t grad_input_stride_w;
  idx_type_t grad_output_stride_n;
  idx_type_t grad_output_stride_h;
  idx_type_t grad_output_stride_w;
  idx_type_t indices_stride_n;
  idx_type_t indices_stride_h;
  idx_type_t indices_stride_w;
};
