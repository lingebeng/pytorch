#include <ATen/native/mps/kernels/AdaptivePooling.h>
#include <c10/metal/atomic.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

// Adaptive pooling index calculation functions
// These match the CPU implementation in ATen/native/AdaptivePooling.h
//
// start_index: computes the start of the input window for output index 'a'
//   Formula: (a / b) * c + ((a % b) * c) / b
//   This is equivalent to: floor(a * c / b)
//
// end_index: computes the end of the input window for output index 'a'
//   Formula: 1 + ((a + 1) * c - 1) / b
//   This is equivalent to: ceil((a + 1) * c / b)

inline int32_t adaptive_start_index(int32_t a, int32_t b, int32_t c) {
  return (a / b) * c + ((a % b) * c) / b;
}

inline int32_t adaptive_end_index(int32_t a, int32_t b, int32_t c) {
  return 1 + ((a + 1) * c - 1) / b;
}

// Adaptive max pooling 2D forward kernel
// Each thread computes one output element
template <typename T>
kernel void adaptive_max_pool2d(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device int64_t* indices [[buffer(2)]],
    constant AdaptiveMaxPoolParams<int32_t>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {

  auto channels = params.channels;
  auto input_height = params.input_height;
  auto input_width = params.input_width;
  auto output_height = params.output_height;
  auto output_width = params.output_width;

  // Compute output element coordinates from thread ID
  // tid = c * output_height * output_width + oh * output_width + ow
  auto output_hw = output_height * output_width;
  auto c = static_cast<int32_t>(tid / output_hw);
  auto remainder = static_cast<int32_t>(tid % output_hw);
  auto oh = remainder / output_width;
  auto ow = remainder % output_width;

  // Bounds check
  if (c >= channels) {
    return;
  }

  // Compute input window bounds using adaptive pooling formulas
  auto ih_start = adaptive_start_index(oh, output_height, input_height);
  auto ih_end = adaptive_end_index(oh, output_height, input_height);
  auto iw_start = adaptive_start_index(ow, output_width, input_width);
  auto iw_end = adaptive_end_index(ow, output_width, input_width);

  // Compute input and output offsets
  auto input_offset = c * params.input_stride_n;
  auto output_offset = c * params.output_stride_n +
                       oh * params.output_stride_h +
                       ow * params.output_stride_w;
  auto indices_offset = c * params.indices_stride_n +
                        oh * params.indices_stride_h +
                        ow * params.indices_stride_w;

  // Find maximum value and its index in the input window
  T max_val = input[input_offset + ih_start * params.input_stride_h + iw_start * params.input_stride_w];
  int64_t max_idx = ih_start * input_width + iw_start;

  for (auto ih = ih_start; ih < ih_end; ih++) {
    for (auto iw = iw_start; iw < iw_end; iw++) {
      auto input_idx = input_offset + ih * params.input_stride_h + iw * params.input_stride_w;
      T val = input[input_idx];

      // Update max if current value is greater or is NaN
      // NaN propagation: if val is NaN, it should become the max
      if (val > max_val || isnan(val)) {
        max_val = val;
        max_idx = ih * input_width + iw;
      }
    }
  }

  // Store results
  output[output_offset] = max_val;
  indices[indices_offset] = max_idx;
}

// Adaptive max pooling 2D backward kernel
template <typename T>
kernel void adaptive_max_pool2d_backward(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant int64_t* indices [[buffer(2)]],
    constant AdaptiveMaxPoolBackwardParams<int32_t>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {

  auto channels = params.channels;
  auto input_height = params.input_height;
  auto input_width = params.input_width;
  auto output_height = params.output_height;
  auto output_width = params.output_width;

  // Compute output element coordinates from thread ID
  auto output_hw = output_height * output_width;
  auto c = static_cast<int32_t>(tid / output_hw);
  auto remainder = static_cast<int32_t>(tid % output_hw);
  auto oh = remainder / output_width;
  auto ow = remainder % output_width;

  if (c >= channels) {
    return;
  }

  // Get gradient output value and the index it came from
  auto grad_output_offset = c * params.grad_output_stride_n +
                            oh * params.grad_output_stride_h +
                            ow * params.grad_output_stride_w;
  auto indices_offset = c * params.indices_stride_n +
                        oh * params.indices_stride_h +
                        ow * params.indices_stride_w;

  T grad_val = grad_output[grad_output_offset];
  int64_t max_idx = indices[indices_offset];

  // Convert flat index to ih, iw
  auto ih = static_cast<int32_t>(max_idx / input_width);
  auto iw = static_cast<int32_t>(max_idx % input_width);

  // Compute grad_input offset and atomically add the gradient
  auto grad_input_offset = c * params.grad_input_stride_n +
                           ih * params.grad_input_stride_h +
                           iw * params.grad_input_stride_w;

  AtomicType<T>::atomic_add(grad_input, grad_input_offset, grad_val);
}

// Template instantiations
#define REGISTER_ADAPTIVE_POOL_OP(DTYPE)                                              \
  template [[host_name("adaptive_max_pool2d_" #DTYPE)]]                               \
  kernel void adaptive_max_pool2d<DTYPE>(                                             \
      constant DTYPE* input [[buffer(0)]],                                            \
      device DTYPE* output [[buffer(1)]],                                             \
      device int64_t* indices [[buffer(2)]],                                          \
      constant AdaptiveMaxPoolParams<int32_t>& params [[buffer(3)]],                  \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_ADAPTIVE_POOL_BACKWARD_OP(DTYPE)                                     \
  template [[host_name("adaptive_max_pool2d_backward_" #DTYPE)]]                      \
  kernel void adaptive_max_pool2d_backward<DTYPE>(                                    \
      device AtomicType_t<DTYPE>* grad_input [[buffer(0)]],                           \
      constant DTYPE* grad_output [[buffer(1)]],                                      \
      constant int64_t* indices [[buffer(2)]],                                        \
      constant AdaptiveMaxPoolBackwardParams<int32_t>& params [[buffer(3)]],          \
      uint tid [[thread_position_in_grid]]);

REGISTER_ADAPTIVE_POOL_OP(float);
REGISTER_ADAPTIVE_POOL_OP(half);
REGISTER_ADAPTIVE_POOL_OP(bfloat);

REGISTER_ADAPTIVE_POOL_BACKWARD_OP(float);
REGISTER_ADAPTIVE_POOL_BACKWARD_OP(half);
REGISTER_ADAPTIVE_POOL_BACKWARD_OP(bfloat);
