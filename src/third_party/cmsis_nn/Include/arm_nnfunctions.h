/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nnfunctions.h
 * Description:  Public header file for CMSIS NN Library
 *
 * $Date:        30 September 2022
 * $Revision:    V.11.0.0
 *
 * Target Processor:  Cortex-M CPUs
 * -------------------------------------------------------------------- */

/**
   \mainpage CMSIS NN Software Library
   *
   * \tableofcontents
   * \section Introduction
   * 
   *
   * This user manual describes the CMSIS NN software library,
   * a collection of efficient neural network kernels developed to maximize the
   * performance and minimize the memory footprint of neural networks on Arm Cortex-M processors.
   *
   * The library is divided into a number of functions each covering a specific category:
   * - \ref NNConv Convolution Functions
   * - \ref Acti "Activation Functions"
   * - \ref FC Fully-connected Layer Functions
   * - \ref SVDF Layer Functions
   * - \ref Pooling Functions
   * - \ref Softmax Functions
   * - \ref groupElementwise Basic math Functions
   *
   *
   * \section Processors Supported Processors
   * 
   * CMSIS-NN targets Cortex-M processors with typically three different implementations for each function. Each
   * targets a different group of processors.
   *  - Processors without Single Instruction Multiple Data(SIMD) capability (e.g, Cortex-M0)
   *  - Processors with DSP extension (e.g Cortex-M4)
   *  - Processors with Arm M-Profile Vector Extension(MVE) instructions (e.g Cortex-M55)
   * The right implementation is picked through feature flags and the user does not have to explicit set it.
   *
   * \section Framework Quantization Specification
   * The library follows the [int8](https://www.tensorflow.org/lite/performance/quantization_spec) and int16
   *  quantization specification of TensorFlow Lite for Microcontrollers. 
   * \section Overview Block Diagram
   * 
   * \image html CMSIS-NN-OVERVIEW.PNG
   *
   * \section Examples
   * 
   *
   * An example image recognition application using TensorFlow Flow Lite for Microcontrollers as an inference engine
   * and CMSIS-NN as the optimized library can be found in the Examples directory.
   *
   * \section Macros Pre-processor Macros
   * 
   * \subsection Feature Feature flag based
   * The macros below are defined in a build system based on feature flags for a chosen processor or architecture
   * input to a compiler.
   * These tie in to the classification in \ref Macros. 
   * 
   * For a CMSIS-NN file compiled as *armclang -mcpu=cortex-m4 --target=arm-arm-none-eabi -I<CMSIS Core Include> 
   * -Ofast -O file.c* , ARM_MATH_DSP is enabled as Cortex-M4 has the DSP extension as a feature.
   * 
   * - `ARM_MATH_DSP`  - Selects code for processors with DSP extension.
   *
   * - `ARM_MATH_MVEI`  - Selects code for processors which supports MVE instructions.
   *
   * \subsection MiscFlags User Set 
   * - `ARM_MATH_AUTOVECTORIZE`
   *  Applicable when ARM_MATH_MVEI is active to let the compiler auto vectorize functions, if available, that uses inline
   *  assembly. This has to be explicitly set at compile time.
   *
   * \section Copyright Copyright Notice
   * 
   *
   * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
   *
   *
   */

/**
 * @defgroup Public Public
 * A collection of functions to perform basic operations for neural network layers. Functions with a _s8 suffix support
 * TensorFlow Lite framework.
 */

#ifndef _ARM_NNFUNCTIONS_H
#define _ARM_NNFUNCTIONS_H

#include "third_party/cmsis_nn/Include/arm_nn_math_types.h"
#include "third_party/cmsis_nn/Include/arm_nn_types.h"

#define USE_INTRINSIC

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Struct for specifying activation function types
 *
 */
typedef enum
{
    ARM_SIGMOID = 0,
    /**< Sigmoid activation function */
    ARM_TANH = 1,
    /**< Tanh activation function */
} arm_nn_activation_type;

/**
 * @defgroup NNConv Convolution Functions
 *
 * Collection of convolution, depthwise convolution functions and their variants.
 *
 * The convolution is implemented in 2 steps: im2col and General Matrix Multiplication(GEMM)
 *
 * im2col is a process of converting each patch of image data into
 * a column. After im2col, the convolution is computed as matrix-matrix
 * multiplication.
 *
 * To reduce the memory footprint, the im2col is performed partially.
 * Each iteration, only a few column (i.e., patches) are generated followed
 * by GEMM.
 *
 */

/**
 * @brief s8 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 *        cmsis-nn  to perform the convolution.
 *
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_wrapper_s8_get_buffer_size will return the buffer_size if required.
 *                                The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 */
arm_cmsis_nn_status arm_convolve_wrapper_s8(const cmsis_nn_context *ctx,
                                            const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const q7_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const q7_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            q7_t *output_data);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s8
 *
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      input_dims     Input (activation) dimensions. Format: [N, H, W, C_IN]
 * @param[in]      filter_dims    Filter dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the spatial
 *                                filter dimensions
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 *
 * @return         The function returns  required buffer size(bytes)
 *
 */
int32_t arm_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims);

/**
 * @brief s16 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 *        cmsis-nn to perform the convolution.
 *
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_wrapper_s8_get_buffer_size will return the buffer_size if required
 *                                The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Bias data pointer. Data type: int64
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int16
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 */
arm_cmsis_nn_status arm_convolve_wrapper_s16(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params *conv_params,
                                             const cmsis_nn_per_channel_quant_params *quant_params,
                                             const cmsis_nn_dims *input_dims,
                                             const q15_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const q7_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const int64_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             q15_t *output_data);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s16
 *
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      input_dims     Input (activation) dimensions. Format: [N, H, W, C_IN]
 * @param[in]      filter_dims    Filter dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the spatial
 *                                filter dimensions
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 *
 * @return         The function returns  required buffer size(bytes)
 *
 */
int32_t arm_convolve_wrapper_s16_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const cmsis_nn_dims *output_dims);

/**
 * @brief Basic s8 convolution function
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_s8_get_buffer_size will return the buffer_size if required.
 *                                The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8

 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *    3. Additional memory is required for optimization. Refer to argument 'ctx' for details.
 *
 */
arm_cmsis_nn_status arm_convolve_s8(const cmsis_nn_context *ctx,
                                    const cmsis_nn_conv_params *conv_params,
                                    const cmsis_nn_per_channel_quant_params *quant_params,
                                    const cmsis_nn_dims *input_dims,
                                    const q7_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const q7_t *filter_data,
                                    const cmsis_nn_dims *bias_dims,
                                    const int32_t *bias_data,
                                    const cmsis_nn_dims *output_dims,
                                    q7_t *output_data);

/**
 * @brief Get the required buffer size for s8 convolution function
 *
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims           Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 * are the spatial filter dimensions
 * @return          The function returns  required buffer size(bytes)
 *
 */
int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Basic s16 convolution function
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_s16_get_buffer_size will return the buffer_size if required.
 *                                The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int64
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int16

 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. q7/q15 is used as data type eventhough it is s8/s16 data. It is done so to be consistent with existing APIs.
 *    3. Additional memory is required for optimization. Refer to argument 'ctx' for details.
 *
 */
arm_cmsis_nn_status arm_convolve_s16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params *conv_params,
                                     const cmsis_nn_per_channel_quant_params *quant_params,
                                     const cmsis_nn_dims *input_dims,
                                     const q15_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const q7_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const int64_t *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     q15_t *output_data);
/**
 * @brief Optimized s16 convolution function
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_fast_s16_get_buffer_size will return the buffer_size if required.
 *                                The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions. (filter_dims->w * filter_dims->h * input_dims->c) must not
 exceed 512
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int64
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int16

 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. q7/q15 is used as data type eventhough it is s8/s16 data. It is done so to be consistent with existing APIs.
 *    3. Additional memory is required for optimization. Refer to argument 'ctx' for details.
 *    4. Implementation supports kernel volumes (filter width * filter height * input channels) < 512.
 *
 */

arm_cmsis_nn_status arm_convolve_fast_s16(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params *conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const q15_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const q7_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int64_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          q15_t *output_data);

/**
 * @brief Get the required buffer size for s16 convolution function
 *
 * @param[in]       input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims   Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 *                                are the spatial filter dimensions
 * @return          The function returns  required buffer size(bytes)
 *
 */
int32_t arm_convolve_s16_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Get the required buffer size for fast s16 convolution function
 *
 * @param[in]       input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims   Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 *                                are the spatial filter dimensions
 * @return          The function returns required buffer size(bytes)
 *
 */
int32_t arm_convolve_fast_s16_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Fast s8 version for 1x1 convolution (non-square shape)
 *
 * @param[in, out] ctx           Function context that contains the additional buffer if required by the function.
 *                               arm_convolve_1x1_s8_fast_get_buffer_size will return the buffer_size if required.
 *                               The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params   Convolution parameters (e.g. strides, dilations, pads,...).
 *                               Range of conv_params->input_offset  : [-127, 128]
 *                               Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Filter tensor dimensions. Format: [C_OUT, 1, 1, C_IN]
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data     Optional bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data   Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# input_dims->c is a multiple of 4
 *      -# conv_params->padding.w = conv_params->padding.h = 0
 *      -# conv_params->stride.w = conv_params->stride.h = 1
 *
 */
arm_cmsis_nn_status arm_convolve_1x1_s8_fast(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params *conv_params,
                                             const cmsis_nn_per_channel_quant_params *quant_params,
                                             const cmsis_nn_dims *input_dims,
                                             const q7_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const q7_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const int32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             q7_t *output_data);

/**
 * @brief Get the required buffer size for arm_convolve_1x1_s8_fast
 *
 * @param[in]       input_dims            Input (activation) dimensions
 * @return          The function returns the required buffer size in bytes
 *
 */
int32_t arm_convolve_1x1_s8_fast_get_buffer_size(const cmsis_nn_dims *input_dims);

/**
 * @brief 1xn convolution
 *
 * @param[in, out] ctx           Function context that contains the additional buffer if required by the function.
 *                               arm_convolve_1_x_n_s8_get_buffer_size will return the buffer_size if required
 *                               The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params   Convolution parameters (e.g. strides, dilations, pads,...).
 *                               Range of conv_params->input_offset  : [-127, 128]
 *                               Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Filter tensor dimensions. Format: [C_OUT, 1, WK, C_IN] where WK is the horizontal
 *                               spatial filter dimension
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data     Optional bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data   Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# input_dims->n equals 1
 *      -# ouput_dims->w is a multiple of 4
 *      -# Explicit constraints(since it is for 1xN convolution)
 *      -## input_dims->h equals 1
 *      -## output_dims->h equals 1
 *      -## filter_dims->h equals 1
 *@todo  Remove constraint on output_dims->w to make the function generic.
 *
 */
arm_cmsis_nn_status arm_convolve_1_x_n_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params *conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const q7_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const q7_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          q7_t *output_data);

/**
 * @brief Get the required additional buffer size for 1xn convolution
 *
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims           Filter tensor dimensions. Format: [C_OUT, 1, WK, C_IN] where WK is the
 *                                        horizontal spatial filter dimension
 * @return          The function returns  required buffer size(bytes)
 *
 */
int32_t arm_convolve_1_x_n_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Wrapper function to pick the right optimized s8 depthwise convolution function
 *
 * @param[in, out] ctx             Function context (e.g. temporary buffer). Check the function
 *                                 definition file to see if an additional buffer is required.
 *                                 Optional function {API}_get_buffer_size() provides the buffer
 *                                 size if required.
 *                                 The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 dw_conv_params->dilation is not used.
 *                                 Range of dw_conv_params->input_offset : [-127, 128]
 *                                 Range of dw_conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params    Per-channel quantization info.
 *                                 It contains the multiplier and shift values to be applied to each
 *                                 output channel
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                 Batch argument N is not used and assumed to be 1.
 * @param[in]      input_data      Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data     Filter data pointer. Data type: int8
 * @param[in]      bias_dims       Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data       Bias data pointer. Data type: int32
 * @param[in]      output_dims     Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in, out] output_data     Output data pointer. Data type: int8
 * @return     The function returns
 *                <code>ARM_CMSIS_NN_SUCCESS</code>   -  Successful completion.
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - Picks one of the the following functions
 *        -# arm_depthwise_conv_s8()
 *        -# arm_depthwise_conv_3x3_s8() - Cortex-M CPUs with DSP extension only
 *        -# arm_depthwise_conv_s8_opt()
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *    - Check details of arm_depthwise_conv_s8_opt() for potential data that can be accessed outside of the
 * boundary.
 */
arm_cmsis_nn_status arm_depthwise_conv_wrapper_s8(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_dw_conv_params *dw_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const q7_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const q7_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  q7_t *output_data);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s8()
 *
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 Range of dw_conv_params->input_offset : [-127, 128]
 *                                 Range of dw_conv_params->input_offset : [-128, 127]
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                 Batch argument N is not used and assumed to be 1.
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      output_dims     Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @return                         Size of additional memory required for optimizations in bytes.
 *
 */
int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const cmsis_nn_dims *output_dims);

/**
 * @brief Basic s8 depthwise convolution function that doesn't have any constraints on the input dimensions.
 *
 * @param[in, out] ctx             Function context (e.g. temporary buffer). Check the function
 *                                 definition file to see if an additional buffer is required.
 *                                 Optional function {API}_get_buffer_size() provides the buffer
 *                                 size if an additional buffer is required exists if additional memory is.
 *                                 The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 dw_conv_params->dilation is not used.
 *                                 Range of dw_conv_params->input_offset : [-127, 128]
 *                                 Range of dw_conv_params->input_offset : [-128, 127]
 * @param[in]      quant_params    Per-channel quantization info.
 *                                 It contains the multiplier and shift values to be applied to each
 *                                 output channel
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                                 Batch argument N is not used.
 * @param[in]      input_data      Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data     Filter data pointer. Data type: int8
 * @param[in]      bias_dims       Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data       Bias data pointer. Data type: int32
 * @param[in]      output_dims     Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[in, out] output_data     Output data pointer. Data type: int8
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 */
arm_cmsis_nn_status arm_depthwise_conv_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_dw_conv_params *dw_conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const q7_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const q7_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          q7_t *output_data);

/**
 * @brief Basic s16 depthwise convolution function that doesn't have any constraints on the input dimensions.
 *
 * @param[in, out] ctx             Function context (e.g. temporary buffer). Check the function
 *                                 definition file to see if an additional buffer is required.
 *                                 Optional function {API}_get_buffer_size() provides the buffer
 *                                 size if an additional buffer is required.
 *                                 exists if additional memory is.
 *                                 The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 conv_params->input_offset  : Not used
 *                                 conv_params->output_offset : Not used
 * @param[in]      quant_params    Per-channel quantization info.
 *                                 It contains the multiplier and shift values to be applied to each
 *                                 output channel
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                                 Batch argument N is not used.
 * @param[in]      input_data      Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data     Filter data pointer. Data type: int8
 * @param[in]      bias_dims       Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data       Bias data pointer. Data type: int64
 * @param[in]      output_dims     Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[in, out] output_data     Output data pointer. Data type: int16
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - q15 is used as data type eventhough it is s16 data. It is done so to be consistent with existing APIs.
 */
arm_cmsis_nn_status arm_depthwise_conv_s16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params *dw_conv_params,
                                           const cmsis_nn_per_channel_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const q15_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const q7_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const int64_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           q15_t *output_data);

/**
 * @brief Wrapper function to pick the right optimized s16 depthwise convolution function
 *
 * @param[in, out] ctx             Function context (e.g. temporary buffer). Check the function
 *                                 definition file to see if an additional buffer is required.
 *                                 Optional function {API}_get_buffer_size() provides the buffer
 *                                 size if required.
 *                                 The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 dw_conv_params->dilation is not used.
 *                                 Range of dw_conv_params->input_offset : Not used
 *                                 Range of dw_conv_params->output_offset : Not used
 * @param[in]      quant_params    Per-channel quantization info.
 *                                 It contains the multiplier and shift values to be applied to each
 *                                 output channel
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                 Batch argument N is not used and assumed to be 1.
 * @param[in]      input_data      Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data     Filter data pointer. Data type: int8
 * @param[in]      bias_dims       Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data       Bias data pointer. Data type: int64
 * @param[in]      output_dims     Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in, out] output_data     Output data pointer. Data type: int16
 * @return     The function returns
 *                <code>ARM_CMSIS_NN_SUCCESS</code>   -  Successful completion.
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - Picks one of the the following functions
 *        -# arm_depthwise_conv_s16()
 *        -# arm_depthwise_conv_fast_s16()  - Cortex-M CPUs with DSP extension only
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 */
arm_cmsis_nn_status arm_depthwise_conv_wrapper_s16(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_dw_conv_params *dw_conv_params,
                                                   const cmsis_nn_per_channel_quant_params *quant_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const q15_t *input_data,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const q7_t *filter_data,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const int64_t *bias_data,
                                                   const cmsis_nn_dims *output_dims,
                                                   q15_t *output_data);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s16()
 *
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 Range of dw_conv_params->input_offset : Not used
 *                                 Range of dw_conv_params->input_offset : Not used
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                 Batch argument N is not used and assumed to be 1.
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      output_dims     Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @return                         Size of additional memory required for optimizations in bytes.
 *
 */
int32_t arm_depthwise_conv_wrapper_s16_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const cmsis_nn_dims *output_dims);

/**
 * @brief Optimized s16 depthwise convolution function with constraint that in_channel equals out_channel.
 *        Refer arm_depthwise_conv_s16() for function argument details.
 *
 * @return     The function returns one of the following
 *                <code>ARM_CMSIS_NN_ARG_ERROR</code> - ctx-buff == NULL and
 *                                                      arm_depthwise_conv_fast_s16_get_buffer_size() > 0 or
 *                                                      input channel != output channel or
 *                                                      ch_mult != 1
 *
 *                <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - The following constrains on the arguments apply
 *        -# Number of input channel equals number of output channels or ch_mult equals 1
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *    - Reccomended when number of channels is 4 or greater.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_fast_s16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_dw_conv_params *dw_conv_params,
                                                const cmsis_nn_per_channel_quant_params *quant_params,
                                                const cmsis_nn_dims *input_dims,
                                                const q15_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const q7_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const int64_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                q15_t *output_data);

/**
 * @brief Get the required buffer size for optimized s16 depthwise convolution
 * function with constraint that in_channel equals out_channel.
 * @param[in]       input_dims   Input (activation) tensor dimensions. Format: [1, H, W, C_IN]
 *                               Batch argument N is not used.
 * @param[in]       filter_dims  Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @return          The function returns  required buffer size in bytes
 *
 */
int32_t arm_depthwise_conv_fast_s16_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Optimized s8 depthwise convolution function for 3x3 kernel size with some constraints on
 *        the input arguments(documented below). Refer arm_depthwise_conv_s8() for function
 *        argument details.
 *
 * @return     The function returns one of the following
 *                <code>ARM_CMSIS_NN_ARG_ERROR</code> - Unsupported dimension of tensors
 *                                                    - Unsupported pad size along the x axis
 *                <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# Number of input channel equals number of output channels
 *      -# Filter height and width equals 3
 *      -# Padding along x is either 0 or 1.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_3x3_s8(const cmsis_nn_context *ctx,
                                              const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const q7_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const q7_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              q7_t *output_data);

/**
 * @brief Optimized s8 depthwise convolution function with constraint that in_channel equals out_channel.
 *        Refer arm_depthwise_conv_s8() for function argument details.
 *
 * @return     The function returns one of the following
 *                <code>ARM_CMSIS_NN_ARG_ERROR</code> - input channel != output channel or
 *                                                      ch_mult != 1
 *                <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @note       If number of channels is not a multiple of 4, upto 3 elements outside the boundary will be read out
 *             for the following if MVE optimizations(Arm Helium Technology) are used.
 *               - Output shift
 *               - Output multiplier
 *               - Output bias
 *               - kernel
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - The following constrains on the arguments apply
 *        -# Number of input channel equals number of output channels or ch_mult equals 1
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *    - Reccomended when number of channels is 4 or greater.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_s8_opt(const cmsis_nn_context *ctx,
                                              const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const q7_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const q7_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              q7_t *output_data);

/**
 * @brief Get the required buffer size for optimized s8 depthwise convolution
 * function with constraint that in_channel equals out_channel.
 * @param[in]       input_dims   Input (activation) tensor dimensions. Format: [1, H, W, C_IN]
 *                               Batch argument N is not used.
 * @param[in]       filter_dims  Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @return          The function returns  required buffer size in bytes
 *
 */
int32_t arm_depthwise_conv_s8_opt_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @defgroup FC Fully-connected Layer Functions
 *
 * Collection of fully-connected and matrix multiplication functions.
 *
 * Fully-connected layer is basically a matrix-vector multiplication
 * with bias. The matrix is the weights and the input/output vectors
 * are the activation values. Supported {weight, activation} precisions
 * include {8-bit, 8-bit} and {8-bit, 16-bit}
 *
 *
 */

/**
 * @brief Basic s8 Fully Connected function.
 *
 * @param[in, out] ctx           Function context (e.g. temporary buffer). Check the function
 *                               definition file to see if an additional buffer is required.
 *                               Optional function {API}_get_buffer_size() provides the buffer
 *                               size if an additional buffer is required.
 *                               The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      fc_params     Fully Connected layer parameters.
 *                               Range of fc_params->input_offset  : [-127, 128]
 *                               fc_params->filter_offset : 0
 *                               Range of fc_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-tensor quantization info.
 *                               It contains the multiplier and shift values to be applied to the output tensor.
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                               Input dimension is taken as Nx(H * W * C_IN)
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Two dimensional filter dimensions. Format: [N, C]
 *                               N : accumulation depth and equals (H * W * C_IN) from input_dims
 *                               C : output depth and equals C_OUT in output_dims
 *                               H & W : Not used
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 *                               N, H, W : Not used
 * @param[in]      bias_data     Bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, C_OUT]
 *                               N : Batches
 *                               C_OUT : Output depth
 *                               H & W : Not used.
 * @param[in, out] output_data    Output data pointer. Data type: int8
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 */
arm_cmsis_nn_status arm_fully_connected_s8(const cmsis_nn_context *ctx,
                                           const cmsis_nn_fc_params *fc_params,
                                           const cmsis_nn_per_tensor_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const q7_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const q7_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const int32_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           q7_t *output_data);

/**
 * @brief Get the required buffer size for S8 basic fully-connected and
 * matrix multiplication layer function for TF Lite
 * @param[in]      filter_dims             dimension of filter
 * @return         The function returns    required buffer size in bytes
 *
 */
int32_t arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims *filter_dims);

/**
 * @brief Basic s16 Fully Connected function.
 *
 * @param[in, out] ctx           Function context (e.g. temporary buffer). Check the function
 *                               definition file to see if an additional buffer is required.
 *                               Optional function {API}_get_buffer_size() provides the buffer
 *                               size if an additional buffer is required.
 *                               The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      fc_params     Fully Connected layer parameters.
 *                               fc_params->input_offset  : 0
 *                               fc_params->filter_offset : 0
 *                               fc_params->output_offset : 0
 * @param[in]      quant_params  Per-tensor quantization info.
 *                               It contains the multiplier and shift values to be applied to the output tensor.
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                               Input dimension is taken as Nx(H * W * C_IN)
 * @param[in]      input_data    Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims   Two dimensional filter dimensions. Format: [N, C]
 *                               N : accumulation depth and equals (H * W * C_IN) from input_dims
 *                               C : output depth and equals C_OUT in output_dims
 *                               H & W : Not used
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 *                               N, H, W : Not used
 * @param[in]      bias_data     Bias data pointer. Data type: int64
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, C_OUT]
 *                               N : Batches
 *                               C_OUT : Output depth
 *                               H & W : Not used.
 * @param[in, out] output_data    Output data pointer. Data type: int16
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - q15 is used as data type eventhough it is s16 data. It is done so to be consistent with existing APIs.
 */
arm_cmsis_nn_status arm_fully_connected_s16(const cmsis_nn_context *ctx,
                                            const cmsis_nn_fc_params *fc_params,
                                            const cmsis_nn_per_tensor_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const q15_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const q7_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int64_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            q15_t *output_data);

/**
 * @brief Get the required buffer size for S16 basic fully-connected and
 * matrix multiplication layer function for TF Lite
 * @param[in]      filter_dims             dimension of filter
 * @return         The function returns    required buffer size in bytes
 *
 */
int32_t arm_fully_connected_s16_get_buffer_size(const cmsis_nn_dims *filter_dims);

/**
 * @brief Q7 opt fully-connected layer function
 * @param[in]       pV          pointer to input vector
 * @param[in]       pM          pointer to matrix weights
 * @param[in]       dim_vec     length of the vector
 * @param[in]       num_of_rows number of rows in weight matrix
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in]       bias        pointer to bias
 * @param[in,out]   pOut        pointer to output vector
 * @param[in,out]   vec_buffer  pointer to buffer space for input
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 */


/**
 * @defgroup groupElementwise Elementwise Functions
 *
 * Elementwise add and multiplication functions.
 *
 */

/**
 * @brief s8 elementwise add of two vectors
 * @param[in]       input_1_vect        pointer to input vector 1
 * @param[in]       input_2_vect        pointer to input vector 2
 * @param[in]       input_1_offset      offset for input 1. Range: -127 to 128
 * @param[in]       input_1_mult        multiplier for input 1
 * @param[in]       input_1_shift       shift for input 1
 * @param[in]       input_2_offset      offset for input 2. Range: -127 to 128
 * @param[in]       input_2_mult        multiplier for input 2
 * @param[in]       input_2_shift       shift for input 2
 * @param[in]       left_shift          input left shift
 * @param[in,out]   output              pointer to output vector
 * @param[in]       out_offset          output offset.  Range: -128 to 127
 * @param[in]       out_mult            output multiplier
 * @param[in]       out_shift           output shift
 * @param[in]       out_activation_min  minimum value to clamp output to. Min: -128
 * @param[in]       out_activation_max  maximum value to clamp output to. Max: 127
 * @param[in]       block_size          number of samples
 * @return          The function returns    ARM_CMSIS_NN_SUCCESS
 */
arm_cmsis_nn_status arm_elementwise_add_s8(const int8_t *input_1_vect,
                                           const int8_t *input_2_vect,
                                           const int32_t input_1_offset,
                                           const int32_t input_1_mult,
                                           const int32_t input_1_shift,
                                           const int32_t input_2_offset,
                                           const int32_t input_2_mult,
                                           const int32_t input_2_shift,
                                           const int32_t left_shift,
                                           int8_t *output,
                                           const int32_t out_offset,
                                           const int32_t out_mult,
                                           const int32_t out_shift,
                                           const int32_t out_activation_min,
                                           const int32_t out_activation_max,
                                           const int32_t block_size);

/**
 * @brief s16 elementwise add of two vectors
 * @param[in]       input_1_vect        pointer to input vector 1
 * @param[in]       input_2_vect        pointer to input vector 2
 * @param[in]       input_1_offset      offset for input 1. Not used.
 * @param[in]       input_1_mult        multiplier for input 1
 * @param[in]       input_1_shift       shift for input 1
 * @param[in]       input_2_offset      offset for input 2. Not used.
 * @param[in]       input_2_mult        multiplier for input 2
 * @param[in]       input_2_shift       shift for input 2
 * @param[in]       left_shift          input left shift
 * @param[in,out]   output              pointer to output vector
 * @param[in]       out_offset          output offset. Not used.
 * @param[in]       out_mult            output multiplier
 * @param[in]       out_shift           output shift
 * @param[in]       out_activation_min  minimum value to clamp output to. Min: -32768
 * @param[in]       out_activation_max  maximum value to clamp output to. Max: 32767
 * @param[in]       block_size          number of samples
 * @return          The function returns  ARM_CMSIS_NN_SUCCESS
 */
arm_cmsis_nn_status arm_elementwise_add_s16(const int16_t *input_1_vect,
                                            const int16_t *input_2_vect,
                                            const int32_t input_1_offset,
                                            const int32_t input_1_mult,
                                            const int32_t input_1_shift,
                                            const int32_t input_2_offset,
                                            const int32_t input_2_mult,
                                            const int32_t input_2_shift,
                                            const int32_t left_shift,
                                            int16_t *output,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max,
                                            const int32_t block_size);

/**
 * @brief s8 elementwise multiplication
 * @param[in]       input_1_vect        pointer to input vector 1
 * @param[in]       input_2_vect        pointer to input vector 2
 * @param[in]       input_1_offset      offset for input 1. Range: -127 to 128
 * @param[in]       input_2_offset      offset for input 2. Range: -127 to 128
 * @param[in,out]   output              pointer to output vector
 * @param[in]       out_offset          output offset. Range: -128 to 127
 * @param[in]       out_mult            output multiplier
 * @param[in]       out_shift           output shift
 * @param[in]       out_activation_min  minimum value to clamp output to. Min: -128
 * @param[in]       out_activation_max  maximum value to clamp output to. Max: 127
 * @param[in]       block_size          number of samples
 * @return          The function returns ARM_CMSIS_NN_SUCCESS
 *
 * @details   Supported framework: TensorFlow Lite micro
 */
arm_cmsis_nn_status arm_elementwise_mul_s8(const int8_t *input_1_vect,
                                           const int8_t *input_2_vect,
                                           const int32_t input_1_offset,
                                           const int32_t input_2_offset,
                                           int8_t *output,
                                           const int32_t out_offset,
                                           const int32_t out_mult,
                                           const int32_t out_shift,
                                           const int32_t out_activation_min,
                                           const int32_t out_activation_max,
                                           const int32_t block_size);

/**
 * @brief s16 elementwise multiplication
 * @param[in]       input_1_vect        pointer to input vector 1
 * @param[in]       input_2_vect        pointer to input vector 2
 * @param[in]       input_1_offset      offset for input 1. Not used.
 * @param[in]       input_2_offset      offset for input 2. Not used.
 * @param[in,out]   output              pointer to output vector
 * @param[in]       out_offset          output offset. Not used.
 * @param[in]       out_mult            output multiplier
 * @param[in]       out_shift           output shift
 * @param[in]       out_activation_min  minimum value to clamp output to. Min: -32768
 * @param[in]       out_activation_max  maximum value to clamp output to. Max: 32767
 * @param[in]       block_size          number of samples
 * @return          The function returns ARM_CMSIS_NN_SUCCESS
 *
 * @details   Supported framework: TensorFlow Lite micro
 */
arm_cmsis_nn_status arm_elementwise_mul_s16(const int16_t *input_1_vect,
                                            const int16_t *input_2_vect,
                                            const int32_t input_1_offset,
                                            const int32_t input_2_offset,
                                            int16_t *output,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max,
                                            const int32_t block_size);

/**
 * @defgroup Acti Activation Functions
 *
 * Perform activation layers, including ReLU (Rectified Linear Unit),
 * sigmoid and tanh
 *
 */

/**
 * @brief Q7 RELU function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 */

void arm_relu_q7(q7_t *data, uint16_t size);

/**
 * @brief s8 ReLU6 function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 */

void arm_relu6_s8(q7_t *data, uint16_t size);

/**
 * @brief Q15 RELU function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 */

void arm_relu_q15(q15_t *data, uint16_t size);

/**
 * @brief s8 average pooling function.
 *
 * @param[in, out] ctx          Function context (e.g. temporary buffer). Check the function
 *                              definition file to see if an additional buffer is required.
 *                              Optional function {API}_get_buffer_size() provides the buffer
 *                              size if an additional buffer is required.
 *                              The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      pool_params  Pooling parameters
 * @param[in]      input_dims   Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                              Argument 'N' is not used.
 * @param[in]      input_data   Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims  Filter tensor dimensions. Format: [H, W]
 *                              Argument N and C are not used.
 * @param[in]      output_dims  Output tensor dimensions. Format: [H, W, C_OUT]
 *                              Argument N is not used.
 *                              C_OUT equals C_IN.
 * @param[in, out] output_data Output data pointer. Data type: int8
 * @return                     The function returns
 *                             <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *
 */
arm_cmsis_nn_status arm_avgpool_s8(const cmsis_nn_context *ctx,
                                   const cmsis_nn_pool_params *pool_params,
                                   const cmsis_nn_dims *input_dims,
                                   const q7_t *input_data,
                                   const cmsis_nn_dims *filter_dims,
                                   const cmsis_nn_dims *output_dims,
                                   q7_t *output_data);

/**
 * @brief Get the required buffer size for S8 average pooling function
 * @param[in]       dim_dst_width         output tensor dimension
 * @param[in]       ch_src                number of input tensor channels
 * @return          The function returns  required buffer size in bytes
 *
 */
int32_t arm_avgpool_s8_get_buffer_size(const int dim_dst_width, const int ch_src);

/**
 * @brief s16 average pooling function.
 *
 * @param[in, out] ctx          Function context (e.g. temporary buffer). Check the function
 *                              definition file to see if an additional buffer is required.
 *                              Optional function {API}_get_buffer_size() provides the buffer
 *                              size if an additional buffer is required.
 *                              The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      pool_params  Pooling parameters
 * @param[in]      input_dims   Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                              Argument 'N' is not used.
 * @param[in]      input_data   Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims  Filter tensor dimensions. Format: [H, W]
 *                              Argument N and C are not used.
 * @param[in]      output_dims  Output tensor dimensions. Format: [H, W, C_OUT]
 *                              Argument N is not used.
 *                              C_OUT equals C_IN.
 * @param[in, out] output_data  Output data pointer. Data type: int16
 * @return                        The function returns
 *                                    <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *                                    <code>ARM_CMSIS_NN_ARG_ERROR</code> - In case of invalid arguments
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *
 */
arm_cmsis_nn_status arm_avgpool_s16(const cmsis_nn_context *ctx,
                                    const cmsis_nn_pool_params *pool_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int16_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const cmsis_nn_dims *output_dims,
                                    int16_t *output_data);

/**
 * @brief Get the required buffer size for S16 average pooling function
 * @param[in]       dim_dst_width         output tensor dimension
 * @param[in]       ch_src                number of input tensor channels
 * @return          The function returns  required buffer size in bytes
 *
 */
int32_t arm_avgpool_s16_get_buffer_size(const int dim_dst_width, const int ch_src);

/**
 * @brief s8 max pooling function.
 *
 * @param[in, out] ctx          Function context (e.g. temporary buffer). Check the function
 *                              definition file to see if an additional buffer is required.
 *                              Optional function {API}_get_buffer_size() provides the buffer
 *                              size if an additional buffer is required.
 *                              The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      pool_params  Pooling parameters
 * @param[in]      input_dims   Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                              Argument 'N' is not used.
 * @param[in]      input_data   Input (activation) data pointer. The input tensor must not
 *                              overlap with the output tensor. Data type: int8
 * @param[in]      filter_dims  Filter tensor dimensions. Format: [H, W]
 *                              Argument N and C are not used.
 * @param[in]      output_dims  Output tensor dimensions. Format: [H, W, C_OUT]
 *                              Argument N is not used.
 *                              C_OUT equals C_IN.
 * @param[in, out] output_data    Output data pointer. Data type: int8
 * @return                        The function returns
 *                                    <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *
 */
arm_cmsis_nn_status arm_max_pool_s8(const cmsis_nn_context *ctx,
                                    const cmsis_nn_pool_params *pool_params,
                                    const cmsis_nn_dims *input_dims,
                                    const q7_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const cmsis_nn_dims *output_dims,
                                    q7_t *output_data);

/**
 * @brief s16 max pooling function.
 *
 * @param[in, out] ctx          Function context (e.g. temporary buffer). Check the function
 *                              definition file to see if an additional buffer is required.
 *                              Optional function {API}_get_buffer_size() provides the buffer
 *                              size if an additional buffer is required.
 *                              The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      pool_params  Pooling parameters
 * @param[in]      input_dims   Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                              Argument 'N' is not used.
 * @param[in]      src          Input (activation) data pointer. The input tensor must not
 *                              overlap with the output tensor. Data type: int16
 * @param[in]      filter_dims  Filter tensor dimensions. Format: [H, W]
 *                              Argument N and C are not used.
 * @param[in]      output_dims  Output tensor dimensions. Format: [H, W, C_OUT]
 *                              Argument N is not used.
 *                              C_OUT equals C_IN.
 * @param[in, out] dst          Output data pointer. Data type: int16
 * @return                        The function returns
 *                                    <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *
 */
arm_cmsis_nn_status arm_max_pool_s16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const int16_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     int16_t *dst);

/**
 * @defgroup Softmax Softmax Functions
 *
 *
 */

/**
 * @brief S8 softmax function
 * @param[in]  input     Pointer to the input tensor
 * @param[in]  num_rows  Number of rows in the input tensor
 * @param[in]  row_size  Number of elements in each input row
 * @param[in]  mult      Input quantization multiplier
 * @param[in]  shift     Input quantization shift within the range [0, 31]
 * @param[in]  diff_min  Minimum difference with max in row. Used to check if
 *                       the quantized exponential operation can be performed
 * @param[out] output    Pointer to the output tensor
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
void arm_softmax_s8(const int8_t *input,
                    const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int32_t diff_min,
                    int8_t *output);

/**
 * @brief S8 to s16 softmax function
 * @param[in]  input     Pointer to the input tensor
 * @param[in]  num_rows  Number of rows in the input tensor
 * @param[in]  row_size  Number of elements in each input row
 * @param[in]  mult      Input quantization multiplier
 * @param[in]  shift     Input quantization shift within the range [0, 31]
 * @param[in]  diff_min  Minimum difference with max in row. Used to check if
 *                       the quantized exponential operation can be performed
 * @param[out] output    Pointer to the output tensor
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
void arm_softmax_s8_s16(const int8_t *input,
                        const int32_t num_rows,
                        const int32_t row_size,
                        const int32_t mult,
                        const int32_t shift,
                        const int32_t diff_min,
                        int16_t *output);

/**
 * @brief S16 softmax function
 * @param[in]  input           Pointer to the input tensor
 * @param[in]  num_rows        Number of rows in the input tensor
 * @param[in]  row_size        Number of elements in each input row
 * @param[in]  mult            Input quantization multiplier
 * @param[in]  shift           Input quantization shift within the range [0, 31]
 * @param[in]  softmax_params  Softmax s16 layer parameters with two pointers to LUTs speficied below.
 *                             For indexing the high 9 bits are used and 7 remaining for interpolation.
 *                             That means 512 entries for the 9-bit indexing and 1 extra for interpolation, i.e. 513
 *                             values for each LUT.
 *                             - Lookup table for exp(x), where x uniform distributed between [-10.0 , 0.0]
 *                             - Lookup table for 1 / (1 + x), where x uniform distributed between [0.0 , 1.0]
 * @param[out] output          Pointer to the output tensor
 * @return                        The function returns
 *                                    <code>ARM_CMSIS_NN_ARG_ERROR</code> Argument error check failed
 *                                    <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
arm_cmsis_nn_status arm_softmax_s16(const int16_t *input,
                                    const int32_t num_rows,
                                    const int32_t row_size,
                                    const int32_t mult,
                                    const int32_t shift,
                                    const cmsis_nn_softmax_lut_s16 *softmax_params,
                                    int16_t *output);

/**
 * @brief U8 softmax function
 * @param[in]  input     Pointer to the input tensor
 * @param[in]  num_rows  Number of rows in the input tensor
 * @param[in]  row_size  Number of elements in each input row
 * @param[in]  mult      Input quantization multiplier
 * @param[in]  shift     Input quantization shift within the range [0, 31]
 * @param[in]  diff_min  Minimum difference with max in row. Used to check if
 *                       the quantized exponential operation can be performed
 * @param[out] output    Pointer to the output tensor
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */

void arm_softmax_u8(const uint8_t *input,
                    const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int32_t diff_min,
                    uint8_t *output);


/**
 * @defgroup Reshape Reshape Functions
 *
 */

/**
 * @brief Reshape a s8 vector into another with different shape
 * @param[in]  input      points to the s8 input vector
 * @param[out] output     points to the s8 output vector
 * @param[in]  total_size total size of the input and output vectors in bytes
 *
 * @note The output is expected to be in a memory area that does not overlap with the input's
 *
 */
void arm_reshape_s8(const int8_t *input, int8_t *output, const uint32_t total_size);

/**
 * @defgroup Concatenation Concatenation Functions
 *
 */

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the X axis
 *        This function should be called for each input tensor to concatenate. The argument offset_x
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_x = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    arm_concatenation_s8_x(&input[i], ..., &output, ..., ..., offset_x)
 *                    offset_x += input_x[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same height of the input tensor
 *        -# The same number of channels of the input tensor
 *        -# The same batch size of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *      does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor. Input tensor must not overlap with the output tensor.
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          (input_x * input_y * input_z * input_w) + offset_x
 *                      bytes.
 * @param[in]  output_x Width of output tensor
 * @param[in]  offset_x The offset (in number of elements) on the X axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 * <b> Input constraints</b>
 * offset_x is less than output_x
 *
 */
void arm_concatenation_s8_x(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint16_t output_x,
                            const uint32_t offset_x);

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the Y axis
 *        This function should be called for each input tensor to concatenate. The argument offset_y
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_y = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    arm_concatenation_s8_y(&input[i], ..., &output, ..., ..., offset_y)
 *                    offset_y += input_y[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same width of the input tensor
 *        -# The same number of channels of the input tensor
 *        -# The same batch size of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *       does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor. Input tensor must not overlap with the output tensor.
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          (input_z * input_w * input_x * input_y) + offset_y
 *                      bytes.
 * @param[in]  output_y Height of output tensor
 * @param[in]  offset_y The offset on the Y axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 * <b> Input constraints</b>
 * offset_y is less than output_y
 *
 */
void arm_concatenation_s8_y(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint16_t output_y,
                            const uint32_t offset_y);

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the Z axis
 *        This function should be called for each input tensor to concatenate. The argument offset_z
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_z = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    arm_concatenation_s8_z(&input[i], ..., &output, ..., ..., offset_z)
 *                    offset_z += input_z[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same width of the input tensor
 *        -# The same height of the input tensor
 *        -# The same batch size of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *       does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor. Input tensor must not overlap with output tensor.
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          (input_x * input_y * input_z * input_w) + offset_z
 *                      bytes.
 * @param[in]  output_z Channels in output tensor
 * @param[in]  offset_z The offset on the Z axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 * <b> Input constraints</b>
 * offset_z is less than output_z
 *
 */
void arm_concatenation_s8_z(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint16_t output_z,
                            const uint32_t offset_z);

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the W axis (Batch size)
 *        This function should be called for each input tensor to concatenate. The argument offset_w
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_w = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    arm_concatenation_s8_w(&input[i], ..., &output, ..., ..., offset_w)
 *                    offset_w += input_w[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same width of the input tensor
 *        -# The same height of the input tensor
 *        -# The same number o channels of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *       does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          input_x * input_y * input_z * input_w
 *                      bytes.
 * @param[in]  offset_w The offset on the W axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 */
void arm_concatenation_s8_w(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint32_t offset_w);
/**
 * @defgroup SVDF SVDF Functions
 *
 */

/**
 * @brief s8 SVDF function with 8 bit state tensor and 8 bit time weights
 *
 * @param[in]   input_ctx             Temporary scratch buffer
 *                                    The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]   output_ctx            Temporary output scratch buffer
 *                                    The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]   svdf_params           SVDF Parameters
 *                                    Range of svdf_params->input_offset  : [-128, 127]
 *                                    Range of svdf_params->output_offset  : [-128, 127]
 * @param[in]   input_quant_params    Input quantization parameters
 * @param[in]   output_quant_params   Output quantization parameters
 * @param[in]   input_dims            Input tensor dimensions
 * @param[in]   input_data            Pointer to input tensor
 * @param[in]   state_dims            State tensor dimensions
 * @param[in]   state_data            Pointer to state tensor
 * @param[in]   weights_feature_dims  Weights (feature) tensor dimensions
 * @param[in]   weights_feature_data  Pointer to the weights (feature) tensor
 * @param[in]   weights_time_dims     Weights (time) tensor dimensions
 * @param[in]   weights_time_data     Pointer to the weights (time) tensor
 * @param[in]   bias_dims             Bias tensor dimensions
 * @param[in]   bias_data             Pointer to bias tensor
 * @param[in]   output_dims           Output tensor dimensions
 * @param[out]  output_data           Pointer to the output tensor
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *
 */
arm_cmsis_nn_status arm_svdf_s8(const cmsis_nn_context *input_ctx,
                                const cmsis_nn_context *output_ctx,
                                const cmsis_nn_svdf_params *svdf_params,
                                const cmsis_nn_per_tensor_quant_params *input_quant_params,
                                const cmsis_nn_per_tensor_quant_params *output_quant_params,
                                const cmsis_nn_dims *input_dims,
                                const q7_t *input_data,
                                const cmsis_nn_dims *state_dims,
                                q7_t *state_data,
                                const cmsis_nn_dims *weights_feature_dims,
                                const q7_t *weights_feature_data,
                                const cmsis_nn_dims *weights_time_dims,
                                const q7_t *weights_time_data,
                                const cmsis_nn_dims *bias_dims,
                                const q31_t *bias_data,
                                const cmsis_nn_dims *output_dims,
                                q7_t *output_data);

/**
 * @brief s8 SVDF function with 16 bit state tensor and 16 bit time weights
 *
 * @param[in]   input_ctx             Temporary scratch buffer
 *                                    The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]   output_ctx            Temporary output scratch buffer
 *                                    The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]   svdf_params           SVDF Parameters
 *                                    Range of svdf_params->input_offset  : [-128, 127]
 *                                    Range of svdf_params->output_offset  : [-128, 127]
 * @param[in]   input_quant_params    Input quantization parameters
 * @param[in]   output_quant_params   Output quantization parameters
 * @param[in]   input_dims            Input tensor dimensions
 * @param[in]   input_data            Pointer to input tensor
 * @param[in]   state_dims            State tensor dimensions
 * @param[in]   state_data            Pointer to state tensor
 * @param[in]   weights_feature_dims  Weights (feature) tensor dimensions
 * @param[in]   weights_feature_data  Pointer to the weights (feature) tensor
 * @param[in]   weights_time_dims     Weights (time) tensor dimensions
 * @param[in]   weights_time_data     Pointer to the weights (time) tensor
 * @param[in]   bias_dims             Bias tensor dimensions
 * @param[in]   bias_data             Pointer to bias tensor
 * @param[in]   output_dims           Output tensor dimensions
 * @param[out]  output_data           Pointer to the output tensor
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *
 */
arm_cmsis_nn_status arm_svdf_state_s16_s8(const cmsis_nn_context *input_ctx,
                                          const cmsis_nn_context *output_ctx,
                                          const cmsis_nn_svdf_params *svdf_params,
                                          const cmsis_nn_per_tensor_quant_params *input_quant_params,
                                          const cmsis_nn_per_tensor_quant_params *output_quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const q7_t *input_data,
                                          const cmsis_nn_dims *state_dims,
                                          q15_t *state_data,
                                          const cmsis_nn_dims *weights_feature_dims,
                                          const q7_t *weights_feature_data,
                                          const cmsis_nn_dims *weights_time_dims,
                                          const q15_t *weights_time_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const q31_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          q7_t *output_data);

#ifdef __cplusplus
}
#endif

#endif
