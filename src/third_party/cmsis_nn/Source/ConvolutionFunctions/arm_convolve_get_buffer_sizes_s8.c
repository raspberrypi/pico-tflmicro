/*
 * SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_convolve_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for the various s8 convolution layer functions.
 *
 * $Date:        30 October 2023
 * $Revision:    V.1.4.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/Internal/arm_nn_compiler.h"
#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"

/**
 *  @ingroup NNConv
 */

/**
 * @addtogroup GetBufferSizeNNConv
 * @{
 */

__STATIC_INLINE int32_t arm_convolve_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims,
                                                            const cmsis_nn_dims *filter_dims)
{
    int32_t col_length = input_dims->c * filter_dims->w * filter_dims->h;
    // Get number of complete int16 lanes(multiple of 8) for given col_length. This is dependent on
    // implementation of  arm_nn_mat_mult_s8
    col_length = (col_length + 7) / 8;
    // 4 -> number of im2col buffers, 8 -> 8 elements per Q register
    return 4 * col_length * 8 * (int32_t)sizeof(int8_t);
}

__STATIC_INLINE int32_t arm_convolve_1_x_n_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims,
                                                                  const cmsis_nn_dims *filter_dims)
{
    (void)input_dims;
    (void)filter_dims;
    return 0;
}

int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_convolve_s8_get_buffer_size_mve(input_dims, filter_dims);
#else
    const int32_t rhs_cols = filter_dims->w * filter_dims->h * input_dims->c;
    const int32_t remainder = rhs_cols % 4;
    const int32_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;
    return (2 * aligned_rhs_cols) * (int32_t)sizeof(int16_t);
#endif
}

int32_t arm_convolve_1_x_n_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if !defined(ARM_MATH_MVEI)
    return arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
#else
    return arm_convolve_1_x_n_s8_get_buffer_size_mve(input_dims, filter_dims);
#endif
}

int32_t arm_convolve_1x1_s8_fast_get_buffer_size(const cmsis_nn_dims *input_dims)
{
    (void)input_dims;
    return 0;
}

/*
 * Get the required buffer size for arm_convolve_wrapper_s8. This is the recommended function convolve wrapper s8
 * function.
 *
 * Refer to header file for details.
 *
 */
int32_t arm_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_convolve_wrapper_s8_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
#else
    (void)output_dims;
    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (filter_dims->w == 1) &&
        (filter_dims->h == 1) && (conv_params->dilation.w == 1 && conv_params->dilation.h == 1))
    {
        if ((conv_params->stride.w == 1) && (conv_params->stride.h == 1))
        {
            return arm_convolve_1x1_s8_fast_get_buffer_size(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else if ((input_dims->h == 1) && (conv_params->dilation.w == 1) && (filter_dims->h == 1) &&
             (conv_params->stride.w * input_dims->c % 4 == 0))
    {
        return arm_convolve_1_x_n_s8_get_buffer_size(input_dims, filter_dims);
    }
    else
    {
        return arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
    }
#endif
}

int32_t arm_convolve_wrapper_s8_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims)
{
    (void)output_dims;
    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (filter_dims->w == 1) &&
        (filter_dims->h == 1) && (conv_params->dilation.w == 1 && conv_params->dilation.h == 1))
    {
        if ((conv_params->stride.w == 1) && (conv_params->stride.h == 1))
        {
            return arm_convolve_1x1_s8_fast_get_buffer_size(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else if ((input_dims->h == 1) && (conv_params->dilation.w == 1) && (filter_dims->h == 1) &&
             (conv_params->stride.w * input_dims->c % 4 == 0))
    {
        return arm_convolve_1_x_n_s8_get_buffer_size_mve(input_dims, filter_dims);
    }
    else
    {
        return arm_convolve_s8_get_buffer_size_mve(input_dims, filter_dims);
    }
}

int32_t arm_convolve_wrapper_s8_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims)
{
    return arm_convolve_wrapper_s8_get_buffer_size(conv_params, input_dims, filter_dims, output_dims);
}

/**
 * @} end of GetBufferSizeNNConv group
 */
