/*
 * Copyright (C) 2010-2020 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_depthwise_conv_3x3_s8.c
 * Description:  Optimized s8 depthwise convolution function for channel
 *               multiplier of 1 and 3x3 kernel size.
 *
 * $Date:        May 14, 2020
 * $Revision:    V.2.0.0
 *
 * Target Processor:  Cortex-M CPUs
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnsupportfunctions.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

typedef struct {
    int32_t core_number;
    const q7_t *input;
    const q7_t *kernel;
    const int32_t *bias;
    q7_t *output;
    int32_t input_x;
    int32_t input_y;
    int32_t input_ch;
    int32_t output_ch;
    int32_t pad_x;
    int32_t pad_y;
    int32_t stride_x;
    int32_t stride_y;
    const int32_t *output_shift;
    const int32_t *output_mult;
    int32_t output_x;
    int32_t output_y;
    int32_t output_offset;
    int32_t input_offset;
    int32_t output_activation_min;
    int32_t output_activation_max;
} DepthwiseConv3x3Args;

/*
   * Optimized s8 depthwise convolution function with constraint that
   * in_channel == out_channel and kernel_x == kernel_y == 3 with pads at most 1
   *
   *  Refer prototype header file for details.
   *
   */

static DepthwiseConv3x3Args g_core1_depthwise_conv_3x3_args;

static void depthwise_conv_3x3_task(const DepthwiseConv3x3Args* args) {
    const int32_t core_number = args->core_number;
    const q7_t *input = args->input;
    const q7_t *kernel = args->kernel;
    const int32_t *bias = args->bias;
    q7_t *output = args->output;
    const int32_t input_x = args->input_x;
    const int32_t input_y = args->input_y;
    const int32_t input_ch = args->input_ch;
    const int32_t output_ch = args->output_ch;
    const int32_t pad_x = args->pad_x;
    const int32_t pad_y = args->pad_y;
    const int32_t stride_x = args->stride_x;
    const int32_t stride_y = args->stride_y;
    const int32_t *output_shift = args->output_shift;
    const int32_t *output_mult = args->output_mult;
    const int32_t output_x = args->output_x;
    const int32_t output_y = args->output_y;
    const int32_t output_offset = args->output_offset;
    const int32_t input_offset = args->input_offset;
    const int32_t output_activation_min = args->output_activation_min;
    const int32_t output_activation_max = args->output_activation_max;

    // Split between cores across y dimension.
    int32_t out_h_start = core_number == 0 ? 0 : output_y / 2;
    int32_t out_h_end = core_number == 0 ? output_y / 2 : output_y;
    int32_t out_idx_start = out_h_start * output_x * input_ch;

    for (int32_t in_h = out_h_start * stride_y - pad_y, out_h = out_h_start, out_idx = out_idx_start; out_h < out_h_end; in_h += stride_y, ++out_h)
    {
        for (int32_t in_w = -pad_x, out_w = 0, ker_h_start = MAX(0, -in_h); out_w < output_x; in_w += stride_x, ++out_w)
        {
            int32_t in_ch = 0;
            int32_t ker_w_start = MAX(0, -in_w);

            for (; in_ch <= (input_ch - 4); in_ch += 4)
            {
                int32_t out_buff0 = bias[in_ch + 0];
                int32_t out_buff1 = bias[in_ch + 1];
                int32_t out_buff2 = bias[in_ch + 2];
                int32_t out_buff3 = bias[in_ch + 3];

                const int8_t *input_ptr = input + (in_h + ker_h_start) * (input_ch * input_x) + in_w * input_ch + in_ch;
                const int8_t *kernel_ptr = kernel + ker_h_start * (input_ch * 3) + in_ch;

                for (int32_t ker_h = ker_h_start; ker_h < MIN(3, input_y - in_h); ++ker_h)
                {
                    int32_t in_val = 0;
                    int32_t ker_val = 0;

                    if (ker_w_start == 0)
                    {
                        in_val = arm_nn_read_q7x4(input_ptr);
                        ker_val = arm_nn_read_q7x4(kernel_ptr);

                        out_buff0 += ((int8_t)in_val + input_offset) * (int8_t)ker_val;
                        out_buff1 += ((int8_t)(in_val >> 8) + input_offset) * (int8_t)(ker_val >> 8);
                        out_buff2 += ((int8_t)(in_val >> 16) + input_offset) * (int8_t)(ker_val >> 16);
                        out_buff3 += ((int8_t)(in_val >> 24) + input_offset) * (int8_t)(ker_val >> 24);
                    }

                    in_val = arm_nn_read_q7x4(input_ptr + input_ch);
                    ker_val = arm_nn_read_q7x4(kernel_ptr + input_ch);

                    out_buff0 += ((int8_t)in_val + input_offset) * (int8_t)ker_val;
                    out_buff1 += ((int8_t)(in_val >> 8) + input_offset) * (int8_t)(ker_val >> 8);
                    out_buff2 += ((int8_t)(in_val >> 16) + input_offset) * (int8_t)(ker_val >> 16);
                    out_buff3 += ((int8_t)(in_val >> 24) + input_offset) * (int8_t)(ker_val >> 24);

                    if ((input_x - in_w) >= 3)
                    {
                        in_val = arm_nn_read_q7x4(input_ptr + (input_ch << 1));
                        ker_val = arm_nn_read_q7x4(kernel_ptr + (input_ch << 1));

                        out_buff0 += ((int8_t)in_val + input_offset) * (int8_t)ker_val;
                        out_buff1 += ((int8_t)(in_val >> 8) + input_offset) * (int8_t)(ker_val >> 8);
                        out_buff2 += ((int8_t)(in_val >> 16) + input_offset) * (int8_t)(ker_val >> 16);
                        out_buff3 += ((int8_t)(in_val >> 24) + input_offset) * (int8_t)(ker_val >> 24);
                    }

                    input_ptr += (input_ch * input_x);
                    kernel_ptr += (input_ch * 3);
                }

                out_buff0 = arm_nn_requantize(out_buff0, output_mult[in_ch + 0], output_shift[in_ch + 0]);
                out_buff1 = arm_nn_requantize(out_buff1, output_mult[in_ch + 1], output_shift[in_ch + 1]);
                out_buff2 = arm_nn_requantize(out_buff2, output_mult[in_ch + 2], output_shift[in_ch + 2]);
                out_buff3 = arm_nn_requantize(out_buff3, output_mult[in_ch + 3], output_shift[in_ch + 3]);

                out_buff0 += output_offset;
                out_buff1 += output_offset;
                out_buff2 += output_offset;
                out_buff3 += output_offset;

                out_buff0 = MIN(MAX(out_buff0, output_activation_min), output_activation_max);
                out_buff1 = MIN(MAX(out_buff1, output_activation_min), output_activation_max);
                out_buff2 = MIN(MAX(out_buff2, output_activation_min), output_activation_max);
                out_buff3 = MIN(MAX(out_buff3, output_activation_min), output_activation_max);

                output[out_idx++] = (int8_t)out_buff0;
                output[out_idx++] = (int8_t)out_buff1;
                output[out_idx++] = (int8_t)out_buff2;
                output[out_idx++] = (int8_t)out_buff3;
            }

            // Leftover
            for (; in_ch < input_ch; ++in_ch)
            {
                int32_t out_buff = bias[in_ch];

                const int8_t *input_ptr = input + (in_h + ker_h_start) * (input_ch * input_x) + in_w * input_ch + in_ch;
                const int8_t *kernel_ptr = kernel + ker_h_start * (input_ch * 3) + in_ch;

                for (int32_t ker_h = ker_h_start; ker_h < MIN(3, input_y - in_h); ++ker_h)
                {
                    if (ker_w_start == 0)
                    {
                        out_buff += (*(input_ptr) + input_offset) * *(kernel_ptr);
                    }

                    out_buff += (*(input_ptr + input_ch) + input_offset) * *(kernel_ptr + input_ch);

                    if ((input_x - in_w) >= 3)
                    {
                        out_buff += (*(input_ptr + (input_ch << 1)) + input_offset) * *(kernel_ptr + (input_ch << 1));
                    }

                    input_ptr += (input_ch * input_x);
                    kernel_ptr += (input_ch * 3);
                }

                out_buff = arm_nn_requantize(out_buff, output_mult[in_ch], output_shift[in_ch]);
                out_buff += output_offset;
                out_buff = MIN(MAX(out_buff, output_activation_min), output_activation_max);
                output[out_idx++] = (int8_t)out_buff;
            }
        }
    }
}

static void core1_depthwise_conv_3x3_worker(void) {
    depthwise_conv_3x3_task(&g_core1_depthwise_conv_3x3_args);

    // Signal we're done by pushing a result of zero.
    multicore_fifo_push_blocking(ARM_MATH_SUCCESS);
}

arm_status arm_depthwise_conv_3x3_s8(const cmsis_nn_context *ctx,
                                     const cmsis_nn_dw_conv_params *dw_conv_params,
                                     const cmsis_nn_per_channel_quant_params *quant_params,
                                     const cmsis_nn_dims *input_dims,
                                     const q7_t *input,
                                     const cmsis_nn_dims *filter_dims,
                                     const q7_t *kernel,
                                     const cmsis_nn_dims *bias_dims,
                                     const int32_t *bias,
                                     const cmsis_nn_dims *output_dims,
                                     q7_t *output)
{
    (void)ctx;
    (void)bias_dims;

    const int32_t input_ch = input_dims->c;
    const int32_t output_ch = output_dims->c;
    const int32_t pad_x = dw_conv_params->padding.w;
    const int32_t pad_y = dw_conv_params->padding.h;
    /* Check input constraints input_ch == output_ch */
    if (input_ch != output_ch)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }
    /* Check input constraints pad_x <= 1 */
    if (pad_x > 1 || filter_dims->w != 3 || filter_dims->h != 3)
    {
        return ARM_MATH_ARGUMENT_ERROR;
    }


    DepthwiseConv3x3Args core1_depthwise_conv_3x3_args = {
      .core_number = 1,
      .input = input,
      .kernel = kernel,
      .bias = bias,
      .output = output,
      .input_x = input_dims->w,
      .input_y = input_dims->h,
      .input_ch = input_ch,
      .output_ch = output_ch,
      .pad_x = pad_x,
      .pad_y = pad_y,
      .stride_x = dw_conv_params->stride.w,
      .stride_y = dw_conv_params->stride.h,
      .output_shift = quant_params->shift,
      .output_mult = quant_params->multiplier,
      .output_x = output_dims->w,
      .output_y = output_dims->h,
      .output_offset = dw_conv_params->output_offset,
      .input_offset = dw_conv_params->input_offset,
      .output_activation_min = dw_conv_params->activation.min,
      .output_activation_max = dw_conv_params->activation.max
    };

    g_core1_depthwise_conv_3x3_args = core1_depthwise_conv_3x3_args;
    multicore_reset_core1();
    multicore_launch_core1(core1_depthwise_conv_3x3_worker);

    DepthwiseConv3x3Args core0_depthwise_conv_3x3_args = core1_depthwise_conv_3x3_args;
    core0_depthwise_conv_3x3_args.core_number = 0;

    depthwise_conv_3x3_task(&core0_depthwise_conv_3x3_args);
    
    return multicore_fifo_pop_blocking();
}
    /**
 * @} end of NNConv group
 */
