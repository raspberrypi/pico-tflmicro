/*
 * SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_types.h
 * Description:  Public header file to contain the CMSIS-NN structs for the
 *               TensorFlowLite micro compliant functions
 *
 * $Date:        27 October 2023
 * $Revision:    V.2.6.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef _ARM_NN_TYPES_H
#define _ARM_NN_TYPES_H

#include <stdint.h>

/** Enum for specifying activation function types */
typedef enum
{
    ARM_SIGMOID = 0, /**< Sigmoid activation function */
    ARM_TANH = 1,    /**< Tanh activation function */
} arm_nn_activation_type;

/** Function return codes */
typedef enum
{
    ARM_CMSIS_NN_SUCCESS = 0,        /**< No error */
    ARM_CMSIS_NN_ARG_ERROR = -1,     /**< One or more arguments are incorrect */
    ARM_CMSIS_NN_NO_IMPL_ERROR = -2, /**<  No implementation available */
    ARM_CMSIS_NN_FAILURE = -3,       /**<  Logical error */
} arm_cmsis_nn_status;

/** CMSIS-NN object to contain the width and height of a tile */
typedef struct
{
    int32_t w; /**< Width */
    int32_t h; /**< Height */
} cmsis_nn_tile;

/** CMSIS-NN object used for the function context. */
typedef struct
{
    void *buf;    /**< Pointer to a buffer needed for the optimization */
    int32_t size; /**< Buffer size */
} cmsis_nn_context;

/** CMSIS-NN object to contain the dimensions of the tensors */
typedef struct
{
    int32_t n; /**< Generic dimension to contain either the batch size or output channels.
                     Please refer to the function documentation for more information */
    int32_t h; /**< Height */
    int32_t w; /**< Width */
    int32_t c; /**< Input channels */
} cmsis_nn_dims;

/** CMSIS-NN object to contain LSTM specific input parameters related to dimensions */
typedef struct
{
    int32_t max_time;
    int32_t num_inputs;
    int32_t num_batches;
    int32_t num_outputs;
} cmsis_nn_lstm_dims;

/** CMSIS-NN object for the per-channel quantization parameters */
typedef struct
{
    int32_t *multiplier; /**< Multiplier values */
    int32_t *shift;      /**< Shift values */
} cmsis_nn_per_channel_quant_params;

/** CMSIS-NN object for the per-tensor quantization parameters */
typedef struct
{
    int32_t multiplier; /**< Multiplier value */
    int32_t shift;      /**< Shift value */
} cmsis_nn_per_tensor_quant_params;

/** CMSIS-NN object for the quantized Relu activation */
typedef struct
{
    int32_t min; /**< Min value used to clamp the result */
    int32_t max; /**< Max value used to clamp the result */
} cmsis_nn_activation;

/** CMSIS-NN object for the convolution layer parameters */
typedef struct
{
    int32_t input_offset;  /**< The negative of the zero value for the input tensor */
    int32_t output_offset; /**< The negative of the zero value for the output tensor */
    cmsis_nn_tile stride;
    cmsis_nn_tile padding;
    cmsis_nn_tile dilation;
    cmsis_nn_activation activation;
} cmsis_nn_conv_params;

/** CMSIS-NN object for the transpose convolution layer parameters */
typedef struct
{
    int32_t input_offset;  /**< The negative of the zero value for the input tensor */
    int32_t output_offset; /**< The negative of the zero value for the output tensor */
    cmsis_nn_tile stride;
    cmsis_nn_tile padding;
    cmsis_nn_tile padding_offsets;
    cmsis_nn_tile dilation;
    cmsis_nn_activation activation;
} cmsis_nn_transpose_conv_params;

/** CMSIS-NN object for the depthwise convolution layer parameters */
typedef struct
{
    int32_t input_offset;  /**< The negative of the zero value for the input tensor */
    int32_t output_offset; /**< The negative of the zero value for the output tensor */
    int32_t ch_mult;       /**< Channel Multiplier. ch_mult * in_ch = out_ch */
    cmsis_nn_tile stride;
    cmsis_nn_tile padding;
    cmsis_nn_tile dilation;
    cmsis_nn_activation activation;
} cmsis_nn_dw_conv_params;
/** CMSIS-NN object for pooling layer parameters */
typedef struct
{
    cmsis_nn_tile stride;
    cmsis_nn_tile padding;
    cmsis_nn_activation activation;
} cmsis_nn_pool_params;

/** CMSIS-NN object for Fully Connected layer parameters */
typedef struct
{
    int32_t input_offset;  /**< The negative of the zero value for the input tensor */
    int32_t filter_offset; /**< The negative of the zero value for the filter tensor. Not used */
    int32_t output_offset; /**< The negative of the zero value for the output tensor */
    cmsis_nn_activation activation;
} cmsis_nn_fc_params;

/** CMSIS-NN object for SVDF layer parameters */
typedef struct
{
    int32_t rank;
    int32_t input_offset;  /**< The negative of the zero value for the input tensor */
    int32_t output_offset; /**< The negative of the zero value for the output tensor */
    cmsis_nn_activation input_activation;
    cmsis_nn_activation output_activation;
} cmsis_nn_svdf_params;

/** CMSIS-NN object for Softmax s16 layer parameters */
typedef struct
{
    const int16_t *exp_lut;
    const int16_t *one_by_one_lut;
} cmsis_nn_softmax_lut_s16;

/** LSTM guard parameters */
typedef struct
{
    int32_t input_variance;
    int32_t forget_variance;
    int32_t cell_variance;
    int32_t output_variance;
} cmsis_nn_lstm_guard_params;

/** LSTM scratch buffer container */
typedef struct
{
    int16_t *input_gate;
    int16_t *forget_gate;
    int16_t *cell_gate;
    int16_t *output_gate;
} cmsis_nn_lstm_context;

/** Quantized clip value for cell and projection of LSTM input. Zero value means no clipping. */
typedef struct
{
    int16_t cell;
    int8_t projection;
} cmsis_nn_lstm_clip_params;

/** CMSIS-NN object for quantization parameters */
typedef struct
{
    int32_t multiplier; /**< Multiplier value */
    int32_t shift;      /**< Shift value */
} cmsis_nn_scaling;

/** CMSIS-NN norm layer coefficients */
typedef struct
{
    int16_t *input_weight;
    int16_t *forget_weight;
    int16_t *cell_weight;
    int16_t *output_weight;
} cmsis_nn_layer_norm;

/** Parameters for integer LSTM, as defined in TFLM */
typedef struct
{
    int32_t time_major; /**< Nonzero (true) if first row of data is timestamps for input */
    cmsis_nn_scaling input_to_input_scaling;
    cmsis_nn_scaling input_to_forget_scaling;
    cmsis_nn_scaling input_to_cell_scaling;
    cmsis_nn_scaling input_to_output_scaling;
    cmsis_nn_scaling recurrent_to_input_scaling;
    cmsis_nn_scaling recurrent_to_forget_scaling;
    cmsis_nn_scaling recurrent_to_cell_scaling;
    cmsis_nn_scaling recurrent_to_output_scaling;
    cmsis_nn_scaling cell_to_input_scaling;
    cmsis_nn_scaling cell_to_forget_scaling;
    cmsis_nn_scaling cell_to_output_scaling;
    cmsis_nn_scaling projection_scaling;
    cmsis_nn_scaling hidden_scaling;
    cmsis_nn_scaling layer_norm_input_scaling;  /**< layer normalization for input layer */
    cmsis_nn_scaling layer_norm_forget_scaling; /**< layer normalization for forget gate */
    cmsis_nn_scaling layer_norm_cell_scaling;   /**< layer normalization for cell */
    cmsis_nn_scaling layer_norm_output_scaling; /**< layer normalization for outpus layer */

    int32_t cell_state_shift;
    int32_t hidden_offset;
    int32_t output_state_offset;

    cmsis_nn_lstm_clip_params clip;
    cmsis_nn_lstm_guard_params guard;
    cmsis_nn_layer_norm layer_norm;

    /* Effective bias is precalculated as bias + zero_point * weight.
    Only applicable to when input/output are s8 and weights are s16 */
    const int32_t *i2i_effective_bias; /**< input to input effective bias */
    const int32_t *i2f_effective_bias; /**< input to forget gate effective bias */
    const int32_t *i2c_effective_bias; /**< input to cell effective bias */
    const int32_t *i2o_effective_bias; /**< input to output effective bias */

    const int32_t *r2i_effective_bias; /**< recurrent gate to input effective bias */
    const int32_t *r2f_effective_bias; /**< recurrent gate to forget gate effective bias */
    const int32_t *r2c_effective_bias; /**< recurrent gate to cell effective bias */
    const int32_t *r2o_effective_bias; /**< recurrent gate to output effective bias */

    const int32_t *projection_effective_bias;

    /* Not precalculated bias */
    const int32_t *input_gate_bias;
    const int32_t *forget_gate_bias;
    const int32_t *cell_gate_bias;
    const int32_t *output_gate_bias;

    /* Activation min and max */
    cmsis_nn_activation activation;

} cmsis_nn_lstm_params;

#endif // _ARM_NN_TYPES_H
