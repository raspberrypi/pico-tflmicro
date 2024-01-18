/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_lstm_unidirectional_s16_s8.c
 * Description:  S8 LSTM function with S16 gate output
 *
 * $Date:        4 November 2022
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M processors
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup LSTM
 * @{
 */

/*
 * S8 LSTM function for TensorFlow Lite with S16 gate output
 *
 * Refer to header file for details.
 *
 */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/*
 * LSTM unidirectional function with 8 bit input and output and 16 bit weights
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_lstm_unidirectional_s16_s8(cmsis_nn_lstm_context *scratch_buffers,
                                                   const int8_t *input_data,
                                                   const cmsis_nn_lstm_dims *lstm_dims,
                                                   const int8_t *in_to_in_weights,
                                                   const int8_t *in_to_forget_weights,
                                                   const int8_t *in_to_cell_weights,
                                                   const int8_t *in_to_out_weights,
                                                   const int8_t *recurrent_to_in_weights,
                                                   const int8_t *recurrent_to_forget_weights,
                                                   const int8_t *recurrent_to_cell_weights,
                                                   const int8_t *recurrent_to_out_weights,
                                                   const int16_t *cell_to_in_weights,
                                                   const int16_t *cell_to_forget_weights,
                                                   const int16_t *cell_to_out_weights,
                                                   const int8_t *projection_weights,
                                                   const cmsis_nn_lstm_params *lstm,
                                                   int8_t *output_state,
                                                   int16_t *cell_state,
                                                   int8_t *output_data)
{
    (void)cell_to_in_weights;
    (void)cell_to_forget_weights;
    (void)cell_to_out_weights;

    const int32_t num_batch = lstm_dims->num_batches;
    const int32_t num_input = lstm_dims->num_inputs;
    const int32_t max_time = lstm_dims->max_time;

    const int32_t num_output = lstm_dims->num_outputs;
    const int32_t out_batch_leading_dim = num_output;

    // num_cell = num_output is considered in the code under the assumption that projection is NULL.
    const int32_t num_cell = num_output;

    if (projection_weights != NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (lstm->i2f_effective_bias == NULL || lstm->i2c_effective_bias == NULL || lstm->i2o_effective_bias == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (lstm->r2f_effective_bias == NULL || lstm->r2c_effective_bias == NULL || lstm->r2o_effective_bias == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (lstm->i2i_effective_bias == NULL || lstm->r2i_effective_bias == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (lstm->time_major)
    {
        const int32_t in_step = num_batch * num_input;
        const int32_t out_step = num_batch * out_batch_leading_dim;
        for (int i_max_time = 0; i_max_time < max_time; i_max_time++)
        {
            arm_cmsis_nn_status status = arm_nn_lstm_step_s8_s16(input_data + i_max_time * in_step,
                                                                 in_to_in_weights,
                                                                 in_to_forget_weights,
                                                                 in_to_cell_weights,
                                                                 in_to_out_weights,
                                                                 recurrent_to_in_weights,
                                                                 recurrent_to_forget_weights,
                                                                 recurrent_to_cell_weights,
                                                                 recurrent_to_out_weights,
                                                                 lstm,
                                                                 num_batch,
                                                                 num_cell,
                                                                 num_input,
                                                                 num_output,
                                                                 output_state,
                                                                 cell_state,
                                                                 output_data + i_max_time * out_step,
                                                                 scratch_buffers);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                return status;
            }
        }
    }
    else
    {
        for (int i_num_batch = 0; i_num_batch < num_batch; i_num_batch++)
        {
            const int32_t in_step = num_input;
            const int32_t out_step = out_batch_leading_dim;
            for (int i_max_time = 0; i_max_time < max_time; i_max_time++)
            {
                const int32_t time_offset = i_num_batch * max_time + i_max_time;

                arm_cmsis_nn_status status = arm_nn_lstm_step_s8_s16(input_data + time_offset * in_step,
                                                                     in_to_in_weights,
                                                                     in_to_forget_weights,
                                                                     in_to_cell_weights,
                                                                     in_to_out_weights,
                                                                     recurrent_to_in_weights,
                                                                     recurrent_to_forget_weights,
                                                                     recurrent_to_cell_weights,
                                                                     recurrent_to_out_weights,
                                                                     lstm,
                                                                     /*num_batch=*/1,
                                                                     num_cell,
                                                                     num_input,
                                                                     num_output,
                                                                     output_state + i_num_batch * out_batch_leading_dim,
                                                                     cell_state + i_num_batch * num_cell,
                                                                     output_data + time_offset * out_step,
                                                                     scratch_buffers);
                if (status != ARM_CMSIS_NN_SUCCESS)
                {
                    return status;
                }
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of LSTM group
 */
