/*
 * SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office.com>
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
 * Title:        arm_nn_lstm_step_s8_s16.c
 * Description:  Update LSTM function for a single iteration step.
 *
 * $Date:        9 Februari 2023
 * $Revision:    V.1.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"
/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportLSTM
 * @{
 */

/*
 * Calculate the output state tensor of an LSTM step, s8 input/output and s16 weight version.
 * Refer to header file for details.
 */
arm_cmsis_nn_status arm_nn_lstm_step_s8_s16(const int8_t *input,
                                            const int8_t *input_to_input_weight,
                                            const int8_t *input_to_forget_weight,
                                            const int8_t *input_to_cell_weight,
                                            const int8_t *input_to_output_weight,
                                            const int8_t *recurrent_to_input_weight,
                                            const int8_t *recurrent_to_forget_weight,
                                            const int8_t *recurrent_to_cell_weight,
                                            const int8_t *recurrent_to_output_weight,
                                            const cmsis_nn_lstm_params *lstm,
                                            const int n_batch,
                                            const int n_cell,
                                            const int n_input,
                                            const int n_output,
                                            int8_t *output_state,
                                            int16_t *cell_state,
                                            int8_t *output,
                                            cmsis_nn_lstm_context *scratch_buffers)
{
    const int32_t n_block = n_batch * n_cell;

    // Calculate the input gate
    arm_nn_lstm_calculate_gate_s8_s16(input,
                                      input_to_input_weight,
                                      lstm->i2i_effective_bias,
                                      lstm->input_to_input_scaling,
                                      output_state,
                                      recurrent_to_input_weight,
                                      lstm->r2i_effective_bias,
                                      lstm->recurrent_to_input_scaling,
                                      n_batch,
                                      n_input,
                                      n_output,
                                      n_cell,
                                      ARM_SIGMOID,
                                      scratch_buffers->input_gate);

    // Calculate the forget gate
    arm_nn_lstm_calculate_gate_s8_s16(input,
                                      input_to_forget_weight,
                                      lstm->i2f_effective_bias,
                                      lstm->input_to_forget_scaling,
                                      output_state,
                                      recurrent_to_forget_weight,
                                      lstm->r2f_effective_bias,
                                      lstm->recurrent_to_forget_scaling,
                                      n_batch,
                                      n_input,
                                      n_output,
                                      n_cell,
                                      ARM_SIGMOID,
                                      scratch_buffers->forget_gate);

    // Calculate the cell update gate
    arm_nn_lstm_calculate_gate_s8_s16(input,
                                      input_to_cell_weight,
                                      lstm->i2c_effective_bias,
                                      lstm->input_to_cell_scaling,
                                      output_state,
                                      recurrent_to_cell_weight,
                                      lstm->r2c_effective_bias,
                                      lstm->recurrent_to_cell_scaling,
                                      n_batch,
                                      n_input,
                                      n_output,
                                      n_cell,
                                      ARM_TANH,
                                      scratch_buffers->cell_gate);

    // Update the cell state
    arm_nn_lstm_update_cell_state_s16(n_block,
                                      lstm->cell_state_shift,
                                      cell_state,
                                      scratch_buffers->input_gate,
                                      scratch_buffers->forget_gate,
                                      scratch_buffers->cell_gate);

    // Calculate the output gate
    arm_nn_lstm_calculate_gate_s8_s16(input,
                                      input_to_output_weight,
                                      lstm->i2o_effective_bias,
                                      lstm->input_to_output_scaling,
                                      output_state,
                                      recurrent_to_output_weight,
                                      lstm->r2o_effective_bias,
                                      lstm->recurrent_to_output_scaling,
                                      n_batch,
                                      n_input,
                                      n_output,
                                      n_cell,
                                      ARM_SIGMOID,
                                      scratch_buffers->output_gate);

    // Update the output state
    arm_nn_lstm_update_output_s8_s16(n_batch,
                                     n_cell,
                                     cell_state,
                                     lstm->cell_state_shift,
                                     scratch_buffers->output_gate,
                                     lstm->hidden_scaling,
                                     lstm->hidden_offset,
                                     output_state,
                                     scratch_buffers->input_gate);

    arm_memcpy_s8(output, output_state, n_batch * n_output * sizeof(int8_t));

    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of supportLSTM group
 */
