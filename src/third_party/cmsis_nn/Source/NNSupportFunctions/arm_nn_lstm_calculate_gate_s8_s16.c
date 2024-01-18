/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office.com>
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
 * Title:        arm_nn_lstm_calculate_gate_s8_s16.c
 * Description:  Update single gate for an incremental step of LSTM function.
 *
 * $Date:        8 September 2022
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nn_tables.h"
#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @defgroup supportLSTM LSTM
 *
 * Support functions for LSTM
 *
 */

/**
 * @addtogroup supportLSTM
 * @{
 */

/*
 * Calculates a single LSTM gate, int8x8_16 version.
 * Refer to header file for details
 */
void arm_nn_lstm_calculate_gate_s8_s16(const int8_t *input,
                                       const int8_t *input_to_gate_weights,
                                       const int32_t *input_to_gate_bias,
                                       const cmsis_nn_scaling input_to_gate_scaling,
                                       const int8_t *output_state,
                                       const int8_t *recurrent_to_gate_weights,
                                       const int32_t *recurrent_to_gate_bias,
                                       const cmsis_nn_scaling recurrent_to_gate,
                                       const int32_t n_batch,
                                       const int32_t n_input,
                                       const int32_t n_output,
                                       const int32_t n_cell,
                                       const arm_nn_activation_type activation_type,
                                       int16_t *gate)
{
    const int32_t n_block = n_batch * n_cell;

    memset(gate, 0, n_block * sizeof(int16_t));
    arm_nn_vec_mat_mul_result_acc_s8(input,
                                     input_to_gate_weights,
                                     input_to_gate_bias,
                                     gate,
                                     0,
                                     input_to_gate_scaling.multiplier,
                                     input_to_gate_scaling.shift,
                                     n_input,
                                     n_cell,
                                     n_batch);

    arm_nn_vec_mat_mul_result_acc_s8(output_state,
                                     recurrent_to_gate_weights,
                                     recurrent_to_gate_bias,
                                     gate,
                                     0,
                                     recurrent_to_gate.multiplier,
                                     recurrent_to_gate.shift,
                                     n_output,
                                     n_cell,
                                     n_batch);

    arm_nn_activation_s16(gate, gate, n_block, 0, activation_type);
}
/**
 * @} end of supportLSTM group
 */
