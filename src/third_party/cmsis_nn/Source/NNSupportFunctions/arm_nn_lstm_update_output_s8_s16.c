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
 * Title:        arm_nn_lstm_update_output_s8_s16.c
 * Description:  Update output gate for an incremental step of LSTM function.
 *
 * $Date:        13 Februari 2023
 * $Revision:    V.2.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
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
 * Refer to header files for details
 */
void arm_nn_lstm_update_output_s8_s16(const int n_batch,
                                      const int n_cell,
                                      int16_t *cell_state,
                                      const int32_t cell_state_scale,
                                      const int16_t *output_gate,
                                      const cmsis_nn_scaling hidden_scaling,
                                      const int32_t hidden_offset,
                                      int8_t *output_state,
                                      int16_t *cell_gate_scratch)
{
    const int32_t size = n_batch * n_cell;

    int32_t tanh_input_left_shift = (15 + cell_state_scale) - 3;
    if (tanh_input_left_shift < 0)
    {
        tanh_input_left_shift = -tanh_input_left_shift;
        for (int32_t i = 0; i < size; i++)
        {
            cell_state[i] = cell_state[i] >> tanh_input_left_shift;
        }
        tanh_input_left_shift = 0;
    }
    arm_nn_activation_s16(cell_state, cell_gate_scratch, size, tanh_input_left_shift, ARM_TANH);

    arm_elementwise_mul_s16_s8(output_gate,
                               cell_gate_scratch,
                               output_state,
                               hidden_offset,
                               hidden_scaling.multiplier,
                               hidden_scaling.shift,
                               size);
}
/**
 * @} end of supportLSTM group
 */
