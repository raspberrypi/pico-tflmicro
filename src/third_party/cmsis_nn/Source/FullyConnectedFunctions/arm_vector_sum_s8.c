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
 * Title:        arm_vector_sum_s8
 * Description:  Generic function for calculating vector sums
 *
 * $Date:        5 September 2023
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup FC
 * @{
 */

/*
 * S8 vector sum fuction in preparation for e.g. kernel sums in fully connected and matrix multiplication layer function
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_vector_sum_s8(int32_t *vector_sum_buf,
                                      const int32_t vector_cols,
                                      const int32_t vector_rows,
                                      const int8_t *vector_data)
{
#if defined(ARM_MATH_MVEI)
    const int32_t row_loop_cnt = vector_rows / 4;

    for (int i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; i_row_loop_cnt++)
    {
        const int32_t col_loop_cnt = (vector_cols + 15) / 16;

        const int8_t *vector_0 = vector_data;
        const int8_t *vector_1 = vector_data + vector_cols;
        const int8_t *vector_2 = vector_data + 2 * vector_cols;
        const int8_t *vector_3 = vector_data + 3 * vector_cols;

        int32_t vector_sum_0 = 0;
        int32_t vector_sum_1 = 0;
        int32_t vector_sum_2 = 0;
        int32_t vector_sum_3 = 0;

        uint32_t col_cnt = (uint32_t)vector_cols;

        for (int i = 0; i < col_loop_cnt; i++)
        {
            mve_pred16_t p = vctp8q(col_cnt);
            col_cnt -= 16;

            const int8x16_t ker_0 = vldrbq_z_s8(vector_0, p);
            vector_sum_0 = vaddvaq_s8(vector_sum_0, ker_0);

            const int8x16_t ker_1 = vldrbq_z_s8(vector_1, p);
            vector_sum_1 = vaddvaq_s8(vector_sum_1, ker_1);

            const int8x16_t ker_2 = vldrbq_z_s8(vector_2, p);
            vector_sum_2 = vaddvaq_s8(vector_sum_2, ker_2);

            const int8x16_t ker_3 = vldrbq_z_s8(vector_3, p);
            vector_sum_3 = vaddvaq_s8(vector_sum_3, ker_3);

            vector_0 += 16;
            vector_1 += 16;
            vector_2 += 16;
            vector_3 += 16;
        }
        vector_data += 4 * vector_cols;

        vector_sum_buf[0] = vector_sum_0;
        vector_sum_buf[1] = vector_sum_1;
        vector_sum_buf[2] = vector_sum_2;
        vector_sum_buf[3] = vector_sum_3;
        vector_sum_buf += 4;
    }

    const int32_t loop_cnt = vector_rows % 4;

    for (int i_row_loop_cnt = 0; i_row_loop_cnt < loop_cnt; i_row_loop_cnt++)
    {
        const int32_t col_loop_cnt = (vector_cols + 15) / 16;

        const int8_t *vector_0 = vector_data;

        int32_t vector_sum_0 = 0;

        uint32_t col_cnt = (uint32_t)vector_cols;

        for (int i = 0; i < col_loop_cnt; i++)
        {
            mve_pred16_t p = vctp8q(col_cnt);
            col_cnt -= 16;

            const int8x16_t ker_0 = vldrbq_z_s8(vector_0, p);
            vector_sum_0 = vaddvaq_s8(vector_sum_0, ker_0);

            vector_0 += 16;
        }
        vector_data += vector_cols;

        vector_sum_buf[i_row_loop_cnt] = vector_sum_0;
    }

    return (ARM_CMSIS_NN_SUCCESS);
#else
    (void)vector_sum_buf;
    (void)vector_rows;
    (void)vector_cols;
    (void)vector_data;

    return (ARM_CMSIS_NN_NO_IMPL_ERROR);
#endif
}

/**
 * @} end of FC group
 */
