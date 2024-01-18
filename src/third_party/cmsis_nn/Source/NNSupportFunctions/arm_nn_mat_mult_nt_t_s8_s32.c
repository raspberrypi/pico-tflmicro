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
 * Title:        arm_nn_mat_mult_s8_nt_t_s8_s32
 * Description:  Matrix multiplication support function with the right-hand-side (rhs) matrix transposed
 *
 * $Date:        5 October 2023
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

/*
 * s32 matrix multiplication with the right-hand-side matrix transposed
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_s8_s32(const int8_t *lhs,
                                                const int8_t *rhs,
                                                int32_t *dst,
                                                const int32_t lhs_rows,
                                                const int32_t rhs_rows,
                                                const int32_t rhs_cols,
                                                const int32_t lhs_offset,
                                                const int32_t dst_idx_offset)
{
    const int32_t dst_idx_col_offset = dst_idx_offset * rhs_cols;

    for (int32_t rhs_rows_idx = 0; rhs_rows_idx <= (rhs_rows - 2); rhs_rows_idx += 2)
    {
        int32_t *dst_ptr = &dst[0];
        const int8_t *lhs_ptr = &lhs[0];

        int32_t lhs_rows_idx = lhs_rows >> 1;

        while (lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];

            const int32_t lhs_value00 = lhs_ptr[0] + lhs_offset;
            const int32_t lhs_value01 = lhs_ptr[1] + lhs_offset;

            const int32_t lhs_value10 = lhs_ptr[rhs_rows] + lhs_offset;
            const int32_t lhs_value11 = lhs_ptr[rhs_rows + 1] + lhs_offset;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                const int32_t rhs_value0 = rhs_ptr[0];
                const int32_t rhs_value1 = rhs_ptr[1];

                const int32_t res00 = lhs_value00 * rhs_value0;
                const int32_t res10 = lhs_value01 * rhs_value1;

                const int32_t res01 = lhs_value10 * rhs_value0;
                const int32_t res11 = lhs_value11 * rhs_value1;

                dst_ptr[0] += res00;
                dst_ptr[0] += res10;
                dst_ptr[dst_idx_col_offset] += res01;
                dst_ptr[dst_idx_col_offset] += res11;
                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
            dst_ptr += dst_idx_col_offset;

            lhs_ptr += rhs_rows << 1;

            lhs_rows_idx--;
        }

        // Left-over rows
        if (lhs_rows % 2)
        {
            const int8_t *rhs_ptr = &rhs[0];
            const int32_t lhs_value = lhs_ptr[0] + lhs_offset;
            const int32_t lhs_value01 = lhs_ptr[1] + lhs_offset;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                const int32_t rhs_value0 = rhs_ptr[0];
                const int32_t rhs_value01 = rhs_ptr[1];
                const int32_t res00 = lhs_value * rhs_value0;
                const int32_t res01 = lhs_value01 * rhs_value01;

                dst_ptr[0] += res00;
                dst_ptr[0] += res01;

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
        }

        rhs += 2;
        lhs += 2;
    }

    if (rhs_rows % 2)
    {
        const int8_t *lhs_ptr = &lhs[0];
        int32_t *dst_ptr = &dst[0];

        for (int32_t lhs_rows_idx = 0; lhs_rows_idx < lhs_rows; ++lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];
            int32_t lhs_value = lhs_ptr[0] + lhs_offset;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int32_t rhs_value = rhs_ptr[0];

                int32_t res00 = lhs_value * rhs_value;

                *dst_ptr += res00;

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
            lhs_ptr += rhs_rows;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
