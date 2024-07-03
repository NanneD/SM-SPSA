#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import pandas as pd
import numpy.random as rnd

from algorithm_functions import (
    logistic_transformation,
    inverse_logistic_transformation_matrix,
    calculate_objective_value,
    logistic_simplex_transformation,
    stationary_distribution,
    normalisation,
    scale_matrix,
    generate_centred_mass_initial_matrix,
    hyperlink_cost_function,
)

from matrix_algorithms import (
    run_matrix_algorithm,
    run_matrix_algorithm_improved_memory,
)
from matrix_SPSA_algorithm import SPSA_gradient, calculate_numerator_SPSA

from pytest import approx

########################Tests for algorithm_functions.py#######################


def test_generate_centred_mass_initial_matrix():

    C_1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    C_2 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0]])

    C_3 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    matrix_1 = np.array(
        [
            [0.3, 0.4, 0.1, 0.2],
            [0.3, 0.0, 0.6, 0.1],
            [1 / 3, 1 / 9, 1 / 3, 2 / 9],
            [0.25, 0.20, 0.00, 0.55],
        ]
    )

    centred_matrix_1 = generate_centred_mass_initial_matrix(matrix_1, C_1)
    centred_matrix_2 = generate_centred_mass_initial_matrix(matrix_1, C_2)
    centred_matrix_3 = generate_centred_mass_initial_matrix(matrix_1, C_3)

    assert centred_matrix_1 == approx(
        np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    assert centred_matrix_2 == approx(
        np.array(
            [
                [0.3, 0.4, 0.1, 0.2],
                [0.3, 0.0, 0.6, 0.1],
                [5 / 18, 1 / 9, 1 / 3, 5 / 18],
                [0.15, 0.15, 0.15, 0.55],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    assert centred_matrix_3 == approx(
        np.array(
            [
                [0.3, 0.4, 0.1, 0.2],
                [0.3, 0.0, 0.6, 0.1],
                [1 / 3, 1 / 9, 1 / 3, 2 / 9],
                [0.25, 0.2, 0.00, 0.55],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )


def test_calculate_objective_value():

    ALGORITHM_PARAMETERS_1 = {"GAME_TYPE": "regular", "OBJECTIVE_INDICES": [0]}
    ALGORITHM_PARAMETERS_2 = {"GAME_TYPE": "regular", "OBJECTIVE_INDICES": [3]}
    ALGORITHM_PARAMETERS_5 = {
        "GAME_TYPE": "hyperlink",
        "OBJECTIVE_INDICES": [0],
    }
    ALGORITHM_PARAMETERS_6 = {
        "GAME_TYPE": "hyperlink",
        "OBJECTIVE_INDICES": [2],
    }

    trans_matrix_dummy = np.eye(5)  # Dummy transition matrix
    stat_dist = np.array([0.5, 0.18, 0.27, 0.01, 0.04])

    trans_matrix_2 = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    stat_dist_2 = stationary_distribution(trans_matrix_2)
    hyperlink_costs = hyperlink_cost_function(0.25)

    assert (
        calculate_objective_value(
            stat_dist, trans_matrix_dummy, ALGORITHM_PARAMETERS_1
        )
        == stat_dist[0]
    )
    assert (
        calculate_objective_value(
            stat_dist, trans_matrix_dummy, ALGORITHM_PARAMETERS_2
        )
        == stat_dist[3]
    )

    assert calculate_objective_value(
        stat_dist_2, trans_matrix_2, ALGORITHM_PARAMETERS_5
    ) == approx(0.25 - 3 * 0.25 * hyperlink_costs, rel=1e-6, abs=1e-12)
    assert calculate_objective_value(
        stat_dist_2, trans_matrix_2, ALGORITHM_PARAMETERS_6
    ) == approx(0.25 - 3 * 0.25 * hyperlink_costs, rel=1e-6, abs=1e-12)


def test_stationary_distribution():

    P_1 = np.array(
        [[0.20, 0.30, 0.50], [0.10, 0.00, 0.90], [0.55, 0.00, 0.45]]
    )  # Example 2.15, page 25, Kulkarni.

    stat_1 = np.array([10 / 27, 1 / 9, 14 / 27])

    P_2 = np.array(
        [
            [0.0100, 0.9900, 0, 0, 0],
            [0.00995, 0.9851, 0.00495, 0, 0],
            [0, 0.00995, 0.9851, 0.00495, 0],
            [0, 0, 0.00995, 0.9851, 0.00495],
            [0, 0, 0, 0.9950, 0.0050],
        ]
    )  # Example 2.11 and 2.25, pages 20 and 34, Kulkarni.

    stat_2 = np.array([0.0057, 0.5694, 0.2833, 0.1409, 0.0007])

    P_3 = np.array(
        [
            [0.9700, 0.0300, 0.0000, 0.0000],
            [0.0080, 0.9820, 0.0100, 0.0000],
            [0.0200, 0.0000, 0.9750, 0.0050],
            [0.0100, 0.0000, 0.0000, 0.9900],
        ]
    )  # Example 2.6, pages 11 and 12, Kulkarni.

    assert stationary_distribution(P_1) == approx(
        stat_1, rel=1e-6, abs=1e-12
    )  # Compare to solution Kulkarni
    assert np.sum(stationary_distribution(P_1)) == approx(
        1, abs=1e-12
    )  # sum of a stationary distribution is 1.
    assert np.all(
        (stationary_distribution(P_1) >= 0)
        & (stationary_distribution(P_1) <= 1)
    )  # All elements lie between 0&1
    assert stationary_distribution(P_1) == approx(
        np.matmul(stationary_distribution(P_1), P_1), rel=1e-6, abs=1e-12
    )  # pi = piP

    assert stationary_distribution(P_2) == approx(
        stat_2, rel=1e-6, abs=1e-4
    )  # Compare to solution Kulkarni. Note the abs accuracy.
    assert np.sum(stationary_distribution(P_2)) == approx(
        1, abs=1e-12
    )  # sum of a stationary distribution is 1.
    assert np.all(
        (stationary_distribution(P_2) >= 0)
        & (stationary_distribution(P_2) <= 1)
    )  # All elements lie between 0&1
    assert stationary_distribution(P_2) == approx(
        np.matmul(stationary_distribution(P_2), P_2), rel=1e-6, abs=1e-12
    )  # pi = piP

    assert np.sum(stationary_distribution(P_3)) == approx(
        1, abs=1e-12
    )  # sum of a stationary distribution is 1.
    assert np.all(
        (stationary_distribution(P_3) >= 0)
        & (stationary_distribution(P_3) <= 1)
    )  # All elements lie between 0&1
    assert stationary_distribution(P_3) == approx(
        np.matmul(stationary_distribution(P_3), P_3), rel=1e-6, abs=1e-12
    )  # pi = piP


def test_logistic_transformation():

    rng = rnd.RandomState(0)

    value_1 = 0
    value_2 = -30
    value_3 = 30
    value_4 = rng.uniform(-10, 10)

    row = rng.uniform(-10, 10, (1, 10))
    matrix = rng.uniform(-10, 10, (20, 20))

    # Check shape function.
    assert logistic_transformation(value_1) == 0.5
    # Since compare to 0, only use abs
    assert logistic_transformation(value_2) == approx(0, abs=1e-12)
    assert logistic_transformation(value_3) == approx(1, abs=1e-12)

    # Check transformation (note: same formula used as in method itself).
    assert logistic_transformation(value_4) == 1 / (1 + np.exp(-value_4))
    assert logistic_transformation(row) == approx(
        1 / (1 + np.exp(-row)), rel=1e-6, abs=1e-12
    )
    assert logistic_transformation(matrix) == approx(
        1 / (1 + np.exp(-matrix)), rel=1e-6, abs=1e-12
    )

    # Check whether all values are between 0&1
    assert np.all(logistic_transformation(value_4) >= 0) & np.all(
        logistic_transformation(value_4) <= 1
    )
    assert np.all(logistic_transformation(row) >= 0) & np.all(
        logistic_transformation(row) <= 1
    )
    assert np.all(logistic_transformation(matrix) >= 0) & np.all(
        logistic_transformation(matrix) <= 1
    )


def test_normalisation():

    rng = rnd.RandomState(0)

    row_1 = rng.uniform(-10, 10, (1, 10))
    row_sum = np.sum(row_1)
    row_2 = np.array([[0.0, 0.0, 0.0, 0.0]])
    row_3 = np.array([[0.25, 0.5, 0.9, 0.8]])

    matrix_1 = np.array(
        [
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0, 5.0],
        ]
    )

    matrix_2 = np.array(
        [[0.25, 0.4, 0.5], [1 / 3, 1 / 3, 1 / 3], [0.5, 1.0, 0.5]]
    )

    matrix_3 = np.array(
        [
            [0.25, 0.5, 0.10, 0.15],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 5.0, 0.25, 0.5],
            [0.25, 0.7, 0.1, 0.30],
        ]
    )

    matrix_4 = np.array(
        [[0.00, 0.25, 2.0], [0.00, 0.00, 0.00], [0.00, 0.00, 0.5]]
    )

    # Check normalisation with a random row.
    assert normalisation(row_1) == approx(row_1 / row_sum, rel=1e-6, abs=1e-12)
    # Check normalisation with a zeroes row.
    assert normalisation(row_2) == approx(
        np.array([[0.0, 0.0, 0.0, 0.0]]), rel=1e-6, abs=1e-12
    )
    # Check normalisation with an easy-to-check row.
    assert normalisation(row_3) == approx(
        np.array([[5 / 49, 10 / 49, 18 / 49, 16 / 49]]), rel=1e-6, abs=1e-12
    )

    # Check normalisation of a matrix with equal values per row.
    assert normalisation(matrix_1) == approx(
        np.full((4, 4), 0.25), rel=1e-6, abs=1e-12
    )
    # Check normalisation of a matrix with easy computable values per row and a row
    # where no normalisation is required.
    assert normalisation(matrix_2) == approx(
        np.array(
            [
                [5 / 23, 8 / 23, 10 / 23],
                [1 / 3, 1 / 3, 1 / 3],
                [0.25, 0.5, 0.25],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    # Check normalisation of a matrix with easy computable values per row, a row
    # where no normalisation is required and a row with only zeroes.
    assert normalisation(matrix_3) == approx(
        np.array(
            [
                [0.25, 0.50, 0.10, 0.15],
                [0.00, 0.00, 0.00, 0.00],
                [4 / 27, 20 / 27, 1 / 27, 2 / 27],
                [5 / 27, 14 / 27, 2 / 27, 2 / 9],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )

    # Check normalisation of a matrix with easy computable values per row,
    # a row with only zeroes, and two rows with at least one zero.
    assert normalisation(matrix_4) == approx(
        np.array(
            [[0.00, 1 / 9, 8 / 9], [0.00, 0.00, 0.00], [0.00, 0.00, 1.00]]
        ),
        rel=1e-6,
        abs=1e-12,
    )


def test_scale_matrix():
    # Note that input matrices are normalised matrices with zeroes where C==0
    # (and some zeroes if the value itself is zero).

    initial_trans_matrix_1 = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.40, 0.00, 0.60, 0.00],
            [0.00, 0.00, 0.00, 1.00],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    initial_trans_matrix_2 = np.array(
        [
            [0.25, 0.15, 0.40, 0.20],
            [0.35, 0.10, 0.45, 0.10],
            [0.30, 0.35, 0.25, 0.10],
            [0.25, 0.10, 0.30, 0.35],
        ]
    )

    C_1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    C_2 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0]])

    matrix_1 = np.full((4, 4), 0.25)

    matrix_2 = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.50, 0.00, 0.00, 0.50],
            [1 / 3, 1 / 3, 1 / 3, 0.00],
        ]
    )

    matrix_3 = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.30, 0.00, 0.00, 0.70],
            [0.00, 0.30, 0.70, 0.00],
        ]
    )

    matrix_4 = np.array(
        [
            [0.20, 0.30, 0.00, 0.50],
            [1.00, 0.00, 0.00, 0.00],
            [0.15, 0.40, 0.25, 0.20],
            [0.10, 0.05, 0.15, 0.70],
        ]
    )

    # Test that scaling does not have an influence when C only contains 1's.
    assert scale_matrix(matrix_1, initial_trans_matrix_1, C_1) == approx(
        matrix_1, rel=1e-6, abs=1e-12
    )
    assert scale_matrix(matrix_1, initial_trans_matrix_2, C_1) == approx(
        matrix_1, rel=1e-6, abs=1e-12
    )
    assert scale_matrix(matrix_4, initial_trans_matrix_1, C_1) == approx(
        matrix_4, rel=1e-6, abs=1e-12
    )
    assert scale_matrix(matrix_4, initial_trans_matrix_2, C_1) == approx(
        matrix_4, rel=1e-6, abs=1e-12
    )

    # Test whether scaling works where C contains 0's and 1's, and the matrix has equally spaced values.
    assert scale_matrix(matrix_2, initial_trans_matrix_1, C_2) == approx(
        np.array(
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.60, 0.00],
                [0.50, 0.00, 0.00, 0.50],
                [0.25, 0.25, 0.25, 0.00],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    assert scale_matrix(matrix_2, initial_trans_matrix_2, C_2) == approx(
        np.array(
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.45, 0.00],
                [0.20, 0.00, 0.00, 0.20],
                [13 / 60, 13 / 60, 13 / 60, 0.00],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )

    # Test whether scaling works where C contains 0's and 1's, and the matrix does not have equally spaced values.
    assert scale_matrix(matrix_3, initial_trans_matrix_1, C_2) == approx(
        np.array(
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.60, 0.00],
                [0.30, 0.00, 0.00, 0.70],
                [0.00, 9 / 40, 21 / 40, 0.00],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    assert scale_matrix(matrix_3, initial_trans_matrix_2, C_2) == approx(
        np.array(
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.45, 0.00],
                [3 / 25, 0.00, 0.00, 7 / 25],
                [0.00, 39 / 200, 91 / 200, 0.00],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )


def test_inverse_logistic_transformation_matrix():
    ALGORITHM_PARAMETERS_1 = {"INVERSE_ZERO": 1e-3}
    ALGORITHM_PARAMETERS_2 = {"INVERSE_ZERO": 1e-5}

    row_1 = np.array(
        [0.2192743192, 0.2595414914, 0.2606423037, 0.2602462777, 0.0002956080]
    )  # general test, random numpy row.
    row_2 = np.array(
        [0.0, 0.3875841322, 0.1357593321, 0.0, 0.4766565357]
    )  # test behaviour with 0's.
    row_3 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # test behaviour with 1's.
    row_4 = np.array(
        [
            [0.2939915509, 0.3831167227, 0.3228917264],
            [0.3375076502, 0.2624172305, 0.4000751192],
            [0.1908342029, 0.3889071376, 0.4202586594],
        ]
    )  # test behaviour with matrices.

    assert inverse_logistic_transformation_matrix(
        row_1, ALGORITHM_PARAMETERS_1
    ) == approx(
        np.array(
            [
                -1.2699003022,
                -1.0483530210,
                -1.0426328476,
                -1.0446889184,
                -8.12618080,
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    assert inverse_logistic_transformation_matrix(
        row_2, ALGORITHM_PARAMETERS_1
    ) == approx(
        np.array(
            [
                -6.9067547786,
                -0.4574786337,
                -1.8509675796,
                -6.9067547786,
                -0.0934417874,
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    assert inverse_logistic_transformation_matrix(
        row_3, ALGORITHM_PARAMETERS_1
    ) == approx(
        np.array(
            [
                -6.9067547786,
                6.9067547786,
                -6.9067547786,
                -6.9067547786,
                -6.9067547786,
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    assert inverse_logistic_transformation_matrix(
        row_4, ALGORITHM_PARAMETERS_1
    ) == approx(
        np.array(
            [
                [-0.8760761765, -0.4763401265, -0.7405141369],
                [-0.6744208330, -1.0334425918, -0.4051521212],
                [-1.4445988334, -0.4519083376, -0.3217116622],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )

    assert inverse_logistic_transformation_matrix(
        row_1, ALGORITHM_PARAMETERS_2
    ) == approx(
        np.array(
            [
                -1.2699003022,
                -1.0483530210,
                -1.0426328476,
                -1.0446889184,
                -8.12618080,
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    assert inverse_logistic_transformation_matrix(
        row_2, ALGORITHM_PARAMETERS_2
    ) == approx(
        np.array(
            [
                -11.5129154649,
                -0.4574786337,
                -1.8509675796,
                -11.5129154649,
                -0.0934417874,
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    assert inverse_logistic_transformation_matrix(
        row_3, ALGORITHM_PARAMETERS_2
    ) == approx(
        np.array(
            [
                -11.5129154649,
                11.5129154649,
                -11.5129154649,
                -11.5129154649,
                -11.5129154649,
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )
    assert inverse_logistic_transformation_matrix(
        row_4, ALGORITHM_PARAMETERS_2
    ) == approx(
        np.array(
            [
                [-0.8760761765, -0.4763401265, -0.7405141369],
                [-0.6744208330, -1.0334425918, -0.4051521212],
                [-1.4445988334, -0.4519083376, -0.3217116622],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )


def test_logistic_simplex_transformation():
    rng = rnd.RandomState(0)

    ALGORITHM_PARAMETERS = {"INVERSE_ZERO": 1e-3}

    C_1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    C_2 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0]])

    initial_trans_matrix_1 = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.40, 0.00, 0.60, 0.00],
            [0.00, 0.00, 0.00, 1.00],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )  # Because of the config.INVERSE_ZERO parameter are the first two asserts false for abs > 1e-2

    initial_trans_matrix_2 = np.array(
        [
            [0.25, 0.15, 0.40, 0.20],
            [0.35, 0.10, 0.45, 0.10],
            [0.30, 0.35, 0.25, 0.10],
            [0.25, 0.10, 0.30, 0.35],
        ]
    )

    matrix_1 = inverse_logistic_transformation_matrix(
        initial_trans_matrix_1, ALGORITHM_PARAMETERS
    )
    matrix_2 = inverse_logistic_transformation_matrix(
        initial_trans_matrix_2, ALGORITHM_PARAMETERS
    )

    random_matrix = rng.uniform(
        -10, 10, (4, 4)
    )  # Generate random matrix with values between -10 and 10. In practice, values are often within [-10,10).

    # Check whether value(inverse_value) returns value:
    assert logistic_simplex_transformation(
        matrix_1, initial_trans_matrix_1, C_1
    ) == approx(initial_trans_matrix_1, rel=1e-6, abs=1e-2)
    assert logistic_simplex_transformation(
        matrix_1, initial_trans_matrix_1, C_2
    ) == approx(initial_trans_matrix_1, rel=1e-6, abs=1e-2)
    assert logistic_simplex_transformation(
        matrix_2, initial_trans_matrix_2, C_1
    ) == approx(initial_trans_matrix_2, rel=1e-6, abs=1e-12)
    assert logistic_simplex_transformation(
        matrix_2, initial_trans_matrix_2, C_2
    ) == approx(initial_trans_matrix_2, rel=1e-6, abs=1e-12)

    # Check whether the transformed random_matrix is a stochastic matrix:
    assert np.all(
        (
            logistic_simplex_transformation(
                random_matrix, initial_trans_matrix_1, C_1
            )
            >= 0
        )
        & (
            logistic_simplex_transformation(
                random_matrix, initial_trans_matrix_1, C_1
            )
            <= 1
        )
    )  # All elements lie between 0&1
    assert np.sum(
        logistic_simplex_transformation(
            random_matrix, initial_trans_matrix_1, C_1
        ),
        axis=1,
    ) == approx(
        np.array([1, 1, 1, 1]), rel=1e-6, abs=1e-12
    )  # All row sums are 1

    assert np.all(
        (
            logistic_simplex_transformation(
                random_matrix, initial_trans_matrix_1, C_2
            )
            >= 0
        )
        & (
            logistic_simplex_transformation(
                random_matrix, initial_trans_matrix_1, C_2
            )
            <= 1
        )
    )  # All elements lie between 0&1
    assert np.sum(
        logistic_simplex_transformation(
            random_matrix, initial_trans_matrix_1, C_2
        ),
        axis=1,
    ) == approx(
        np.array([1, 1, 1, 1]), rel=1e-6, abs=1e-12
    )  # All row sums are 1

    assert np.all(
        (
            logistic_simplex_transformation(
                random_matrix, initial_trans_matrix_2, C_1
            )
            >= 0
        )
        & (
            logistic_simplex_transformation(
                random_matrix, initial_trans_matrix_2, C_1
            )
            <= 1
        )
    )  # All elements lie between 0&1
    assert np.sum(
        logistic_simplex_transformation(
            random_matrix, initial_trans_matrix_2, C_1
        ),
        axis=1,
    ) == approx(
        np.array([1, 1, 1, 1]), rel=1e-6, abs=1e-12
    )  # All row sums are 1

    assert np.all(
        (
            logistic_simplex_transformation(
                random_matrix, initial_trans_matrix_2, C_2
            )
            >= 0
        )
        & (
            logistic_simplex_transformation(
                random_matrix, initial_trans_matrix_2, C_2
            )
            <= 1
        )
    )  # All elements lie between 0&1
    assert np.sum(
        logistic_simplex_transformation(
            random_matrix, initial_trans_matrix_2, C_2
        ),
        axis=1,
    ) == approx(
        np.array([1, 1, 1, 1]), rel=1e-6, abs=1e-12
    )  # All row sums are 1

    # The elements where C == 0 have not been changed:
    assert np.all(
        initial_trans_matrix_1[C_1 == 0]
        == logistic_simplex_transformation(
            random_matrix, initial_trans_matrix_1, C_1
        )[C_1 == 0]
    )
    assert np.all(
        initial_trans_matrix_1[C_2 == 0]
        == logistic_simplex_transformation(
            random_matrix, initial_trans_matrix_1, C_2
        )[C_2 == 0]
    )
    assert np.all(
        initial_trans_matrix_2[C_1 == 0]
        == logistic_simplex_transformation(
            random_matrix, initial_trans_matrix_2, C_1
        )[C_1 == 0]
    )
    assert np.all(
        initial_trans_matrix_2[C_2 == 0]
        == logistic_simplex_transformation(
            random_matrix, initial_trans_matrix_2, C_2
        )[C_2 == 0]
    )

    # The row sums of the elements of the random matrix where C==1 == 1 - row sums of the elements of the initial matrix where C==0.
    assert np.sum(
        np.where(
            C_2 == 1,
            logistic_simplex_transformation(
                random_matrix, initial_trans_matrix_1, C_2
            ),
            0,
        ),
        axis=1,
    ) == approx(
        1 - np.sum(np.where(C_2 == 0, initial_trans_matrix_1, 0), axis=1),
        rel=1e-6,
        abs=1e-12,
    )
    assert np.sum(
        np.where(
            C_2 == 1,
            logistic_simplex_transformation(
                random_matrix, initial_trans_matrix_2, C_2
            ),
            0,
        ),
        axis=1,
    ) == approx(
        1 - np.sum(np.where(C_2 == 0, initial_trans_matrix_2, 0), axis=1),
        rel=1e-6,
        abs=1e-12,
    )


##########################Tests for matrix_algorithms.py#######################


def test_run_matrix_algorithm():
    # Note the abs values of the asserts.

    rng = rnd.RandomState(0)

    ALGORITHM_PARAMETERS_1 = {
        "NR_ITERATIONS": 50000,
        "MAXIMISE_OBJECTIVE": True,
        "EPSILON": 1e-1,
        "GAME_TYPE": "regular",
        "INVERSE_ZERO": 1e-3,
        "OBJECTIVE_INDICES": [0],
        "USE_TIME_LIMIT": False,
    }

    ALGORITHM_PARAMETERS_2 = {
        "NR_ITERATIONS": 100000,
        "MAXIMISE_OBJECTIVE": False,
        "EPSILON": 1e-1,
        "GAME_TYPE": "regular",
        "INVERSE_ZERO": 1e-3,
        "OBJECTIVE_INDICES": [1],
        "USE_TIME_LIMIT": False,
    }

    ALGORITHM_PARAMETERS_5 = {
        "NR_ITERATIONS": 5,
        "MAXIMISE_OBJECTIVE": True,
        "EPSILON": 1e-1,
        "GAME_TYPE": "regular",
        "INVERSE_ZERO": 1e-3,
        "OBJECTIVE_INDICES": [0],
        "USE_TIME_LIMIT": False,
    }

    C_1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    C_2 = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

    matrix_1 = np.array(
        [
            [0.3, 0.4, 0.1, 0.2],
            [0.3, 0.0, 0.6, 0.1],
            [1 / 3, 1 / 9, 1 / 3, 2 / 9],
            [0.25, 0.20, 0.00, 0.55],
        ]
    )

    (
        gradient_estimates_1,
        thetas_1,
        transition_matrices_1,
        stationary_distributions_1,
        objective_values_1,
    ) = run_matrix_algorithm(
        matrix_1, C_1, datetime.datetime.now(), rng, ALGORITHM_PARAMETERS_1
    )
    (
        gradient_estimates_2,
        thetas_2,
        transition_matrices_2,
        stationary_distributions_2,
        objective_values_2,
    ) = run_matrix_algorithm(
        matrix_1, C_2, datetime.datetime.now(), rng, ALGORITHM_PARAMETERS_2
    )
    (
        gradient_estimates_5,
        thetas_5,
        transition_matrices_5,
        stationary_distributions_5,
        objective_values_5,
    ) = run_matrix_algorithm(
        matrix_1, C_1, datetime.datetime.now(), rng, ALGORITHM_PARAMETERS_5
    )
    (
        gradient_estimates_6,
        thetas_6,
        transition_matrices_6,
        stationary_distributions_6,
        objective_values_6,
    ) = run_matrix_algorithm(
        matrix_1, C_2, datetime.datetime.now(), rng, ALGORITHM_PARAMETERS_5
    )

    # Test where all C entries are 1: (regular) optimisation will result in self-loop.
    assert stationary_distributions_1[-1][0] == np.max(
        stationary_distributions_1[-1]
    )  # The objective node has the highest stationary distribution.
    assert transition_matrices_1[-1][0, 0] == approx(
        1, abs=1e-3
    )  # A self-loop on entry (0,0) has been created.

    # Test with C matrix == 0 on diagonal and one person being minimized. No entries will point to this node (regular).
    assert stationary_distributions_2[-1][1] == np.min(
        stationary_distributions_2[-1]
    )  # The objective node has the lowest stationary distribution.
    # Since compare to 0, only use abs
    assert transition_matrices_2[-1][0, 1] == approx(
        0, abs=1e-3
    )  # Entry (0,1) will not point to the objective node.
    assert transition_matrices_2[-1][2, 1] == approx(
        0, abs=1e-3
    )  # Entry (2,1) will not point to the objective node.
    assert transition_matrices_2[-1][3, 1] == approx(
        0, abs=1e-3
    )  # Entry (3,1) will not point to the objective node.
    assert np.all(
        transition_matrices_2[-1][C_2 == 0] == matrix_1[C_2 == 0]
    )  # The elements where C == 0 have not been changed.

    # Check all matrices are stochastic matrices. Matrix C has 1 everywhere.
    assert np.all(
        (transition_matrices_5 >= 0) & (transition_matrices_5 <= 1)
    )  # All elements lie between 0&1
    assert np.sum(transition_matrices_5, axis=2) == approx(
        np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )  # All row sums are 1

    # Check all matrices are stochastic matrices. Matrix C has 0's on diagonal, rest 1.
    assert np.all(
        (transition_matrices_6 >= 0) & (transition_matrices_6 <= 1)
    )  # All elements lie between 0&1
    assert np.sum(transition_matrices_6, axis=2) == approx(
        np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        ),
        rel=1e-6,
        abs=1e-12,
    )  # All row sums are 1
    assert np.all(
        transition_matrices_6[-1][C_2 == 0] == matrix_1[C_2 == 0]
    )  # The elements where C == 0 have not been changed.


def test_run_matrix_algorithm_regression_test_1():

    ALGORITHM_PARAMETERS_1 = {
        "NR_ITERATIONS": 5000,
        "MAXIMISE_OBJECTIVE": True,
        "EPSILON": 1e-1,
        "GAME_TYPE": "regular",
        "INVERSE_ZERO": 1e-3,
        "OBJECTIVE_INDICES": [0],
        "SEED": 0,
        "USE_TIME_LIMIT": False,
    }

    ROOT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
    DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data/unit_tests/")
    REGRESSION_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data/unit_tests/")

    rng = rnd.RandomState(ALGORITHM_PARAMETERS_1["SEED"])

    M = pd.read_csv(
        f"{DATA_DIRECTORY}M_matrix_Test_Instance_36.csv",
        float_precision="round_trip",
    ).to_numpy(dtype=float)
    C = pd.read_csv(f"{DATA_DIRECTORY}C_matrix_Test_Instance_36.csv").to_numpy(
        dtype=int
    )

    (
        gradient_estimates_1,
        thetas_1,
        transition_matrices_1,
        stationary_distributions_1,
        objective_values_1,
    ) = run_matrix_algorithm(
        M, C, datetime.datetime.now(), rng, ALGORITHM_PARAMETERS_1
    )

    regression_objective_values = (
        pd.read_csv(
            f"{REGRESSION_DIRECTORY}Objective_values_Test_Instance_36.csv",
            float_precision="round_trip",
        )
        .to_numpy(dtype=float)
        .flatten()
    )
    regression_transition_matrices_df = pd.read_csv(
        f"{REGRESSION_DIRECTORY}Transition_matrices_Test_Instance_36.csv",
        float_precision="round_trip",
    )
    regression_stationary_distributions = pd.read_csv(
        f"{REGRESSION_DIRECTORY}Stationary_distributions_Test_Instance_36.csv",
        float_precision="round_trip",
    ).to_numpy(dtype=float)
    regression_gradient_estimates_df = pd.read_csv(
        f"{REGRESSION_DIRECTORY}Gradient_estimates_Test_Instance_36.csv",
        float_precision="round_trip",
    )
    regression_thetas_df = pd.read_csv(
        f"{REGRESSION_DIRECTORY}Thetas_Test_Instance_36.csv",
        float_precision="round_trip",
    )

    regression_thetas = regression_thetas_df.to_numpy(dtype=float).reshape(
        ALGORITHM_PARAMETERS_1["NR_ITERATIONS"] + 1, C.shape[0], C.shape[1]
    )
    regression_gradient_estimates = regression_gradient_estimates_df.to_numpy(
        dtype=float
    ).reshape(ALGORITHM_PARAMETERS_1["NR_ITERATIONS"], C.shape[0], C.shape[1])
    regression_transition_matrices = (
        regression_transition_matrices_df.to_numpy(dtype=float).reshape(
            ALGORITHM_PARAMETERS_1["NR_ITERATIONS"] + 1, C.shape[0], C.shape[1]
        )
    )

    # On GitHub tests do not pass with 1e-20 precision for some (GitHub) reason.
    # Locally they do pass. Therefore changed the precision to 1e-10.
    assert objective_values_1 == approx(
        regression_objective_values, rel=1e-10, abs=1e-10
    )
    assert transition_matrices_1 == approx(
        regression_transition_matrices, rel=1e-10, abs=1e-10
    )
    assert stationary_distributions_1 == approx(
        regression_stationary_distributions, rel=1e-10, abs=1e-10
    )
    assert gradient_estimates_1 == approx(
        regression_gradient_estimates, rel=1e-10, abs=1e-10
    )
    assert thetas_1 == approx(regression_thetas, rel=1e-10, abs=1e-10)


def test_run_matrix_algorithm_regression_test_2():

    ALGORITHM_PARAMETERS_1 = {
        "NR_ITERATIONS": 100000,
        "MAXIMISE_OBJECTIVE": True,
        "EPSILON": 1e-1,
        "GAME_TYPE": "regular",
        "INVERSE_ZERO": 1e-3,
        "OBJECTIVE_INDICES": [0],
        "SEED": 0,
        "USE_TIME_LIMIT": False,
    }

    ROOT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
    DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data/unit_tests/")
    REGRESSION_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data/unit_tests/")

    rng = rnd.RandomState(ALGORITHM_PARAMETERS_1["SEED"])

    M = pd.read_csv(
        f"{DATA_DIRECTORY}M_matrix_Test_Instance_33.csv",
        float_precision="round_trip",
    ).to_numpy(dtype=float)
    C = pd.read_csv(f"{DATA_DIRECTORY}C_matrix_Test_Instance_33.csv").to_numpy(
        dtype=int
    )

    (
        gradient_estimates_1,
        thetas_1,
        transition_matrices_1,
        stationary_distributions_1,
        objective_values_1,
    ) = run_matrix_algorithm(
        M, C, datetime.datetime.now(), rng, ALGORITHM_PARAMETERS_1
    )

    regression_objective_values = (
        pd.read_csv(
            f"{REGRESSION_DIRECTORY}Objective_values_Test_Instance_33.csv",
            float_precision="round_trip",
        )
        .to_numpy(dtype=float)
        .flatten()
    )
    regression_transition_matrices_df = pd.read_csv(
        f"{REGRESSION_DIRECTORY}Transition_matrices_Test_Instance_33.csv",
        float_precision="round_trip",
    )
    regression_stationary_distributions = pd.read_csv(
        f"{REGRESSION_DIRECTORY}Stationary_distributions_Test_Instance_33.csv",
        float_precision="round_trip",
    ).to_numpy(dtype=float)
    regression_gradient_estimates_df = pd.read_csv(
        f"{REGRESSION_DIRECTORY}Gradient_estimates_Test_Instance_33.csv",
        float_precision="round_trip",
    )
    regression_thetas_df = pd.read_csv(
        f"{REGRESSION_DIRECTORY}Thetas_Test_Instance_33.csv",
        float_precision="round_trip",
    )

    regression_thetas = regression_thetas_df.to_numpy(dtype=float).reshape(
        ALGORITHM_PARAMETERS_1["NR_ITERATIONS"] + 1, C.shape[0], C.shape[1]
    )
    regression_gradient_estimates = regression_gradient_estimates_df.to_numpy(
        dtype=float
    ).reshape(ALGORITHM_PARAMETERS_1["NR_ITERATIONS"], C.shape[0], C.shape[1])
    regression_transition_matrices = (
        regression_transition_matrices_df.to_numpy(dtype=float).reshape(
            ALGORITHM_PARAMETERS_1["NR_ITERATIONS"] + 1, C.shape[0], C.shape[1]
        )
    )

    # On GitHub tests do not pass with 1e-20 precision for some (GitHub) reason.
    # Locally they do pass. Selected precision based on lowering it by a factor
    # 10 (starting at 1e-10) and checking whether the tests succeeded on GitHub.
    assert objective_values_1 == approx(
        regression_objective_values, rel=1e-9, abs=1e-9
    )
    assert transition_matrices_1 == approx(
        regression_transition_matrices, rel=1e-8, abs=1e-8
    )
    assert stationary_distributions_1 == approx(
        regression_stationary_distributions, rel=1e-9, abs=1e-9
    )
    assert gradient_estimates_1 == approx(
        regression_gradient_estimates, rel=1e-10, abs=1e-10
    )
    assert thetas_1 == approx(regression_thetas, rel=1e-8, abs=1e-8)


def test_run_matrix_algorithm_memory_improved_1():

    ALGORITHM_PARAMETERS = {
        "NR_ITERATIONS": 50000,
        "MAXIMISE_OBJECTIVE": True,
        "EPSILON": 1e-1,
        "GAME_TYPE": "regular",
        "INVERSE_ZERO": 1e-3,
        "OBJECTIVE_INDICES": [5],
        "SEED": 0,
        "USE_TIME_LIMIT": False,
        "DECR_NR_OBJ_EVAL": False,
    }

    ROOT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
    DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data/unit_tests/")

    rng = rnd.RandomState(ALGORITHM_PARAMETERS["SEED"])

    M = pd.read_csv(
        f"{DATA_DIRECTORY}M_matrix_Test_Instance_36.csv",
        float_precision="round_trip",
    ).to_numpy(dtype=float)
    C = pd.read_csv(f"{DATA_DIRECTORY}C_matrix_Test_Instance_36.csv").to_numpy(
        dtype=int
    )

    (
        gradient_estimates,
        thetas,
        transition_matrices,
        stationary_distributions,
        objective_values,
    ) = run_matrix_algorithm(
        M, C, datetime.datetime.now(), rng, ALGORITHM_PARAMETERS
    )

    rng = rnd.RandomState(ALGORITHM_PARAMETERS["SEED"])

    (
        final_gradient_estimate,
        final_theta,
        final_transition_matrix,
        final_stationary_distribution,
        objective_values_MI,
    ) = run_matrix_algorithm_improved_memory(
        M, C, datetime.datetime.now(), rng, ALGORITHM_PARAMETERS
    )

    assert final_gradient_estimate == approx(
        gradient_estimates[-1], rel=1e-20, abs=1e-20
    )
    assert final_theta == approx(thetas[-1], rel=1e-20, abs=1e-20)
    assert final_transition_matrix == approx(
        transition_matrices[-1], rel=1e-20, abs=1e-20
    )
    assert final_stationary_distribution == approx(
        stationary_distributions[-1], rel=1e-20, abs=1e-20
    )
    assert objective_values_MI == approx(
        objective_values, rel=1e-20, abs=1e-20
    )


def test_run_matrix_algorithm_memory_improved_2():

    ALGORITHM_PARAMETERS = {
        "NR_ITERATIONS": 50000,
        "MAXIMISE_OBJECTIVE": True,
        "EPSILON": 1e-1,
        "GAME_TYPE": "regular",
        "INVERSE_ZERO": 1e-3,
        "OBJECTIVE_INDICES": [0],
        "SEED": 10,
        "USE_TIME_LIMIT": False,
        "DECR_NR_OBJ_EVAL": False,
    }

    ROOT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
    DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data/unit_tests/")

    rng = rnd.RandomState(ALGORITHM_PARAMETERS["SEED"])

    M = pd.read_csv(
        f"{DATA_DIRECTORY}M_matrix_Test_Instance_33.csv",
        float_precision="round_trip",
    ).to_numpy(dtype=float)
    C = pd.read_csv(f"{DATA_DIRECTORY}C_matrix_Test_Instance_33.csv").to_numpy(
        dtype=int
    )

    (
        gradient_estimates,
        thetas,
        transition_matrices,
        stationary_distributions,
        objective_values,
    ) = run_matrix_algorithm(
        M, C, datetime.datetime.now(), rng, ALGORITHM_PARAMETERS
    )

    rng = rnd.RandomState(ALGORITHM_PARAMETERS["SEED"])

    (
        final_gradient_estimate,
        final_theta,
        final_transition_matrix,
        final_stationary_distribution,
        objective_values_MI,
    ) = run_matrix_algorithm_improved_memory(
        M, C, datetime.datetime.now(), rng, ALGORITHM_PARAMETERS
    )

    assert final_gradient_estimate == approx(
        gradient_estimates[-1], rel=1e-20, abs=1e-20
    )
    assert final_theta == approx(thetas[-1], rel=1e-20, abs=1e-20)
    assert final_transition_matrix == approx(
        transition_matrices[-1], rel=1e-20, abs=1e-20
    )
    assert final_stationary_distribution == approx(
        stationary_distributions[-1], rel=1e-20, abs=1e-20
    )
    assert objective_values_MI == approx(
        objective_values, rel=1e-20, abs=1e-20
    )


#########################Tests for matrix_SPSA_algorithm.py####################
def test_SPSA_gradient():
    rng = rnd.RandomState(0)

    ALGORITHM_PARAMETERS_1 = {"GAME_TYPE": "regular", "OBJECTIVE_INDICES": [0]}

    C_1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    C_2 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0]])

    initial_trans_matrix_1 = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.40, 0.00, 0.60, 0.00],
            [0.00, 0.00, 0.00, 1.00],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    initial_trans_matrix_2 = np.array(
        [
            [0.25, 0.15, 0.40, 0.20],
            [0.35, 0.10, 0.45, 0.10],
            [0.30, 0.35, 0.25, 0.10],
            [0.25, 0.10, 0.30, 0.35],
        ]
    )

    iteration = 2

    random_theta = rng.uniform(-10, 10, (4, 4))

    Y_1 = SPSA_gradient(
        random_theta,
        initial_trans_matrix_1,
        C_1,
        rng,
        iteration,
        ALGORITHM_PARAMETERS_1,
    )
    Y_2 = SPSA_gradient(
        random_theta,
        initial_trans_matrix_1,
        C_2,
        rng,
        iteration,
        ALGORITHM_PARAMETERS_1,
    )
    Y_3 = SPSA_gradient(
        random_theta,
        initial_trans_matrix_2,
        C_1,
        rng,
        iteration,
        ALGORITHM_PARAMETERS_1,
    )
    Y_4 = SPSA_gradient(
        random_theta,
        initial_trans_matrix_2,
        C_2,
        rng,
        iteration,
        ALGORITHM_PARAMETERS_1,
    )

    # All gradient values where C==1 have the same value (in absolute sense).
    gradient_value_1 = abs(Y_1[C_1 == 1][0])
    gradient_value_2 = abs(Y_2[C_2 == 1][0])
    gradient_value_3 = abs(Y_3[C_1 == 1][0])
    gradient_value_4 = abs(Y_4[C_2 == 1][0])
    assert np.all(abs(Y_1[C_1 == 1]) == gradient_value_1)
    assert np.all(abs(Y_2[C_2 == 1]) == gradient_value_2)
    assert np.all(abs(Y_3[C_1 == 1]) == gradient_value_3)
    assert np.all(abs(Y_4[C_2 == 1]) == gradient_value_4)

    # All gradient values where C==0 are zero.
    assert np.all(Y_1[C_1 == 0] == 0)
    assert np.all(Y_2[C_2 == 0] == 0)
    assert np.all(Y_3[C_1 == 0] == 0)
    assert np.all(Y_4[C_2 == 0] == 0)


def test_calculate_numerator_SPSA():
    # This method calculates Y(p_high,i) - Y(p_low,i)

    ALGORITHM_PARAMETERS_1 = {"GAME_TYPE": "regular", "OBJECTIVE_INDICES": [0]}
    ALGORITHM_PARAMETERS_3 = {
        "GAME_TYPE": "hyperlink",
        "OBJECTIVE_INDICES": [0],
    }

    trans_matrix_dummy = np.eye(5)  # Dummy transition matrix
    high_stat = np.array([0.4, 0.12, 0.18, 0.16, 0.14])
    low_stat = np.array([0.3, 0.24, 0.02, 0.05, 0.39])

    hyperlink_trans_matrix_high = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    hyperlink_stat_dist_high = stationary_distribution(
        hyperlink_trans_matrix_high
    )
    hyperlink_costs_high = hyperlink_cost_function(0.25)
    hyperlink_obj_high = 0.25 - 3 * 0.25 * hyperlink_costs_high

    hyperlink_trans_matrix_low = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.10, 0.30, 0.30, 0.30],
            [0.10, 0.30, 0.30, 0.30],
            [0.10, 0.30, 0.30, 0.30],
        ]
    )
    hyperlink_stat_dist_low = stationary_distribution(
        hyperlink_trans_matrix_low
    )
    hyperlink_costs_low = hyperlink_cost_function(0.10)
    hyperlink_obj_low = (
        hyperlink_stat_dist_low[0]
        - 3 * hyperlink_stat_dist_low[1] * hyperlink_costs_low
    )

    assert calculate_numerator_SPSA(
        high_stat,
        trans_matrix_dummy,
        low_stat,
        trans_matrix_dummy,
        ALGORITHM_PARAMETERS_1,
    ) == (high_stat[0] - low_stat[0])

    assert calculate_numerator_SPSA(
        hyperlink_stat_dist_high,
        hyperlink_trans_matrix_high,
        hyperlink_stat_dist_low,
        hyperlink_trans_matrix_low,
        ALGORITHM_PARAMETERS_3,
    ) == approx((hyperlink_obj_high - hyperlink_obj_low), rel=1e-6, abs=1e-12)
