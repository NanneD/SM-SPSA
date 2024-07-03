#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math
import numpy as np

from typing import Any


def generate_centred_mass_initial_matrix(
    M: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """
    Applies the centred mass heuristic to a matrix.

    The elements where ``C=0`` are not changed.

    Parameters
    ----------
    M : np.ndarray
        The transition matrix to which the centred mass heuristic is applied.
        A copy of this matrix is created.
    C : np.ndarray
        The binary adjustment matrix.

    Returns
    -------
    matrix : np.ndarray
        The matrix obtained with the centred mass heuristic.

    """

    matrix = copy.deepcopy(M)

    for i in range(matrix.shape[0]):
        if matrix[i][C[i] == 1].size != 0:
            available_prob = 1 - np.sum(matrix[i][C[i] == 0])
            matrix[i][C[i] == 1] = available_prob / matrix[i][C[i] == 1].size

    print("The centred start matrix is: \n", matrix)
    return matrix


def hyperlink_cost_function(x: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the hyperlink cost function for a transition probability or matrix.

    The objective function is equal to :math:`0.42sin(1.5\\pi x)+ 1.92x^3`.

    Parameters
    ----------
    x : float | np.ndarray
        The transition probability (float) or
        the transition matrix (np.ndarray).

    Returns
    -------
    float | np.ndarray
        The hyperlink costs as float (if input is float) or as np.ndarray
        (if input is np.ndarray).

    """

    return 0.42 * np.sin(1.5 * math.pi * x) + np.power(x, 3) * 1.92


def calculate_objective_value(
    stat_dist: np.ndarray,
    transition_matrix: np.ndarray,
    ALGORITHM_PARAMETERS: dict[str, Any],
) -> float:
    """
    Calculates the objective function value.

    If ``GAME_TYPE="regular"``, the stationary distribution of a single node
    is optimised. If ``GAME_TYPE="hyperlink"``, the stationary distribution of
    a single node minus costs is optimised.

    Parameters
    ----------
    stat_dist : np.ndarray
        The stationary distribution.
    transition_matrix : np.ndarray
        The transition matrix.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameters ``GAME_TYPE`` and
        ``OBJECTIVE_INDICES`` are at least necessary. See ``main.py`` for
        parameter explanations.

    Raises
    ------
    Exception
        If an invalid ``GAME_TYPE`` is provided (not "regular" or "hyperlink").

    Returns
    -------
    function : float
        The objective function value.

    """

    if ALGORITHM_PARAMETERS["GAME_TYPE"] == "hyperlink":
        hyperlink_index = ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"][0]
        function = stat_dist[hyperlink_index]
        loop_nodes = list(range(0, len(stat_dist)))
        loop_nodes.remove(hyperlink_index)
        for i in loop_nodes:
            function -= stat_dist[i] * hyperlink_cost_function(
                transition_matrix[i, hyperlink_index]
            )
    elif ALGORITHM_PARAMETERS["GAME_TYPE"] == "regular":
        function = stat_dist[ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"][0]]
    else:
        raise Exception("Wrong GAME_TYPE specified. Please change it.")

    return function


def stationary_distribution(Matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the stationary distribution of a transition matrix.

    A QR decomposition is used to calculate the stationary distribution.

    Parameters
    ----------
    Matrix : np.ndarray
        A transition matrix.

    Raises
    ------
    Exception
        If the provided matrix is not a transition matrix (if it contains
        values larger than 1, smaller than 0 or the row sum is not equal to 1).

    Returns
    -------
    np.ndarray
        The stationary distribution.

    """

    if np.any(Matrix > 1):
        raise Exception(
            "The transition matrix contains values larger than 1. "
            "This is not correct."
        )
    if np.any(Matrix < 0):
        raise Exception(
            "The transition matrix contains negative values. "
            "This is not correct."
        )
    if (np.any(np.sum(Matrix, axis=1) >= 1.00000001)) or (
        np.any(np.sum(Matrix, axis=1) <= 0.99999999)
    ):
        raise Exception(
            "The transition matrix contains a row that does not have sum 1. "
            "This is not correct."
        )

    Matrix = Matrix.T - np.eye(Matrix.shape[0])
    A = np.vstack((Matrix, np.ones((1, Matrix.shape[1]))))
    b = np.vstack((np.zeros((Matrix.shape[0], 1)), [1]))

    Q, R = np.linalg.qr(A)
    QTb = np.dot(Q.T, b)

    stat = np.linalg.solve(R, QTb)
    return stat.flatten()


def inverse_logistic_transformation_matrix(
    Matrix: np.ndarray, ALGORITHM_PARAMETERS: dict[str, Any]
) -> np.ndarray:
    """
    Calculates the inverse logistic transformation of a matrix.

    If the matrix contains 0 or 1 values, they are first set to the values
    ``INVERSE_ZERO`` and ``1-INVERSE_ZERO``, respectively, to avoid
    computational problems due to the use of the natural logarithm. The inverse
    function is equal to :math:`ln(x/(1-x))` for input :math:`x`.

    Parameters
    ----------
    Matrix : np.ndarray
        The matrix that is inversely transformed.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameter ``INVERSE_ZERO`` is at least
        necessary. See ``main.py`` for parameter explanations.
    Returns
    -------
    transformed_matrix : np.ndarray
        The inversely transformed matrix.

    """

    X = copy.deepcopy(Matrix)

    X[X == 0.0] = ALGORITHM_PARAMETERS["INVERSE_ZERO"]
    X[X == 1.0] = 1 - ALGORITHM_PARAMETERS["INVERSE_ZERO"]

    transformed_matrix = np.log(X / (np.subtract(1, X)))

    return transformed_matrix


def logistic_transformation(matrix: np.ndarray) -> np.ndarray:
    """
    Applies the logistic function to a matrix.

    The logistic function is equal to :math:`1/(1+exp(-x))` for
    input :math:`x`.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix.

    Returns
    -------
    np.ndarray
        The transformed matrix.

    """

    return 1 / (1 + np.exp(-matrix))


def normalisation(matrix: np.ndarray) -> np.ndarray:
    """
    Normalises a matrix so that the row sums are equal to 1.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix that is normalised.

    Returns
    -------
    np.ndarray
        The normalised matrix.

    """

    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix, row_sum, out=np.zeros_like(matrix), where=row_sum != 0
    )


def scale_matrix(
    matrix: np.ndarray, initial_trans_matrix: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """
    Scales a matrix.

    The matrix is scaled so that row sums are equal to 1 while taking the
    elements that cannot be optimized (``C=0``) into account. These elements
    receive the value of the initial transition matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix that is scaled.
    initial_trans_matrix : np.ndarray
        The initial transition matrix.
    C : np.ndarray
        The binary adjustment matrix.

    Returns
    -------
    np.ndarray
        The scaled matrix.

    """

    scale_factors = 1 - (
        np.sum(initial_trans_matrix, axis=1, where=C == 0, keepdims=True)
    )
    return np.multiply(scale_factors, matrix)


def logistic_simplex_transformation(
    matrix: np.ndarray, initial_trans_matrix: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """
    Performs the complete transformation yielding a transition matrix.

    The transformation is performed to transform a matrix in the unconstrained
    optimization space to a matrix in the stochastic matrix space. First, the
    logistic transformation is applied to the matrix and next the rows are
    normalised and scaled so that a transition matrix that takes the elements
    where ``C=0`` into account is obtained. These elements receive the value of
    the initial transition matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix in the unconstrained space that is transformed. A copy of
        this matrix is created.
    initial_trans_matrix : np.ndarray
        The initial transition matrix.
    C : np.ndarray
        The binary adjustment matrix.

    Returns
    -------
    X : np.ndarray
        The transformed matrix.

    """

    X = copy.deepcopy(matrix)

    X = logistic_transformation(X)

    X[C == 0] = 0
    X = normalisation(X)
    X = scale_matrix(X, initial_trans_matrix, C)
    X[C == 0] = initial_trans_matrix[C == 0]
    return X
