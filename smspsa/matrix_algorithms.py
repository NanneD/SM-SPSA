#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import numpy as np

from typing import Any
from algorithm_functions import (
    inverse_logistic_transformation_matrix,
    stationary_distribution,
    logistic_simplex_transformation,
    calculate_objective_value,
)
from matrix_SPSA_algorithm import SPSA_gradient


def run_matrix_algorithm(
    M: np.ndarray,
    C: np.ndarray,
    start_time_algorithm: datetime.datetime,
    rng: np.random.mtrand.RandomState | np.random._generator.Generator,
    ALGORITHM_PARAMETERS: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs SM-SPSA on a transition matrix and binary matrix C.

    Parameters
    ----------
    M : np.ndarray
        The transition matrix that is optimised.
    C : np.ndarray
        The binary adjustment matrix.
    start_time_algorithm : datetime.datetime
        The start time of the algorithm.
    rng : np.random.mtrand.RandomState | np.random._generator.Generator
        A pseudo-random number generator.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameters ``NR_ITERATIONS``,
        ``MAXIMISE_OBJECTIVE``, ``EPSILON``,
        ``USE_TIME_LIMIT`` and ``TIME_LIMIT_SEC`` are at least necessary.
        See ``main.py`` for parameter explanations. Note that methods that are
        called within this method may require more parameters. Note that
        ``NR_ITERATIONS`` is set in the methods of ``run_algorithm.py``.

    Returns
    -------
    gradient_estimates : np.ndarray
        The gradient estimate at each iteration.
    thetas : np.ndarray
        The matrix in the unconstrained space at each iteration.
    transition_matrices : np.ndarray
        The transition matrix in the stochastic matrix space at each
        iteration.
    stationary_distributions : np.ndarray
        The stationary distribution of all nodes at each iteration.
    objective_values : np.ndarray
        The objective value at each iteration.

    """

    # Initialization
    gradient_estimates = np.zeros(
        (ALGORITHM_PARAMETERS["NR_ITERATIONS"], M.shape[0], M.shape[1])
    )
    thetas = np.zeros(
        (ALGORITHM_PARAMETERS["NR_ITERATIONS"] + 1, M.shape[0], M.shape[1])
    )
    transition_matrices = np.zeros(
        (ALGORITHM_PARAMETERS["NR_ITERATIONS"] + 1, M.shape[0], M.shape[1])
    )
    stationary_distributions = np.zeros(
        (ALGORITHM_PARAMETERS["NR_ITERATIONS"] + 1, M.shape[0])
    )
    objective_values = np.zeros(ALGORITHM_PARAMETERS["NR_ITERATIONS"] + 1)

    transition_matrices[0] = M
    stationary_distributions[0] = stationary_distribution(M)
    objective_values[0] = calculate_objective_value(
        stationary_distributions[0], M, ALGORITHM_PARAMETERS
    )

    # Inverse transformation
    thetas[0] = inverse_logistic_transformation_matrix(M, ALGORITHM_PARAMETERS)

    for i in range(1, ALGORITHM_PARAMETERS["NR_ITERATIONS"] + 1):

        if i % 50000 == 0:
            print(f"At {datetime.datetime.now()} at iteration {i}.")

        Y_grad = SPSA_gradient(
            thetas[i - 1], M, C, rng, i, ALGORITHM_PARAMETERS
        )

        gradient_estimates[i - 1] = Y_grad

        if ALGORITHM_PARAMETERS["MAXIMISE_OBJECTIVE"]:
            thetas[i] = (
                thetas[i - 1] + ALGORITHM_PARAMETERS["EPSILON"] * Y_grad
            )
        else:
            thetas[i] = (
                thetas[i - 1] - ALGORITHM_PARAMETERS["EPSILON"] * Y_grad
            )

        transition_matrices[i] = logistic_simplex_transformation(
            thetas[i], M, C
        )
        stationary_distributions[i] = stationary_distribution(
            transition_matrices[i]
        )
        objective_values[i] = calculate_objective_value(
            stationary_distributions[i],
            transition_matrices[i],
            ALGORITHM_PARAMETERS,
        )

        if ALGORITHM_PARAMETERS["USE_TIME_LIMIT"]:
            if (
                datetime.datetime.now() - start_time_algorithm
            ).total_seconds() > ALGORITHM_PARAMETERS["TIME_LIMIT_SEC"]:
                print(
                    "The time limit has passed. Break from algorithm "
                    f"for-loop. i={i}."
                )
                gradient_estimates = gradient_estimates[:i, :, :]
                thetas = thetas[: i + 1, :, :]
                transition_matrices = transition_matrices[: i + 1, :, :]
                stationary_distributions = stationary_distributions[: i + 1, :]
                objective_values = objective_values[: i + 1]
                break

    return (
        gradient_estimates,
        thetas,
        transition_matrices,
        stationary_distributions,
        objective_values,
    )


def run_matrix_algorithm_improved_memory(
    M: np.ndarray,
    C: np.ndarray,
    start_time_algorithm: datetime.datetime,
    rng: np.random.mtrand.RandomState | np.random._generator.Generator,
    ALGORITHM_PARAMETERS: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs SM-SPSA "memory optimised" on a transition matrix and binary matrix C.


    The main difference between the "normal" and "memory optimised" version of
    the algorithm is that, in case of the memory optimised version, not all
    matrices are stored for each iteration. Instead, only the matrices of the
    last iteration are stored. This version is especially useful for large
    networks.

    Parameters
    ----------
    M : np.ndarray
        The transition matrix that is optimised.
    C : np.ndarray
        The binary adjustment matrix.
    start_time_algorithm : datetime.datetime
        The start time of the algorithm.
    rng : np.random.mtrand.RandomState | np.random._generator.Generator
        A pseudo-random number generator.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameters ``DECR_NR_OBJ_EVAL``,
        ``NR_ITERATIONS``, ``MAXIMISE_OBJECTIVE``,
        ``EPSILON``, ``OBJ_EVAL_ITERATION``, ``USE_TIME_LIMIT`` and
        ``TIME_LIMIT_SEC`` are at least necessary. See ``main.py`` for
        parameter explanations. Note that methods that are called within this
        method may require more parameters. Note that ``NR_ITERATIONS`` is set
        in the methods of ``run_algorithm.py``.

    Returns
    -------
    Y_grad : np.ndarray
        The gradient estimate at the last iteration.
    current_theta : np.ndarray
        The matrix in the unconstrained space at the last iteration.
    current_transition_matrix : np.ndarray
        The transition matrix in the stochastic matrix space at the last
        iteration.
    current_stationary_distribution : np.ndarray
        The stationary distribution of all nodes at the last iteration.
    objective_values : np.ndarray
        The objective value at each iteration or at each ``OBJ_EVAL_ITERATION``
        th iteration if ``DECR_NR_OBJ_EVAL=True``. Note that the last objective
        value is the objective value of the last ``OBJ_EVAL_ITERATION`` th
        iteration if ``USE_TIME_LIMIT=True`` and the optimisation has not
        finished before the time limit has passed, i.e., the objective value
        is not calculated at the specific iteration the time limit has passed
        if that iteration is not divisible by ``OBJ_EVAL_ITERATION``.

    """

    if ALGORITHM_PARAMETERS["DECR_NR_OBJ_EVAL"]:
        objective_values = np.zeros(
            int(
                ALGORITHM_PARAMETERS["NR_ITERATIONS"]
                / ALGORITHM_PARAMETERS["OBJ_EVAL_ITERATION"]
                + 1
            )
        )
    else:
        objective_values = np.zeros(ALGORITHM_PARAMETERS["NR_ITERATIONS"] + 1)

    objective_values[0] = calculate_objective_value(
        stationary_distribution(M), M, ALGORITHM_PARAMETERS
    )

    # Inverse transformation
    current_theta = inverse_logistic_transformation_matrix(
        M, ALGORITHM_PARAMETERS
    )

    obj_counter = 1
    for i in range(1, ALGORITHM_PARAMETERS["NR_ITERATIONS"] + 1):

        if i % 50000 == 0:
            print(f"At {datetime.datetime.now()} at iteration {i}.")

        Y_grad = SPSA_gradient(
            current_theta, M, C, rng, i, ALGORITHM_PARAMETERS
        )

        if ALGORITHM_PARAMETERS["MAXIMISE_OBJECTIVE"]:
            new_theta = (
                current_theta + ALGORITHM_PARAMETERS["EPSILON"] * Y_grad
            )
        else:
            new_theta = (
                current_theta - ALGORITHM_PARAMETERS["EPSILON"] * Y_grad
            )

        if ALGORITHM_PARAMETERS["DECR_NR_OBJ_EVAL"]:
            # Calculate objective value every OBJ_EVAL_ITERATION iteration.
            if i % ALGORITHM_PARAMETERS["OBJ_EVAL_ITERATION"] == 0:
                obj_transition_matrix = logistic_simplex_transformation(
                    new_theta, M, C
                )
                objective_values[obj_counter] = calculate_objective_value(
                    stationary_distribution(obj_transition_matrix),
                    obj_transition_matrix,
                    ALGORITHM_PARAMETERS,
                )
                obj_counter += 1
        else:
            # Calculate objective value every iteration.
            obj_transition_matrix = logistic_simplex_transformation(
                new_theta, M, C
            )
            objective_values[i] = calculate_objective_value(
                stationary_distribution(obj_transition_matrix),
                obj_transition_matrix,
                ALGORITHM_PARAMETERS,
            )

        current_theta = new_theta

        if ALGORITHM_PARAMETERS["USE_TIME_LIMIT"]:
            if (
                datetime.datetime.now() - start_time_algorithm
            ).total_seconds() > ALGORITHM_PARAMETERS["TIME_LIMIT_SEC"]:
                print(
                    "The time limit has passed. Break from algorithm "
                    f"for-loop. i={i}. obj_counter={obj_counter}."
                )
                if ALGORITHM_PARAMETERS["DECR_NR_OBJ_EVAL"]:
                    objective_values = objective_values[:obj_counter]
                else:
                    objective_values = objective_values[: i + 1]
                break

    current_transition_matrix = logistic_simplex_transformation(
        current_theta, M, C
    )
    current_stationary_distribution = stationary_distribution(
        current_transition_matrix
    )

    return (
        Y_grad,
        current_theta,
        current_transition_matrix,
        current_stationary_distribution,
        objective_values,
    )
