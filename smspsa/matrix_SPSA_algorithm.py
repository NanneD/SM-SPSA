#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from typing import Any
from algorithm_functions import (
    stationary_distribution,
    logistic_simplex_transformation,
    calculate_objective_value,
)


def calculate_numerator_SPSA(
    high_stat: np.ndarray,
    high_transition_matrix: np.ndarray,
    low_stat: np.ndarray,
    low_transition_matrix: np.ndarray,
    ALGORITHM_PARAMETERS: dict[str, Any],
) -> float:
    """
    Calculates the numerator of the SPSA gradient.

    Parameters
    ----------
    high_stat : np.ndarray
        The stationary distribution of the plus perturbation.
    high_transition_matrix : np.ndarray
        The transition matrix of the plus perturbation.
    low_stat : np.ndarray
        The stationary distribution of the minus perturbation.
    low_transition_matrix : np.ndarray
        The transition matrix of the minus perturbation.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. Methods that are called within this method
        require parameters. See these methods for explanations.

    Returns
    -------
    num : float
        The numerator of the SPSA gradient.

    """

    num = calculate_objective_value(
        high_stat, high_transition_matrix, ALGORITHM_PARAMETERS
    ) - calculate_objective_value(
        low_stat, low_transition_matrix, ALGORITHM_PARAMETERS
    )

    return num


def SPSA_gradient(
    theta: np.ndarray,
    initial_trans_matrix: np.ndarray,
    C: np.ndarray,
    rng: np.random.mtrand.RandomState | np.random._generator.Generator,
    i: int,
    ALGORITHM_PARAMETERS: dict[str, Any],
) -> np.ndarray:
    """
    Calculates the SPSA gradient.

    Note that the elements where ``C=0`` have value 0.

    Parameters
    ----------
    theta : np.ndarray
        The matrix in the unconstrained space that is optimised.
    initial_trans_matrix : np.ndarray
        The transition matrix that is optimised.
    C : np.ndarray
        The binary adjustment matrix.
    rng : np.random.mtrand.RandomState|np.random._generator.Generator
        A pseudo-random number generator.
    i : int
        The iteration number.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. Methods that are called within this method
        require parameters. See these methods for explanations.

    Returns
    -------
    Y : np.ndarray
        The SPSA gradient.

    """

    delta_matrix = rng.choice((-1, 1), size=(theta.shape))
    eta = 1 / i

    perturbation_matrix = np.multiply(eta * delta_matrix, C)

    high_perturbation = theta + perturbation_matrix
    low_perturbation = theta - perturbation_matrix

    high_perturbation_transformation = logistic_simplex_transformation(
        high_perturbation, initial_trans_matrix, C
    )
    low_perturbation_transformation = logistic_simplex_transformation(
        low_perturbation, initial_trans_matrix, C
    )

    high_stat = stationary_distribution(high_perturbation_transformation)
    low_stat = stationary_distribution(low_perturbation_transformation)

    numerator = calculate_numerator_SPSA(
        high_stat,
        high_perturbation_transformation,
        low_stat,
        low_perturbation_transformation,
        ALGORITHM_PARAMETERS,
    )
    denominator = 2 * eta * delta_matrix

    Y = np.multiply(numerator / denominator, C)

    return Y
