#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import datetime
import numpy as np
import pandas as pd

from typing import Any
from input_output_functions import (
    print_parameters,
    save_and_plot_results,
    check_final_output,
    check_input_parameters,
    check_stochasticity_input_matrix,
    save_and_plot_results_memory_optimised,
    save_input_parameters,
    check_input_run_multiple_instances,
)

from matrix_algorithms import (
    run_matrix_algorithm,
    run_matrix_algorithm_improved_memory,
)
from algorithm_functions import generate_centred_mass_initial_matrix


def select_algorithm_run_type(ALGORITHM_PARAMETERS: dict[str, Any]) -> None:
    """
    Calls the single instance or multiple instances code based on the
    ``ALGORITHM_RUN_TYPE`` variable.

    Parameters
    ----------
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameter ``ALGORITHM_RUN_TYPE`` is at
        least necessary. See ``main.py`` for parameter explanations. Note that
        methods that are called within this method may require more parameters.

    Raises
    ------
    Exception
        If an invalid ``ALGORITHM_RUN_TYPE`` is selected (not "single" or
        "multiple").

    """

    if ALGORITHM_PARAMETERS["ALGORITHM_RUN_TYPE"] == "single":
        run_one_instance(ALGORITHM_PARAMETERS)
    elif ALGORITHM_PARAMETERS["ALGORITHM_RUN_TYPE"] == "multiple":
        run_multiple_instances(ALGORITHM_PARAMETERS)
    else:
        raise Exception(
            "Invalid ALGORITHM_RUN_TYPE specified. Please either use "
            '"single" or "multiple".'
        )


def run_one_instance(ALGORITHM_PARAMETERS: dict[str, Any]) -> None:
    """
    Runs the SM-SPSA algorithm on one instance.

    Includes additional code for checking and saving input parameters,
    applying heuristics, setting additional required variables,
    plotting and saving results.

    Parameters
    ----------
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameters ``SAVE_OUTPUT``,
        ``MATRICES_DIRECTORY``, ``M_FILE_NAME``, ``C_FILE_NAME``,
        ``OUTPUT_DIRECTORY``, ``INSTANCE_NAME``, ``HEURISTIC_TYPE``
        ``NR_ITERATIONS_TYPE``, ``NR_ITERATIONS_FIXED``, ``CRN_GENERATOR``,
        ``CENTRED_MASS_START``, ``SEED_VALUE``, ``MEMORY_OPTIMISED`` and
        ``RUNNING_TIME_FILE_NAME`` are at least necessary. See ``main.py``
        for parameter explanations. Note that methods that are called within
        this method may require more parameters.

    Raises
    ------
    Exception
        1. If ``NR_ITERATIONS_TYPE="fixed"``.
        2. If an invalid value for ``CRN_GENERATOR`` is specified (not
        "RandomState" or "Generator").
    """

    start_time = datetime.datetime.now()

    check_input_parameters(ALGORITHM_PARAMETERS)
    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        save_input_parameters(ALGORITHM_PARAMETERS)

    M = pd.read_csv(
        f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
        f"{ALGORITHM_PARAMETERS['M_FILE_NAME']}.csv",
        float_precision="round_trip",
    ).to_numpy(dtype=float)
    C = pd.read_csv(
        f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
        f"{ALGORITHM_PARAMETERS['C_FILE_NAME']}.csv"
    ).to_numpy(dtype=int)
    check_stochasticity_input_matrix(M)

    ALGORITHM_PARAMETERS["INSTANCE_OUTPUT_DIRECTORY"] = ALGORITHM_PARAMETERS[
        "OUTPUT_DIRECTORY"
    ]
    ALGORITHM_PARAMETERS["CURRENT_INSTANCE_NAME"] = (
        ALGORITHM_PARAMETERS["INSTANCE_NAME"]
        + "_"
        + ALGORITHM_PARAMETERS["HEURISTIC_TYPE"]
    )

    if ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] != "fixed":
        raise Exception(
            "The NR_ITERATIONS_TYPE should be 'fixed', but it is not."
        )
    ALGORITHM_PARAMETERS["NR_ITERATIONS"] = ALGORITHM_PARAMETERS[
        "NR_ITERATIONS_FIXED"
    ]

    print_parameters(ALGORITHM_PARAMETERS, M, C)

    start_time_algorithm = datetime.datetime.now()
    if ALGORITHM_PARAMETERS["CENTRED_MASS_START"]:
        run_M = generate_centred_mass_initial_matrix(M, C)
    else:
        run_M = copy.deepcopy(M)

    rng: np.random._generator.Generator | np.random.mtrand.RandomState
    if ALGORITHM_PARAMETERS["CRN_GENERATOR"] == "RandomState":
        rng = np.random.RandomState(ALGORITHM_PARAMETERS["SEED_VALUE"])
    elif ALGORITHM_PARAMETERS["CRN_GENERATOR"] == "Generator":
        rng = np.random.default_rng(ALGORITHM_PARAMETERS["SEED_VALUE"])
    else:
        raise Exception(
            "Wrong value specified for CRN_GENERATOR. Please use "
            "'RandomState' or 'Generator'."
        )

    if ALGORITHM_PARAMETERS["MEMORY_OPTIMISED"]:
        (
            final_gradient_estimate,
            final_theta,
            final_transition_matrix,
            final_stationary_distribution,
            objective_values,
        ) = run_matrix_algorithm_improved_memory(
            run_M, C, start_time_algorithm, rng, ALGORITHM_PARAMETERS
        )
        running_time_algorithm = (
            datetime.datetime.now() - start_time_algorithm
        ).total_seconds()
        print(f"The algorithm running time is: {running_time_algorithm}")
        check_final_output(
            final_transition_matrix, final_stationary_distribution
        )
        save_and_plot_results_memory_optimised(
            run_M,
            C,
            final_stationary_distribution,
            final_transition_matrix,
            final_theta,
            objective_values,
            final_gradient_estimate,
            ALGORITHM_PARAMETERS,
        )
    else:
        (
            gradient_estimates,
            thetas,
            transition_matrices,
            stationary_distributions,
            objective_values,
        ) = run_matrix_algorithm(
            run_M, C, start_time_algorithm, rng, ALGORITHM_PARAMETERS
        )
        running_time_algorithm = (
            datetime.datetime.now() - start_time_algorithm
        ).total_seconds()
        print(f"The algorithm running time is: {running_time_algorithm}")
        check_final_output(
            transition_matrices[-1], stationary_distributions[-1]
        )
        save_and_plot_results(
            run_M,
            C,
            stationary_distributions,
            transition_matrices,
            thetas,
            objective_values,
            gradient_estimates,
            ALGORITHM_PARAMETERS,
        )

        final_transition_matrix = transition_matrices[-1]
        final_stationary_distribution = stationary_distributions[-1]

    print("The final transition matrix \n", final_transition_matrix)
    print("The stationary distribution: ", final_stationary_distribution)

    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        pd.DataFrame(
            {
                "Instance_name": ALGORITHM_PARAMETERS["INSTANCE_NAME"],
                "Running_time (sec)": running_time_algorithm,
            },
            index=[0],
        ).to_csv(
            f"{ALGORITHM_PARAMETERS['OUTPUT_DIRECTORY']}"
            f"{ALGORITHM_PARAMETERS['RUNNING_TIME_FILE_NAME']}.csv",
            index=False,
        )

    print(f"The total running time is: {datetime.datetime.now()-start_time}")


def run_multiple_instances(ALGORITHM_PARAMETERS: dict[str, Any]) -> None:
    """
    Runs the SM-SPSA algorithm on multiple instances.

    Includes additional code for checking and saving input parameters,
    applying heuristics, setting additional required variables,
    plotting and saving results.

    Parameters
    ----------
    ALGORITHM_PARAMETERS : dict[str,Any]
        The algorithm parameters. The parameters ``SAVE_OUTPUT``,
        ``NETWORK_PLOT_FILE_NAME``, ``MATRICES_DIRECTORY``,
        ``INSTANCE_PARAMETERS_FILE_NAME``, ``INSTANCE_NAME``,
        ``HEURISTIC_TYPE``, ``OUTPUT_DIRECTORY``, ``M_FILE_NAME``,
        ``C_FILE_NAME``, ``NR_ITERATIONS_TYPE``, ``NR_ITERATIONS_FIXED``,
        ``NR_ITERATIONS_FACTOR``, ``CRN_GENERATOR``,
        ``CENTRED_MASS_START``, ``SEED_VALUE``, ``MEMORY_OPTIMISED`` and
        ``RUNNING_TIME_FILE_NAME`` are at least necessary. See ``main.py``
        for parameter explanations. Note that methods that are called within
        this method may require more parameters.

    Raises
    ------
    Exception
        (1) If an invalid ``NR_ITERATIONS_TYPE`` is provided (not "fixed" or
        "variable").
        (2) If an invalid value for ``CRN_GENERATOR`` is specified (not
        "RandomState" or "Generator").

    """

    start_time = datetime.datetime.now()

    check_input_parameters(ALGORITHM_PARAMETERS)
    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        save_input_parameters(ALGORITHM_PARAMETERS)

    NETWORK_PLOT_FILE_NAME = ALGORITHM_PARAMETERS["NETWORK_PLOT_FILE_NAME"]

    INSTANCE_PARAMETERS = pd.read_csv(
        f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
        f"{ALGORITHM_PARAMETERS['INSTANCE_PARAMETERS_FILE_NAME']}.csv"
    )
    NR_INSTANCES = INSTANCE_PARAMETERS.shape[0]
    running_times = np.zeros(NR_INSTANCES)

    for instance_nr in range(NR_INSTANCES):
        CURRENT_INSTANCE_NAME = (
            ALGORITHM_PARAMETERS["INSTANCE_NAME"]
            + "_"
            + ALGORITHM_PARAMETERS["HEURISTIC_TYPE"]
            + "_"
            + str(instance_nr)
        )

        ALGORITHM_PARAMETERS["INSTANCE_OUTPUT_DIRECTORY"] = (
            f"{ALGORITHM_PARAMETERS['OUTPUT_DIRECTORY']}"
            f"{CURRENT_INSTANCE_NAME}/"
        )
        if not os.path.exists(
            ALGORITHM_PARAMETERS["INSTANCE_OUTPUT_DIRECTORY"]
        ):
            os.makedirs(ALGORITHM_PARAMETERS["INSTANCE_OUTPUT_DIRECTORY"])

        ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"] = [
            INSTANCE_PARAMETERS.loc[instance_nr, "Objective_index"]
        ]
        ALGORITHM_PARAMETERS["CURRENT_INSTANCE_NAME"] = CURRENT_INSTANCE_NAME
        ALGORITHM_PARAMETERS["NETWORK_PLOT_FILE_NAME"] = (
            NETWORK_PLOT_FILE_NAME + "_" + str(instance_nr)
        )

        check_input_run_multiple_instances(ALGORITHM_PARAMETERS, instance_nr)

        M = pd.read_csv(
            f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
            f"{ALGORITHM_PARAMETERS['M_FILE_NAME']}_{str(instance_nr)}.csv",
            float_precision="round_trip",
        ).to_numpy(dtype=float)
        C = pd.read_csv(
            f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
            f"{ALGORITHM_PARAMETERS['C_FILE_NAME']}_{str(instance_nr)}.csv"
        ).to_numpy(dtype=int)

        if ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] == "fixed":
            ALGORITHM_PARAMETERS["NR_ITERATIONS"] = ALGORITHM_PARAMETERS[
                "NR_ITERATIONS_FIXED"
            ]
        elif ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] == "variable":
            ALGORITHM_PARAMETERS["NR_ITERATIONS"] = (
                M.shape[0]
                * M.shape[0]
                * ALGORITHM_PARAMETERS["NR_ITERATIONS_FACTOR"]
            )
        else:
            raise Exception(
                "The NR_ITERATIONS_TYPE should be 'fixed' or"
                "'variable', but it is not."
            )

        check_stochasticity_input_matrix(M)

        start_time_algorithm = datetime.datetime.now()
        if ALGORITHM_PARAMETERS["CENTRED_MASS_START"]:
            run_M = generate_centred_mass_initial_matrix(M, C)
        else:
            run_M = copy.deepcopy(M)

        rng: np.random._generator.Generator | np.random.mtrand.RandomState
        if ALGORITHM_PARAMETERS["CRN_GENERATOR"] == "RandomState":
            rng = np.random.RandomState(ALGORITHM_PARAMETERS["SEED_VALUE"])
        elif ALGORITHM_PARAMETERS["CRN_GENERATOR"] == "Generator":
            rng = np.random.default_rng(ALGORITHM_PARAMETERS["SEED_VALUE"])
        else:
            raise Exception(
                "Wrong value specified for CRN_GENERATOR. Please use "
                "'RandomState' or 'Generator'."
            )

        if ALGORITHM_PARAMETERS["MEMORY_OPTIMISED"]:
            (
                final_gradient_estimate,
                final_theta,
                final_transition_matrix,
                final_stationary_distribution,
                objective_values,
            ) = run_matrix_algorithm_improved_memory(
                run_M, C, start_time_algorithm, rng, ALGORITHM_PARAMETERS
            )
            running_time_algorithm = (
                datetime.datetime.now() - start_time_algorithm
            ).total_seconds()
            print(f"The algorithm running time is: {running_time_algorithm}")
            check_final_output(
                final_transition_matrix, final_stationary_distribution
            )
            save_and_plot_results_memory_optimised(
                run_M,
                C,
                final_stationary_distribution,
                final_transition_matrix,
                final_theta,
                objective_values,
                final_gradient_estimate,
                ALGORITHM_PARAMETERS,
            )
        else:
            (
                gradient_estimates,
                thetas,
                transition_matrices,
                stationary_distributions,
                objective_values,
            ) = run_matrix_algorithm(
                run_M, C, start_time_algorithm, rng, ALGORITHM_PARAMETERS
            )
            running_time_algorithm = (
                datetime.datetime.now() - start_time_algorithm
            ).total_seconds()
            print(f"The algorithm running time is: {running_time_algorithm}")
            check_final_output(
                transition_matrices[-1], stationary_distributions[-1]
            )
            save_and_plot_results(
                run_M,
                C,
                stationary_distributions,
                transition_matrices,
                thetas,
                objective_values,
                gradient_estimates,
                ALGORITHM_PARAMETERS,
            )

            final_transition_matrix = transition_matrices[-1]
            final_stationary_distribution = stationary_distributions[-1]

        running_times[instance_nr] = running_time_algorithm

        print("The final transition matrix \n", final_transition_matrix)
        print("The stationary distribution: ", final_stationary_distribution)

    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        pd.DataFrame(
            {
                "Instance_name": INSTANCE_PARAMETERS["Instance_name"],
                "Running_time (sec)": running_times,
            }
        ).to_csv(
            f"{ALGORITHM_PARAMETERS['OUTPUT_DIRECTORY']}"
            f"{ALGORITHM_PARAMETERS['RUNNING_TIME_FILE_NAME']}.csv",
            index=False,
        )

    print(f"The total running time is: {datetime.datetime.now()-start_time}")
