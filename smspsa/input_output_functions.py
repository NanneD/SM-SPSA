#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from typing import Any
from pytest import approx
from pathlib import Path
from algorithm_functions import stationary_distribution
from plot_functions import (
    plot_matrix,
    plot_network,
    plot_objective,
    plot_stat_dist,
)


def print_parameters(
    ALGORITHM_PARAMETERS: dict[str, Any], M: np.ndarray, C: np.ndarray
) -> None:
    """
    Prints the algorithm parameters.

    Parameters
    ----------
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters.
    M : np.ndarray
        The initial transition matrix that is optimised.
    C : np.ndarray
        The binary adjustment matrix.

    """

    print("The used parameters are:")

    for key, value in ALGORITHM_PARAMETERS.items():
        if key == "M_FILE_NAME":
            print(f"{key} is: {value}")
            print("The M matrix is: \n", M)
            print(
                "The starting stationary distribution is: ",
                stationary_distribution(M),
            )
        elif key == "C_FILE_NAME":
            print(f"{key} is: {value}")
            print("The C matrix is: \n", C)
        else:
            print(f"{key} is: {value}")

    print("\n")


def save_input_parameters(ALGORITHM_PARAMETERS: dict[str, Any]) -> None:
    """
    Saves the algorithm parameters in a text file.

    The text file is saved in directory ``OUTPUT_DIRECTORY`` with name
    ``RUN_PARAMETERS_FILE_NAME``.

    Parameters
    ----------
    SIMULATION_PARAMETERS : dict[str, Any]
        The simulation parameters. The parameters ``OUTPUT_DIRECTORY``
        and ``RUN_PARAMETERS_FILE_NAME`` are at least necessary. See
        ``main.py`` for parameter explanations.

    """

    with open(
        f"{ALGORITHM_PARAMETERS['OUTPUT_DIRECTORY']}"
        f"{ALGORITHM_PARAMETERS['RUN_PARAMETERS_FILE_NAME']}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for key, value in ALGORITHM_PARAMETERS.items():
            f.write(f"{key}: {value}\n")

        f.close()


def check_input_parameters(ALGORITHM_PARAMETERS: dict[str, Any]):
    """
    Checks the algorithm parameters for invalid input parameters.

    Parameters
    ----------
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters.

    Raises
    ------
    KeyError
        If the parameter used in a check is not present in the
        ``ALGORITHM_PARAMETERS`` dictionary.

    Exception
        If invalid input parameters are detected.

    """

    if ALGORITHM_PARAMETERS["SEED_VALUE"] < 0:
        raise Exception(
            "The value of SEED_VALUE should be larger than 0, "
            f'but is {ALGORITHM_PARAMETERS["SEED_VALUE"]}.'
        )

    if ALGORITHM_PARAMETERS["SEED_LAYOUT"] < 0:
        raise Exception(
            "The value of SEED_LAYOUT should be "
            "larger than 0 but is "
            f'{ALGORITHM_PARAMETERS["SEED_LAYOUT"]}.'
        )

    if (
        ALGORITHM_PARAMETERS["SEED_VALUE"]
        == ALGORITHM_PARAMETERS["SEED_LAYOUT"]
    ):
        raise Exception(
            "The values of SEED_VALUE and SEED_LAYOUT "
            "are equal. These should be different."
        )

    if not (
        ALGORITHM_PARAMETERS["CRN_GENERATOR"] == "RandomState"
        or ALGORITHM_PARAMETERS["CRN_GENERATOR"] == "Generator"
    ):
        raise Exception(
            "The value of 'CRN_GENERATOR' is incorrect. "
            "It should be 'RandomState' or 'Generator'."
        )

    if not (
        ALGORITHM_PARAMETERS["EPSILON"] >= 0
        and ALGORITHM_PARAMETERS["EPSILON"] <= 1
    ):
        raise Exception(
            "The EPSILON value should be between 0 and 1 "
            f', but is {ALGORITHM_PARAMETERS["EPSILON"]}.'
        )

    if not (
        ALGORITHM_PARAMETERS["ALGORITHM_RUN_TYPE"] == "single"
        or ALGORITHM_PARAMETERS["ALGORITHM_RUN_TYPE"] == "multiple"
    ):
        raise Exception(
            "Wrong ALGORITHM_RUN_TYPE specified. Please either "
            'use "single" or "multiple".'
        )

    if not (
        ALGORITHM_PARAMETERS["INVERSE_ZERO"] > 0
        and ALGORITHM_PARAMETERS["INVERSE_ZERO"] <= 1e-1
    ):
        raise Exception(
            "For good algorithm performance, the INVERSE_ZERO "
            "value should at least be between 0 and 1e-1 "
            f'but is {ALGORITHM_PARAMETERS["INVERSE_ZERO"]}.'
        )

    if not Path(ALGORITHM_PARAMETERS["MATRICES_DIRECTORY"]).exists():
        raise Exception(
            "The directory "
            f'{ALGORITHM_PARAMETERS["MATRICES_DIRECTORY"]} '
            "does not exist. Please specify a "
            "different MATRICES_DIRECTORY."
        )

    if not (
        ALGORITHM_PARAMETERS["GAME_TYPE"] == "hyperlink"
        or ALGORITHM_PARAMETERS["GAME_TYPE"] == "regular"
    ):
        raise Exception(
            "The GAME_TYPE should be hyperlink or regular, "
            f'but is {ALGORITHM_PARAMETERS["GAME_TYPE"]}.'
        )

    if (
        ALGORITHM_PARAMETERS["GAME_TYPE"] == "hyperlink"
        and not ALGORITHM_PARAMETERS["MAXIMISE_OBJECTIVE"]
    ):
        raise Exception(
            "The GAME_TYPE is hyperlink, so the objective should "
            "be maximised, but it is minimised. Please change "
            "MAXIMISE_OBJECTIVE."
        )

    if (
        "hyperlink" in ALGORITHM_PARAMETERS["INSTANCE_NAME"]
        and not ALGORITHM_PARAMETERS["GAME_TYPE"] == "hyperlink"
    ):
        raise Exception(
            "The INSTANCE_NAME contains 'hyperlink', but the "
            "GAME_TYPE is not 'hyperlink'."
        )

    if not Path(ALGORITHM_PARAMETERS["OUTPUT_DIRECTORY"]).exists():
        raise Exception(
            "The directory "
            f'{ALGORITHM_PARAMETERS["OUTPUT_DIRECTORY"]} does not '
            "exist. Please specify a different OUTPUT_DIRECTORY."
        )

    if (
        ALGORITHM_PARAMETERS["PLOT_NETWORK"]
        and ALGORITHM_PARAMETERS["NETWORK_PLOT_FILE_NAME"] == ""
    ):
        raise Exception(
            "The PLOT_NETWORK value is True, but the "
            "NETWORK_PLOT_FILE_NAME "
            f'({ALGORITHM_PARAMETERS["NETWORK_PLOT_FILE_NAME"]}) '
            "is an empty string."
        )

    if (
        ALGORITHM_PARAMETERS["PLOT_NETWORK"]
        and len(ALGORITHM_PARAMETERS["GRAPH_COLOURS"]) != 2
    ):
        raise Exception(
            "Two GRAPH_COLOURS should be specified, "
            f'but {len(ALGORITHM_PARAMETERS["GRAPH_COLOURS"])} '
            "GRAPH_COLOURS were provided."
        )

    if ALGORITHM_PARAMETERS["EDGE_FACTOR"] <= 0:
        raise Exception(
            "The EDGE_FACTOR should be larger than 0, but is "
            f"{ALGORITHM_PARAMETERS['EDGE_FACTOR']}."
        )

    if (
        ALGORITHM_PARAMETERS["MEMORY_OPTIMISED"]
        and ALGORITHM_PARAMETERS["PLOT_MATRICES"]
    ):
        raise Exception(
            "The MEMORY_OPTIMISED and PLOT_MATRICES are both "
            "True, but the MEMORY_OPTIMISED option does not plot "
            "matrices. Please make PLOT_MATRICES False."
        )

    if not (
        ALGORITHM_PARAMETERS["HEURISTIC_TYPE"] == "no_heuristic"
        or ALGORITHM_PARAMETERS["HEURISTIC_TYPE"] == "centred_mass"
    ):
        raise Exception(
            "The HEURISTIC_TYPE should be 'no_heuristic' or "
            "'centred_mass', but it is not. Please change this."
        )

    if (
        ALGORITHM_PARAMETERS["HEURISTIC_TYPE"] == "no_heuristic"
        and ALGORITHM_PARAMETERS["CENTRED_MASS_START"]
    ):
        raise Exception(
            "The HEURISTIC_TYPE is 'no_heuristic', but the "
            "heuristic boolean CENTRED_MASS_START is True."
            "Please change this."
        )

    if (
        ALGORITHM_PARAMETERS["HEURISTIC_TYPE"] == "centred_mass"
        and not ALGORITHM_PARAMETERS["CENTRED_MASS_START"]
    ):
        raise Exception(
            "The HEURISTIC_TYPE is 'centred_mass', but "
            "CENTRED_MASS_START is False. Please make it True."
        )

    if (
        not ALGORITHM_PARAMETERS["USE_TIME_LIMIT"]
        and ALGORITHM_PARAMETERS["TIME_LIMIT_SEC"] is not None
    ):
        raise Exception(
            "The USE_TIME_LIMIT is False, so TIME_LIMIT_SEC should"
            " be None, but it is not. Please change it to None."
        )

    if (
        ALGORITHM_PARAMETERS["USE_TIME_LIMIT"]
        and ALGORITHM_PARAMETERS["TIME_LIMIT_SEC"] is None
    ):
        raise Exception(
            "The USE_TIME_LIMIT is True, so TIME_LIMIT_SEC should"
            " have a value, but it is None. Please change this."
        )

    if (
        ALGORITHM_PARAMETERS["USE_TIME_LIMIT"]
        and ALGORITHM_PARAMETERS["TIME_LIMIT_SEC"] <= 0
    ):
        raise Exception(
            "The USE_TIME_LIMIT is True, so TIME_LIMIT_SEC should "
            "be greater than 0, but it is smaller. "
            "Please change this."
        )

    if (
        ALGORITHM_PARAMETERS["DECR_NR_OBJ_EVAL"]
        and not ALGORITHM_PARAMETERS["MEMORY_OPTIMISED"]
    ):
        raise Exception(
            "DECR_NR_OBJ_EVAL can only be True if MEMORY_OPTIMISED"
            " is True. Please change one of the two parameters."
        )

    if (
        not ALGORITHM_PARAMETERS["DECR_NR_OBJ_EVAL"]
        and ALGORITHM_PARAMETERS["OBJ_EVAL_ITERATION"] is not None
    ):
        raise Exception(
            "DECR_NR_OBJ_EVAL is False, but OBJ_EVAL_ITERATION "
            "is not None. Please make it None."
        )

    if (
        ALGORITHM_PARAMETERS["DECR_NR_OBJ_EVAL"]
        and ALGORITHM_PARAMETERS["OBJ_EVAL_ITERATION"] is None
    ):
        raise Exception(
            "DECR_NR_OBJ_EVAL is True, but OBJ_EVAL_ITERATION "
            "is None. Please make it not None."
        )

    if (
        ALGORITHM_PARAMETERS["DECR_NR_OBJ_EVAL"]
        and ALGORITHM_PARAMETERS["OBJ_EVAL_ITERATION"] <= 0
    ):
        raise Exception(
            "DECR_NR_OBJ_EVAL is True, so OBJ_EVAL_ITERATION "
            "should be greater than 0, but it is smaller. "
            "Please change this."
        )

    if (
        ALGORITHM_PARAMETERS["DECR_NR_OBJ_EVAL"]
        and ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] != "fixed"
    ):
        raise Exception(
            "DECR_NR_OBJ_EVAL is True, so NR_ITERATIONS_TYPE "
            "should be 'fixed', but it is not. "
            "Please change one of the two."
        )

    if (
        ALGORITHM_PARAMETERS["DECR_NR_OBJ_EVAL"]
        and ALGORITHM_PARAMETERS["NR_ITERATIONS_FIXED"]
        % ALGORITHM_PARAMETERS["OBJ_EVAL_ITERATION"]
        != 0
    ):
        raise Exception(
            "The NR_ITERATIONS_FIXED should be divisible by "
            "OBJ_EVAL_ITERATIONS, but it is not. "
            "Please change it."
        )

    if Path(
        ALGORITHM_PARAMETERS["OUTPUT_DIRECTORY"]
        + ALGORITHM_PARAMETERS["RUN_PARAMETERS_FILE_NAME"]
        + ".txt"
    ).exists():
        raise Exception(
            "The RUN_PARAMETERS_FILE_NAME already exists and "
            "will be overwritten."
        )

    if Path(
        ALGORITHM_PARAMETERS["OUTPUT_DIRECTORY"]
        + ALGORITHM_PARAMETERS["RUNNING_TIME_FILE_NAME"]
        + ".csv"
    ).exists():
        raise Exception(
            "The RUNNING_TIME_FILE_NAME already exists and "
            "will be overwritten."
        )

    if ALGORITHM_PARAMETERS["ALGORITHM_RUN_TYPE"] == "single":

        if ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] != "fixed":
            raise Exception(
                "The single algorithm run only works if "
                "NR_ITERATIONS_TYPE is 'fixed'. Please change it."
            )

        if ALGORITHM_PARAMETERS["NR_ITERATIONS_FIXED"] is None:
            raise Exception(
                "The value of NR_ITERATIONS_FIXED should have a "
                "value, but is None. Please change this."
            )

        if ALGORITHM_PARAMETERS["NR_ITERATIONS_FIXED"] <= 0:
            raise Exception(
                "The value of NR_ITERATIONS_FIXED should be "
                "larger than 0 but is "
                f'{ALGORITHM_PARAMETERS["NR_ITERATIONS_FIXED"]}.'
            )

        if ALGORITHM_PARAMETERS["NR_ITERATIONS_FACTOR"] is not None:
            raise Exception(
                "The value of NR_ITERATIONS_FACTOR should be "
                "None, but it is not."
            )

        if ALGORITHM_PARAMETERS["INSTANCE_PARAMETERS_FILE_NAME"] is not None:
            raise Exception(
                "The ALGORITHM_RUN_TYPE is 'single'. Please make "
                "the 'INSTANCE_PARAMETERS_FILE_NAME' None."
            )

        if ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"] is None:
            raise Exception(
                "The ALGORITHM_RUN_TYPE is 'single'. Please make "
                "'OBJECTIVE_INDICES' not None."
            )

        if not Path(
            ALGORITHM_PARAMETERS["MATRICES_DIRECTORY"]
            + ALGORITHM_PARAMETERS["M_FILE_NAME"]
            + ".csv"
        ).exists():
            raise Exception(
                "The M_FILE_NAME "
                f'({ALGORITHM_PARAMETERS["M_FILE_NAME"]}) '
                "points to a non-existing file."
            )

        if not Path(
            ALGORITHM_PARAMETERS["MATRICES_DIRECTORY"]
            + ALGORITHM_PARAMETERS["C_FILE_NAME"]
            + ".csv"
        ).exists():
            raise Exception(
                "The C_FILE_NAME "
                f'({ALGORITHM_PARAMETERS["C_FILE_NAME"]}) '
                "points to a non-existing file."
            )

        if (
            pd.read_csv(
                f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
                f"{ALGORITHM_PARAMETERS['C_FILE_NAME']}.csv"
            ).dtypes
            != np.dtype("int64")
        ).any():
            raise Exception(
                "The C matrix contains entries that are not ints. " "Error."
            )

        C = pd.read_csv(
            f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
            f"{ALGORITHM_PARAMETERS['C_FILE_NAME']}"
            ".csv"
        ).to_numpy(dtype=int)

        if C.shape[0] != C.shape[1]:
            raise Exception("The C matrix is not a square matrix. Error.")

        if np.all(C == 0):
            raise Exception(
                "All C values are zero. "
                "This will not result in any optimisation "
                "being done."
            )

        if np.all(np.sum(C, axis=1) == 1):
            raise Exception(
                "Every row of the C matrix only contains one 1. "
                "This will not result in any optimisation "
                "being done."
            )

        if np.all((np.sum(C, axis=1) == 1) | (np.sum(C, axis=1) == 0)):
            raise Exception(
                "Every row of the C matrix only contains one 1 or "
                "only zeroes. This will not result in any "
                "optimisation being done."
            )

        if len(ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]) == 0:
            raise Exception(
                "The OBJECTIVE_INDICES "
                f'({ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]}) '
                "does not contain an objective index. "
                "Please add one."
            )

        if (
            ALGORITHM_PARAMETERS["GAME_TYPE"] == "regular"
            and not len(ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]) == 1
        ):
            raise Exception(
                "The GAME_TYPE is regular, but the number of "
                "OBJECTIVE_INDICES "
                f'({ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]}) '
                "is not equal to 1."
            )

        if (
            ALGORITHM_PARAMETERS["GAME_TYPE"] == "hyperlink"
            and not len(ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]) == 1
        ):
            raise Exception(
                "The GAME_TYPE is hyperlink, but the number of "
                "OBJECTIVE_INDICES "
                f'({ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]}) '
                "is not equal to 1."
            )

        if (
            pd.read_csv(
                f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
                f"{ALGORITHM_PARAMETERS['M_FILE_NAME']}.csv",
                float_precision="round_trip",
            ).dtypes
            != np.dtype("float64")
        ).any():
            raise Exception(
                "The M matrix contains entries that are not float." " Error."
            )

        M = pd.read_csv(
            f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
            f"{ALGORITHM_PARAMETERS['M_FILE_NAME']}"
            ".csv",
            float_precision="round_trip",
        ).to_numpy(dtype=float)

        if M.shape[0] != M.shape[1]:
            raise Exception("The M matrix is not a square matrix. Error.")

        if M.shape != C.shape:
            raise Exception("The M and C matrices do not have the same shape.")

        if not all(
            i in np.arange(M.shape[0])
            for i in ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]
        ):
            raise Exception(
                "The OBJECTIVE_INDICES parameter "
                f'({ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]}) '
                "contains indices that are not part "
                "of the network."
            )

    if ALGORITHM_PARAMETERS["ALGORITHM_RUN_TYPE"] == "multiple":

        if not (
            ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] == "fixed"
            or ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] == "variable"
        ):
            raise Exception(
                "The parameter NR_ITERATIONS_TYPE should have "
                "value 'fixed' or 'variable'."
            )

        if (
            ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] == "fixed"
            and ALGORITHM_PARAMETERS["NR_ITERATIONS_FACTOR"] is not None
        ):
            raise Exception(
                "The NR_ITERATIONS_TYPE is 'fixed', so the "
                "NR_ITERATIONS_FACTOR should be None, "
                "but it is not."
            )

        if (
            ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] == "fixed"
            and ALGORITHM_PARAMETERS["NR_ITERATIONS_FIXED"] is None
        ):
            raise Exception(
                "The NR_ITERATIONS_TYPE is 'fixed', so the "
                "NR_ITERATIONS_FIXED should have a value, "
                "but it is None."
            )

        if (
            ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] == "fixed"
            and ALGORITHM_PARAMETERS["NR_ITERATIONS_FIXED"] <= 0
        ):
            raise Exception(
                "The NR_ITERATIONS_TYPE is 'fixed', so the "
                "NR_ITERATIONS_FIXED should be "
                "larger than 0, but is "
                f"{ALGORITHM_PARAMETERS['NR_ITERATIONS_FIXED']}."
            )

        if (
            ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] == "variable"
            and ALGORITHM_PARAMETERS["NR_ITERATIONS_FACTOR"] is None
        ):
            raise Exception(
                "The NR_ITERATIONS_TYPE is 'variable', so the "
                "NR_ITERATIONS_FACTOR should have a value, "
                "but it is None."
            )

        if (
            ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] == "variable"
            and ALGORITHM_PARAMETERS["NR_ITERATIONS_FIXED"] is not None
        ):
            raise Exception(
                "The NR_ITERATIONS_TYPE is 'variable', so the "
                "NR_ITERATIONS_FIXED should be None, "
                "but it is not."
            )

        if (
            ALGORITHM_PARAMETERS["NR_ITERATIONS_TYPE"] == "variable"
            and ALGORITHM_PARAMETERS["NR_ITERATIONS_FACTOR"] <= 0
        ):
            raise Exception(
                "The NR_ITERATIONS_TYPE is 'variable', so the "
                "NR_ITERATIONS_FACTOR should be "
                "larger than 0, but is "
                f"{ALGORITHM_PARAMETERS['NR_ITERATIONS_FACTOR']}."
            )

        if ALGORITHM_PARAMETERS["INSTANCE_PARAMETERS_FILE_NAME"] is None:
            raise Exception(
                "The ALGORITHM_RUN_TYPE is 'multiple'. Please "
                "specify 'INSTANCE_PARAMETERS_FILE_NAME'."
            )

        if not Path(
            ALGORITHM_PARAMETERS["MATRICES_DIRECTORY"]
            + ALGORITHM_PARAMETERS["INSTANCE_PARAMETERS_FILE_NAME"]
            + ".csv"
        ).exists():
            raise Exception(
                "The INSTANCE_PARAMETERS_FILE_NAME "
                f"({ALGORITHM_PARAMETERS['INSTANCE_PARAMETERS_FILE_NAME']})"
                " points to a non-existing file."
            )

        if ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"] is not None:
            raise Exception(
                "The ALGORITHM_RUN_TYPE is 'multiple'. "
                "Please make the 'OBJECTIVE_INDICES' "
                "parameter None."
            )


def check_input_run_multiple_instances(
    ALGORITHM_PARAMETERS: dict[str, Any], instance_nr: int
) -> None:
    """
    Checks the algorithm parameters for invalid input parameters if multiple
    instances are run.

    Parameters
    ----------
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters.
    instance_nr : int
        The instance number.

    Raises
    ------
    KeyError
        If the parameter used in a check is not present in the
        ``ALGORITHM_PARAMETERS`` dictionary.

    Exception
        If invalid input parameters are detected.

    """

    if not Path(
        ALGORITHM_PARAMETERS["MATRICES_DIRECTORY"]
        + ALGORITHM_PARAMETERS["M_FILE_NAME"]
        + "_"
        + str(instance_nr)
        + ".csv"
    ).exists():
        raise Exception(
            "The M_FILE_NAME "
            f"({ALGORITHM_PARAMETERS['M_FILE_NAME']+'_'+str(instance_nr)}) "
            "points to a non-existing file."
        )

    if not Path(
        ALGORITHM_PARAMETERS["MATRICES_DIRECTORY"]
        + ALGORITHM_PARAMETERS["C_FILE_NAME"]
        + "_"
        + str(instance_nr)
        + ".csv"
    ).exists():
        raise Exception(
            "The C_FILE_NAME "
            f"({ALGORITHM_PARAMETERS['C_FILE_NAME']+'_'+str(instance_nr)}) "
            "points to a non-existing file."
        )

    if (
        pd.read_csv(
            f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
            f"{ALGORITHM_PARAMETERS['C_FILE_NAME']}"
            f"_{str(instance_nr)}.csv"
        ).dtypes
        != np.dtype("int64")
    ).any():
        raise Exception(
            "The C matrix contains entries that are not ints. " "Error."
        )

    C = pd.read_csv(
        f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
        f"{ALGORITHM_PARAMETERS['C_FILE_NAME']}_{str(instance_nr)}.csv"
    ).to_numpy(dtype=int)

    if C.shape[0] != C.shape[1]:
        raise Exception("The C matrix is not a square matrix. Error.")

    if np.all(C == 0):
        raise Exception(
            "All C values are zero. This will not result in any optimisation "
            "being done."
        )

    if np.all(np.sum(C, axis=1) == 1):
        raise Exception(
            "Every row of the C matrix only contains one 1. "
            "This will not result in any optimisation "
            "being done."
        )

    if np.all((np.sum(C, axis=1) == 1) | (np.sum(C, axis=1) == 0)):
        raise Exception(
            "Every row of the C matrix only contains one 1 or "
            "only zeroes. This will not result in any "
            "optimisation being done."
        )

    if len(ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]) == 0:
        raise Exception(
            "The OBJECTIVE_INDICES parameter "
            f"({ALGORITHM_PARAMETERS['OBJECTIVE_INDICES']}) does not contain "
            "an objective index. Please add one."
        )

    if (
        ALGORITHM_PARAMETERS["GAME_TYPE"] == "regular"
        and not len(ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]) == 1
    ):
        raise Exception(
            "The GAME_TYPE is regular, but the number of OBJECTIVE_INDICES "
            f"({ALGORITHM_PARAMETERS['OBJECTIVE_INDICES']}) is not equal to 1."
        )

    if (
        ALGORITHM_PARAMETERS["GAME_TYPE"] == "hyperlink"
        and not len(ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]) == 1
    ):
        raise Exception(
            "The GAME_TYPE is hyperlink, but the number of "
            "OBJECTIVE_INDICES "
            f'({ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]}) '
            "is not equal to 1."
        )

    if (
        pd.read_csv(
            f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
            f"{ALGORITHM_PARAMETERS['M_FILE_NAME']}"
            f"_{str(instance_nr)}.csv",
            float_precision="round_trip",
        ).dtypes
        != np.dtype("float64")
    ).any():
        raise Exception(
            "The M matrix contains entries that are not float." " Error."
        )

    M = pd.read_csv(
        f"{ALGORITHM_PARAMETERS['MATRICES_DIRECTORY']}"
        f"{ALGORITHM_PARAMETERS['M_FILE_NAME']}_{str(instance_nr)}.csv",
        float_precision="round_trip",
    ).to_numpy(dtype=float)

    if M.shape[0] != M.shape[1]:
        raise Exception("The M matrix is not a square matrix. Error.")

    if M.shape != C.shape:
        raise Exception("The M and C matrices do not have the same shape.")

    if not all(
        i in np.arange(M.shape[0])
        for i in ALGORITHM_PARAMETERS["OBJECTIVE_INDICES"]
    ):
        raise Exception(
            "The OBJECTIVE_INDICES parameter "
            f"({ALGORITHM_PARAMETERS['OBJECTIVE_INDICES']}) contains indices "
            "that are not part of the network."
        )


def save_and_plot_results_memory_optimised(
    M: np.ndarray,
    C: np.ndarray,
    final_stationary_distribution: np.ndarray,
    final_transition_matrix: np.ndarray,
    final_theta: np.ndarray,
    objective_values: np.ndarray,
    final_gradient_estimate: np.ndarray,
    ALGORITHM_PARAMETERS: dict[str, Any],
) -> None:
    """
    Saves and creates the plots and output matrices for the memory optimised SM-SPSA.

    Parameters
    ----------
    M : np.ndarray
        The transition matrix that is optimised.
    C : np.ndarray
        The binary adjustment matrix.
    final_stationary_distribution : np.ndarray
        The stationary distribution at the last iteration.
    final_transition_matrix : np.ndarray
        The transition matrix at the last iteration.
    final_theta : np.ndarray
        The matrix in the unconstrained space at the last iteration.
    objective_values : np.ndarray
        The objective value at each iteration or at each
        ``OBJ_EVAL_ITERATION`` th iteration if ``DECR_NR_OBJ_EVAL=True``.
    final_gradient_estimate : np.ndarray
        The gradient estimate at the last iteration.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameters ``PLOT_OBJECTIVE``,
        ``PLOT_NETWORK``, ``NETWORK_PLOT_FILE_NAME``, ``SAVE_OUTPUT``,
        ``INSTANCE_OUTPUT_DIRECTORY`` and ``CURRENT_INSTANCE_NAME`` are at
        least necessary. See ``main.py`` for parameter explanations. Note that
        ``INSTANCE_OUTPUT_DIRECTORY`` and ``CURRENT_INSTANCE_NAME`` are set in
        the methods of ``run_algorithm.py``.

    """

    if ALGORITHM_PARAMETERS["PLOT_OBJECTIVE"]:
        plot_objective(objective_values, ALGORITHM_PARAMETERS)

    if ALGORITHM_PARAMETERS["PLOT_NETWORK"]:
        plot_network(
            M,
            final_transition_matrix,
            C,
            ALGORITHM_PARAMETERS["NETWORK_PLOT_FILE_NAME"],
            ALGORITHM_PARAMETERS,
        )

    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        pd.DataFrame(final_stationary_distribution).to_csv(
            f"{ALGORITHM_PARAMETERS['INSTANCE_OUTPUT_DIRECTORY']}"
            "Final_stationary_distribution_"
            f"{ALGORITHM_PARAMETERS['CURRENT_INSTANCE_NAME']}.csv",
            index=False,
        )
        pd.DataFrame(final_transition_matrix).to_csv(
            f"{ALGORITHM_PARAMETERS['INSTANCE_OUTPUT_DIRECTORY']}"
            "Final_transition_matrix_"
            f"{ALGORITHM_PARAMETERS['CURRENT_INSTANCE_NAME']}.csv",
            index=False,
        )
        pd.DataFrame(final_theta).to_csv(
            f"{ALGORITHM_PARAMETERS['INSTANCE_OUTPUT_DIRECTORY']}"
            "Final_theta_"
            f"{ALGORITHM_PARAMETERS['CURRENT_INSTANCE_NAME']}.csv",
            index=False,
        )
        pd.DataFrame(objective_values, columns=["Objective_value"]).to_csv(
            f"{ALGORITHM_PARAMETERS['INSTANCE_OUTPUT_DIRECTORY']}"
            "Objective_values_"
            f"{ALGORITHM_PARAMETERS['CURRENT_INSTANCE_NAME']}.csv",
            index=False,
        )
        pd.DataFrame(final_gradient_estimate).to_csv(
            f"{ALGORITHM_PARAMETERS['INSTANCE_OUTPUT_DIRECTORY']}"
            "Final_gradient_estimate_"
            f"{ALGORITHM_PARAMETERS['CURRENT_INSTANCE_NAME']}.csv",
            index=False,
        )


def save_and_plot_results(
    M: np.ndarray,
    C: np.ndarray,
    stationary_distributions: np.ndarray,
    transition_matrices: np.ndarray,
    thetas: np.ndarray,
    objective_values: np.ndarray,
    gradient_estimates: np.ndarray,
    ALGORITHM_PARAMETERS: dict[str, Any],
) -> None:
    """
    Saves and creates the plots and output matrices for SM-SPSA.

    Parameters
    ----------
    M : np.ndarray
        The transition matrix that is optimised.
    C : np.ndarray
        The binary adjustment matrix.
    stationary_distributions : np.ndarray
        The stationary distribution of all nodes at each iteration.
    transition_matrices : np.ndarray
        The transition matrix in the stochastic matrix space at each
        iteration.
    thetas : np.ndarray
        The matrix in the unconstrained space at each iteration.
    objective_values : np.ndarray
        The objective value at each iteration.
    gradient_estimates : np.ndarray
        The gradient estimate at each iteration.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameters ``PLOT_OBJECTIVE``,
        ``PLOT_MATRICES``, ``PLOT_NETWORK``,
        ``NETWORK_PLOT_FILE_NAME``, ``SAVE_OUTPUT``,
        ``INSTANCE_OUTPUT_DIRECTORY`` and ``CURRENT_INSTANCE_NAME`` are at
        least necessary. See ``main.py`` for parameter explanations. Note that
        ``INSTANCE_OUTPUT_DIRECTORY`` and ``CURRENT_INSTANCE_NAME`` are set in
        the methods of ``run_algorithm.py``.

    """

    USE_ROW_COLOURS = True if M.shape[0] <= 5 else False

    if ALGORITHM_PARAMETERS["PLOT_OBJECTIVE"]:
        plot_objective(objective_values, ALGORITHM_PARAMETERS)

    if ALGORITHM_PARAMETERS["PLOT_MATRICES"]:
        plot_matrix(
            Matrix=thetas,
            C=C,
            PLOT_TYPE="theta",
            USE_COLOURS=USE_ROW_COLOURS,
            ALGORITHM_PARAMETERS=ALGORITHM_PARAMETERS,
        )
        plot_matrix(
            Matrix=transition_matrices,
            C=C,
            PLOT_TYPE="transition",
            USE_COLOURS=USE_ROW_COLOURS,
            ALGORITHM_PARAMETERS=ALGORITHM_PARAMETERS,
        )
        plot_stat_dist(
            stationary_distributions=stationary_distributions,
            USE_COLOURS=USE_ROW_COLOURS,
            ALGORITHM_PARAMETERS=ALGORITHM_PARAMETERS,
        )
        plot_matrix(
            Matrix=gradient_estimates,
            C=C,
            PLOT_TYPE="gradient",
            USE_COLOURS=USE_ROW_COLOURS,
            ALGORITHM_PARAMETERS=ALGORITHM_PARAMETERS,
        )

    if ALGORITHM_PARAMETERS["PLOT_NETWORK"]:
        plot_network(
            M,
            transition_matrices[-1],
            C,
            ALGORITHM_PARAMETERS["NETWORK_PLOT_FILE_NAME"],
            ALGORITHM_PARAMETERS,
        )

    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        pd.DataFrame(
            transition_matrices.reshape(
                transition_matrices.shape[0],
                transition_matrices.shape[1] * transition_matrices.shape[2],
            )
        ).to_csv(
            f"{ALGORITHM_PARAMETERS['INSTANCE_OUTPUT_DIRECTORY']}"
            "Transition_matrices_"
            f"{ALGORITHM_PARAMETERS['CURRENT_INSTANCE_NAME']}.csv",
            index=False,
        )
        pd.DataFrame(stationary_distributions).to_csv(
            f"{ALGORITHM_PARAMETERS['INSTANCE_OUTPUT_DIRECTORY']}"
            "Stationary_distributions_"
            f"{ALGORITHM_PARAMETERS['CURRENT_INSTANCE_NAME']}.csv",
            index=False,
        )
        pd.DataFrame(objective_values, columns=["Objective_value"]).to_csv(
            f"{ALGORITHM_PARAMETERS['INSTANCE_OUTPUT_DIRECTORY']}"
            "Objective_values_"
            f"{ALGORITHM_PARAMETERS['CURRENT_INSTANCE_NAME']}.csv",
            index=False,
        )
        pd.DataFrame(
            thetas.reshape(thetas.shape[0], thetas.shape[1] * thetas.shape[2])
        ).to_csv(
            f"{ALGORITHM_PARAMETERS['INSTANCE_OUTPUT_DIRECTORY']}"
            "Thetas_"
            f"{ALGORITHM_PARAMETERS['CURRENT_INSTANCE_NAME']}.csv",
            index=False,
        )
        pd.DataFrame(
            gradient_estimates.reshape(
                gradient_estimates.shape[0],
                gradient_estimates.shape[1] * gradient_estimates.shape[2],
            )
        ).to_csv(
            f"{ALGORITHM_PARAMETERS['INSTANCE_OUTPUT_DIRECTORY']}"
            "Gradient_estimates_"
            f"{ALGORITHM_PARAMETERS['CURRENT_INSTANCE_NAME']}.csv",
            index=False,
        )


def check_stochasticity_input_matrix(
    initial_transition_matrix: np.ndarray,
) -> None:
    """
    Checks whether the initial transition matrix is stochastic or not.

    Parameters
    ----------
    initial_transition_matrix : np.ndarray
        The initial transition matrix.

    Raises
    ------
    Exception
        If it contains values larger than 1, smaller than 0 or the row sum is
        not equal to 1.

    """

    # All row sums should be equal to (approx) 1.
    for i in range(np.shape(initial_transition_matrix)[0]):
        if np.sum(initial_transition_matrix[i]) != approx(1, abs=1e-12):
            raise Exception(
                f"The row sum with index {i} is not equal to 1. "
                "The row sum is: "
                f"{np.sum(initial_transition_matrix[i])}. Error."
            )

    # All initial_transition_matrix entries are larger or equal to 0 and smaller or equal to 1.
    if not np.all(
        (initial_transition_matrix >= 0) & (initial_transition_matrix <= 1)
    ):
        raise Exception(
            "The initial_transition_matrix contains values lower "
            "than 0 or bigger than 1. Error."
        )


def check_final_output(
    final_transition_matrix: np.ndarray,
    final_stationary_distribution: np.ndarray,
) -> None:
    """
    Checks whether the optimised transition matrix and corresponding stationary
    distribution are valid.

    Parameters
    ----------
    final_transition_matrix : np.ndarray
        The optimised transition matrix.
    final_stationary_distribution : np.ndarray
        The optimised stationary distribution corresponding to the optimised
        transition matrix.

    Raises
    ------
    Exception
        1. if the sum of the optimised stationary distribution is not equal
        to 1.
        2. if the optimised stationary distribution contains values lower than
        0 or larger than 1.
        3. if the optimised transition matrix contains falues lower than 0 or
        larger than 1.
        4. if the optimised transition matrix has at least one row sum that
        is not equal to 1.

    """

    # Sum final_stationary_distribution is (approx) 1.
    if np.sum(final_stationary_distribution) != approx(1, abs=1e-12):
        raise Exception(
            "ERROR. The sum of the final_stationary_distribution is not 1. "
            "The sum of the final_stationary_distribution is equal to: "
            f"{np.sum(final_stationary_distribution)}."
        )

    # All final_stationary_distribution entries are larger or equal to 0 and smaller or equal to 1.
    if not np.all(
        (final_stationary_distribution >= 0)
        & (final_stationary_distribution <= 1)
    ):
        raise Exception(
            "ERROR. The final_stationary_distribution contains values "
            "lower than 0 or bigger than 1."
        )

    # All final_transition_matrix entries are larger or equal to 0 and smaller or equal to 1.
    if not np.all(
        (final_transition_matrix >= 0) & (final_transition_matrix <= 1)
    ):
        raise Exception(
            "ERROR. The final_transition_matrix contains values "
            "lower than 0 or bigger than 1."
        )

    # All row sums should be equal to (approx) 1.
    for i in range(np.shape(final_transition_matrix)[0]):
        if np.sum(final_transition_matrix[i]) != approx(1, abs=1e-12):
            raise Exception(
                f"Error. The row sum with index {i} not equal to 1. "
                f"The row sum is: {np.sum(final_transition_matrix[i])}."
            )
