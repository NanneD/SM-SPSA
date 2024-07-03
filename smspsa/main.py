#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main interface of the SM-SPSA algorithm.

You can set all SM-SPSA parameters with this script and then run the
algorithm. Below, all parameters are discussed. The input data section on the
SM-SPSA website also contains more information on the required input data.

Parameters
----------
SEED_VALUE : int
    The seed value of the pseudo-random number generator for generating the
    random perturbations. It cannot be the same as ``SEED_LAYOUT``.
SEED_LAYOUT : int
    The seed value for the layout of a network plot. It cannot be the same as
    ``SEED_VALUE``.
CRN_GENERATOR : str
    The type of pseudo-random number generator that should be used. It should
    be either "Generator" for using NumPy's default or "RandomState" for
    NumPy's legacy generator.
MATRICES_DIRECTORY : str
    The folder, relative to the ``ROOT_DIRECTORY`` (automatically determined),
    where the input matrices are located.
OUTPUT_DIRECTORY : str
    The folder, relative to the ``ROOT_DIRECTORY`` (automatically determined),
    where the algorithm output should be saved if ``SAVE_OUTPUT=True``.
ALGORITHM_RUN_TYPE : str
    Whether you want to run a single instance ("single") or multiple instances
    ("multiple") with the algorithm.
HEURISTIC_TYPE : str
    Either "no_heuristic" if you do not want to use an heuristic or
    "centred_mass" if you want to use the centred mass heuristic. This variable
    is also used to define ``RUN_PARAMETERS_FILE_NAME``,
    ``RUNNING_TIME_FILE_NAME`` and ``NETWORK_PLOT_FILE_NAME``.
INSTANCE_NAME : str
    The name of the network instance that you want to optimise. If
    ``ALGORITHM_RUN_TYPE="multiple"``, a suffix "_i" (where i is the instance
    number), is automatically added when reading the data. This to facilitate
    running the algorithm consecutively on multiple instances. Note that the
    actual filenames should contain this suffix. Thus, the first instance has
    suffix "_0", the second "_1", etc. This variable is also used to define
    ``M_FILE_NAME``, ``C_FILE_NAME``, ``RUN_PARAMETERS_FILE_NAME``,
    ``RUNNING_TIME_FILE_NAME`` and ``NETWORK_PLOT_FILE_NAME``.
M_FILE_NAME : str
    The name of the file that contains the initial transition matrix that is
    optimised. If ``ALGORITHM_RUN_TYPE="multiple"``, it should not contain
    the suffix "_i" (where i is the instance number), since this is
    automatically added when reading the data.
C_FILE_NAME : str
    The name of the file that contains the binary adjustment matrix containing
    0/1 indicating whether element [i,j] can be optimised (1) or not (0). If
    ``ALGORITHM_RUN_TYPE="multiple"``, it should not contain the suffix "_i",
    (where i is the instance number) since this is automatically added when
    reading the data.
INSTANCE_PARAMETERS_FILE_NAME : str | None
    The name of the file that contains the instance parameter table if
    ``ALGORITHM_RUN_TYPE="multiple"``. Otherwise, it should be ``None``. All
    networks that are present in this table will be optimised.
RUN_PARAMETERS_FILE_NAME : str
    The name of the file where the run parameters will be saved if
    ``SAVE_OUTPUT=True``.
RUNNING_TIME_FILE_NAME : str
    The name of the file where the running times will be saved if
    ``SAVE_OUTPUT=True``.
CENTRED_MASS_START : bool
    Whether the centred mass start is used (``True``) or not (``False``). It
    is automatically set based on the value of ``HEURISTIC_TYPE``.
MAXIMISE_OBJECTIVE : bool
    Whether the objective should be maximised (``True``) or minimised
    (``False``).
GAME_TYPE : str
    The game type of the instance. Either "regular" (optimising a single node)
    or "hyperlink" (optimising a hyperlink objective function).
OBJECTIVE_INDICES : list[int] | None
    A list that contains the node that will be optimised if
    ``ALGORITHM_RUN_TYPE="single"``. Otherwise, it should be ``None``.
MEMORY_OPTIMISED : bool
    Whether the memory optimised version of the algorithm is used (``True``)
    or not (``False``). The main difference is that the output matrices are not
    stored for each iteration, instead only the matrices of the last iteration
    are stored. This version is especially useful for large networks.
DECR_NR_OBJ_EVAL : bool
    Whether the number of objective value evaluations should be decreased
    (``True``) or not (``False``). This option is only available if
    ``MEMORY_OPTIMISED=True``.
OBJ_EVAL_ITERATION : int | None
    If ``DECR_NR_OBJ_EVAL=True``, the objective value is calculated every
    ``OBJ_EVAL_ITERATION``th iteration starting at iteration 0. Otherwise, it
    should be ``None``.
NR_ITERATIONS_TYPE : str
    Whether a fixed number of algoritihm iterations ("fixed") or a variable
    number of iterations depending on the instance size ("variable") should be
    used. If ``NR_ITERATIONS_TYPE="variable"``, the number of iterations is set
    to the size of the transition matrix squared times``NR_ITERATIONS_FACTOR``.
    If ``ALGORITHM_RUN_TYPE="single"``, only the "fixed" version can be used.
NR_ITERATIONS_FIXED : int | None
    If ``NR_ITERATIONS_TYPE="fixed"``, it specifies the number of algorithm
    iterations that are performed. Otherwise, it should be ``None``.
NR_ITERATIONS_FACTOR : int | None
    If ``NR_ITERATIONS_TYPE="variable"``, the number of iterations is set
    to the size of the transition matrix squared times``NR_ITERATIONS_FACTOR``.
    Otherwise, it should be ``None``.
EPSILON : float
    The value of the constant gain size.
INVERSE_ZERO : float
    If the initial transition matrix contains zeroes or ones, these values
    are replaced by ``INVERSE_ZERO`` and ``1-INVERSE_ZERO``, respectively, to
    ensure numerical feasibility.
USE_TIME_LIMIT : bool
    Whether a time limit should be used (``True``) or not (``False``) for
    running the algorithm.
TIME_LIMIT_SEC : float | None
    If ``USE_TIME_LIMIT=True``, it specifies the running time of the algorithm
    in seconds. Otherwise, it should be ``None``.
NETWORK_PLOT_FILE_NAME : str
    The name of the file where the network plots will be saved if
    ``PlOT_NETWORK=True`` and ``SAVE_OUTPUT=True``. Note that a suffix
    "_start" and "_final" will be added for the network plot of the initial
    and the optimised transition matrix, respectively.
GRAPH_COLOURS : list[str]
    A list with the colours that are used for the network plot. The first
    colour is for edges that cannot be adjusted, the second for edges that can
    be adjusted.
COLOUR_MAP : matplotlib.colors.LinearSegmentedColormap
    The colour map that should be used to color the nodes of the network plot
    according to their stationary distribution value.
EDGE_FACTOR : int
    The factor with which the weights of the edges should be multiplied
    for determining the edge width of the network plot.
PLOT_NETWORK
    Whether the network plots should be created (``True``) or not (``False``).
    A plot is created of the initial and the optimised network.
PLOT_MATRICES
    Whether the various matrix plots should be created (``True``) or not
    (``False``). This option is only available if ``MEMORY_OPTIMISED=False``.
PLOT_OBJECTIVE
    Whether the objective value should be plotted (``True``) or not
    (``False``).
SAVE_OUTPUT
    Whether the output should be saved (``True``) or not (``False``).
"""

import os
import datetime
import matplotlib.pyplot as plt

from typing import Any
from run_algorithm import select_algorithm_run_type

print(f"The code starts running at {datetime.datetime.now()}")
###################################Seed########################################
SEED_VALUE: int = 0
SEED_LAYOUT: int = 10
CRN_GENERATOR: str = "Generator"  # Generator or RandomState
################################Directories####################################
ROOT_DIRECTORY: str = os.path.dirname(os.path.dirname(__file__))
MATRICES_DIRECTORY: str = os.path.join(
    ROOT_DIRECTORY, "data/example_networks/"
)
OUTPUT_DIRECTORY: str = os.path.join(ROOT_DIRECTORY, "results/")
##############################Algorithm run type###############################
ALGORITHM_RUN_TYPE: str = "multiple"
HEURISTIC_TYPE = "centred_mass"  #'no_heuristic', 'centred_mass'
CENTRED_MASS_START: bool = True if HEURISTIC_TYPE == "centred_mass" else False
#################################File names####################################
INSTANCE_NAME: str = "example_network"
M_FILE_NAME: str = f"M_matrix_{INSTANCE_NAME}"
C_FILE_NAME: str = f"C_matrix_{INSTANCE_NAME}"
INSTANCE_PARAMETERS_FILE_NAME: str | None = (
    f"Instance_Parameters_{INSTANCE_NAME}"
)
RUN_PARAMETERS_FILE_NAME: str = (
    f"Run_main_input_parameters_{INSTANCE_NAME}_{HEURISTIC_TYPE}"
)
RUNNING_TIME_FILE_NAME: str = f"Running_times_{INSTANCE_NAME}_{HEURISTIC_TYPE}"
###############################Game Parameters#################################
MAXIMISE_OBJECTIVE: bool = True
GAME_TYPE: str = "regular"  # hyperlink, regular
OBJECTIVE_INDICES: list[int] | None = None
#############################Algorithm Parameters##############################
MEMORY_OPTIMISED: bool = False
DECR_NR_OBJ_EVAL: bool = False
OBJ_EVAL_ITERATION: None | int = None
NR_ITERATIONS_TYPE: str = "fixed"  #'fixed' or 'variable'
NR_ITERATIONS_FIXED: None | int = 100000  # int if 'fixed', None if 'variable'.
NR_ITERATIONS_FACTOR: None | int = None  # None if 'fixed', int if 'variable'.
EPSILON: float = 1e-1
INVERSE_ZERO: float = 1e-3
USE_TIME_LIMIT: bool = False
TIME_LIMIT_SEC: None | float = None
##############################Plot Parameters##################################
NETWORK_PLOT_FILE_NAME: str = f"Network_plot_{INSTANCE_NAME}_{HEURISTIC_TYPE}"
GRAPH_COLOURS: list[str] = ["grey", "#d95f02"]
COLOUR_MAP = plt.cm.Blues
EDGE_FACTOR: int = 2
PLOT_NETWORK: bool = True
PLOT_MATRICES: bool = False
PLOT_OBJECTIVE: bool = True
##############################Output Parameters################################
SAVE_OUTPUT: bool = True
##############################Initialization###################################
ALGORITHM_PARAMETERS: dict[str, Any] = {
    "SEED_VALUE": SEED_VALUE,
    "SEED_LAYOUT": SEED_LAYOUT,
    "CRN_GENERATOR": CRN_GENERATOR,
    "NR_ITERATIONS_TYPE": NR_ITERATIONS_TYPE,
    "NR_ITERATIONS_FIXED": NR_ITERATIONS_FIXED,
    "NR_ITERATIONS_FACTOR": NR_ITERATIONS_FACTOR,
    "MEMORY_OPTIMISED": MEMORY_OPTIMISED,
    "EPSILON": EPSILON,
    "ALGORITHM_RUN_TYPE": ALGORITHM_RUN_TYPE,
    "INSTANCE_NAME": INSTANCE_NAME,
    "M_FILE_NAME": M_FILE_NAME,
    "C_FILE_NAME": C_FILE_NAME,
    "INSTANCE_PARAMETERS_FILE_NAME": INSTANCE_PARAMETERS_FILE_NAME,
    "RUN_PARAMETERS_FILE_NAME": RUN_PARAMETERS_FILE_NAME,
    "RUNNING_TIME_FILE_NAME": RUNNING_TIME_FILE_NAME,
    "INVERSE_ZERO": INVERSE_ZERO,
    "MAXIMISE_OBJECTIVE": MAXIMISE_OBJECTIVE,
    "HEURISTIC_TYPE": HEURISTIC_TYPE,
    "CENTRED_MASS_START": CENTRED_MASS_START,
    "GAME_TYPE": GAME_TYPE,
    "OBJECTIVE_INDICES": OBJECTIVE_INDICES,
    "MATRICES_DIRECTORY": MATRICES_DIRECTORY,
    "OUTPUT_DIRECTORY": OUTPUT_DIRECTORY,
    "PLOT_NETWORK": PLOT_NETWORK,
    "NETWORK_PLOT_FILE_NAME": NETWORK_PLOT_FILE_NAME,
    "GRAPH_COLOURS": GRAPH_COLOURS,
    "EDGE_FACTOR": EDGE_FACTOR,
    "PLOT_MATRICES": PLOT_MATRICES,
    "PLOT_OBJECTIVE": PLOT_OBJECTIVE,
    "SAVE_OUTPUT": SAVE_OUTPUT,
    "USE_TIME_LIMIT": USE_TIME_LIMIT,
    "TIME_LIMIT_SEC": TIME_LIMIT_SEC,
    "DECR_NR_OBJ_EVAL": DECR_NR_OBJ_EVAL,
    "OBJ_EVAL_ITERATION": OBJ_EVAL_ITERATION,
    "COLOUR_MAP": COLOUR_MAP,
}
################################Run algorithm##################################
if __name__ == "__main__":
    select_algorithm_run_type(ALGORITHM_PARAMETERS)
