#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from typing import Any
from algorithm_functions import stationary_distribution


def plot_objective(
    objective_values: np.ndarray, ALGORITHM_PARAMETERS: dict[str, Any]
) -> None:
    """
    Plots the objective value.

    The objective value at each iteration or at each ``OBJ_EVAL_ITERATION`` th
    iteration if ``DECR_NR_OBJ_EVAL=True`` is plotted.

    Parameters
    ----------
    objective_values : np.ndarray
        The objective value at each iteration.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameters ``DECR_NR_OBJ_EVAL``,
        ``OBJ_EVAL_ITERATION``, ``SAVE_OUTPUT``, ``INSTANCE_OUTPUT_DIRECTORY``
        and ``CURRENT_INSTANCE_NAME`` are at least necessary. See ``main.py``
        for parameter explanations. Note that ``INSTANCE_OUTPUT_DIRECTORY``
        and ``CURRENT_INSTANCE_NAME`` are set in the methods of
        ``run_algorithm.py``.

    """

    fig = plt.figure(figsize=(12, 8))

    plt.plot(objective_values, color="darkblue")

    plt.title(r"Objective function value per iteration $i$", fontsize=14)
    plt.xlabel(r"Iteration $i$", fontsize=14)
    plt.ylabel(r"Objective function value", fontsize=14)

    if ALGORITHM_PARAMETERS["DECR_NR_OBJ_EVAL"]:
        locs, cur_labels = plt.xticks()  # Current locs and labels.
        locs = locs[1:-1]  # Remove first and last ticks.
        new_labels = (
            locs * ALGORITHM_PARAMETERS["OBJ_EVAL_ITERATION"]
        )  # Scale labels.
        new_labels = [
            int(label) for label in new_labels
        ]  # Cast labels to int.
        plt.xticks(locs, new_labels)

    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        plt.savefig(
            ALGORITHM_PARAMETERS["INSTANCE_OUTPUT_DIRECTORY"]
            + ALGORITHM_PARAMETERS["CURRENT_INSTANCE_NAME"]
            + "_objective_function"
            + ".pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


def plot_stat_dist(
    stationary_distributions: np.ndarray,
    USE_COLOURS: bool,
    ALGORITHM_PARAMETERS: dict[str, Any],
) -> None:
    """
    Plots the stationary distribution at each iteration.

    Parameters
    ----------
    stationary_distributions : np.ndarray
        The stationary distribution of all nodes at each iteration.
    USE_COLOURS : bool
        Whether specific colours that make it easier to recognize the nodes
        should be used (``True``) or not (``False``). Only possible if
        networks are used of at most size 5x5.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameters ``SAVE_OUTPUT``,
        ``INSTANCE_OUTPUT_DIRECTORY`` and ``CURRENT_INSTANCE_NAME`` are at
        least necessary. See ``main.py`` for parameter explanations. Note
        that ``INSTANCE_OUTPUT_DIRECTORY`` and ``CURRENT_INSTANCE_NAME`` are
        set in the methods of ``run_algorithm.py``.

    """

    fig = plt.figure(figsize=(12, 8))
    nr_nodes = stationary_distributions.shape[1]

    names = [r"$\pi^{{(i)}}_{}$".format(i) for i in range(0, nr_nodes)]
    colours = {0: "orange", 1: "blue", 2: "green", 3: "red", 4: "grey"}

    for i in range(nr_nodes):
        if USE_COLOURS:
            plt.plot(
                stationary_distributions[:, i],
                color=colours[i],
                label=names[i],
            )
        else:
            plt.plot(stationary_distributions[:, i], label=names[i])

    plt.title(
        r"Stationary distribution of each node $m$ per iteration $i$",
        fontsize=14,
    )
    plt.ylabel(r"$\pi^{(i)}_m$ value", fontsize=14)
    plt.xlabel(r"Iteration $i$", fontsize=14)
    plt.legend(bbox_to_anchor=(1, 1), ncol=4, fontsize=12)

    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        plt.savefig(
            ALGORITHM_PARAMETERS["INSTANCE_OUTPUT_DIRECTORY"]
            + ALGORITHM_PARAMETERS["CURRENT_INSTANCE_NAME"]
            + "_stat_dist"
            + ".pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


def plot_matrix(
    Matrix: np.ndarray,
    C: np.ndarray,
    PLOT_TYPE: str,
    USE_COLOURS: bool,
    ALGORITHM_PARAMETERS: dict[str, Any],
) -> None:
    """
    Plots all adjustable elements of a matrix at each iteration.

    Either the "theta" matrices (the matrices in the unconstrained space), the
    "transition" matrices (the transition matrices in the stochastic matrix
    space) or the "gradient" matrices (the estimated gradient) can be plotted.

    Parameters
    ----------
    Matrix : np.ndarray
        The matrix that is plotted.
    C : np.ndarray
        The binary adjustment matrix.
    PLOT_TYPE : str
        Either "theta", "transition" or "gradient", depending on the type of
        matrix that is plotted.
    USE_COLOURS : bool
        Whether specific colours that make it easier to recognize the nodes
        should be used (``True``) or not (``False``). Only possible if
        networks are used of at most size 5x5.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameters ``SAVE_OUTPUT``,
        ``INSTANCE_OUTPUT_DIRECTORY`` and ``CURRENT_INSTANCE_NAME`` are at
        least necessary. See ``main.py`` for parameter explanations. Note
        that ``INSTANCE_OUTPUT_DIRECTORY`` and ``CURRENT_INSTANCE_NAME`` are
        set in the methods of ``run_algorithm.py``.

    Raises
    ------
    Exception
        If an invalid ``PLOT_TYPE`` is provided (not "theta", "transition"
        or "gradient").

    """

    fig = plt.figure(figsize=(12, 8))
    nr_nodes = Matrix.shape[1]

    if PLOT_TYPE == "theta":
        names = [
            r"$\Theta^{{(i)}}_{{{}{}}}$".format(i, j)
            for i in range(0, nr_nodes)
            for j in range(0, nr_nodes)
        ]
        x = np.arange(0, Matrix.shape[0])
    elif PLOT_TYPE == "transition":
        names = [
            r"$P^{{(i)}}_{{{}{}}}$".format(i, j)
            for i in range(0, nr_nodes)
            for j in range(0, nr_nodes)
        ]
        x = np.arange(0, Matrix.shape[0])
    elif PLOT_TYPE == "gradient":
        names = [
            r"$G^{{(i)}}_{{{}{}}}$".format(i, j)
            for i in range(0, nr_nodes)
            for j in range(0, nr_nodes)
        ]
        x = np.arange(1, Matrix.shape[0] + 1)
    else:
        raise Exception(
            "Wrong PLOT_TYPE specified. Please use "
            "'theta', 'transition' or 'gradient' "
        )

    colours = {
        0: ["orange", "wheat", "peru", "tan", "darkorange", "bisque"],
        1: ["blue", "darkblue", "royalblue", "cornflowerblue", "slateblue"],
        2: ["green", "lime", "mediumspringgreen", "darkgreen", "aquamarine"],
        3: ["red", "salmon", "darkred", "coral", "indianred"],
        4: ["grey", "black", "lightgrey", "dimgrey", "darkgrey"],
    }

    flatten_matrix = Matrix.reshape(
        Matrix.shape[0], Matrix.shape[1] * Matrix.shape[2]
    ).T
    for i in range(flatten_matrix.shape[0]):
        if C[int(names[i][-4]), int(names[i][-3])] == 1:
            if USE_COLOURS:
                plt.plot(
                    x,
                    flatten_matrix[i, :],
                    label=names[i],
                    color=colours[int(names[i][-4])][int(names[i][-3])],
                )
            else:
                plt.plot(x, flatten_matrix[i, :], label=names[i])

    if PLOT_TYPE == "transition":
        plt.title(
            r"Adjustable transition probabilities $P^{(i)}_{mn}$ per iteration $i$",
            fontsize=14,
        )
        plt.ylabel(r"$P^{(i)}_{mn}$ value", fontsize=14)
    elif PLOT_TYPE == "theta":
        plt.title(
            r"Adjustable $\Theta^{(i)}_{mn}$ values per iteration $i$",
            fontsize=14,
        )
        plt.ylabel(r"$\Theta^{(i)}_{mn}$ value", fontsize=14)
    elif PLOT_TYPE == "gradient":
        plt.title(
            "Adjustable gradient estimates per iteration $i$", fontsize=14
        )
        plt.ylabel(r"$G^{(i)}_{mn}$ value", fontsize=14)
    else:
        raise Exception(
            "Wrong PLOT_TYPE specified. Please use "
            "'theta', 'transition' or 'gradient' "
        )

    plt.legend(bbox_to_anchor=(1, 1), ncol=4, fontsize=12)
    plt.xlabel(r"Iteration $i$", fontsize=14)

    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        plt.savefig(
            ALGORITHM_PARAMETERS["INSTANCE_OUTPUT_DIRECTORY"]
            + ALGORITHM_PARAMETERS["CURRENT_INSTANCE_NAME"]
            + f"_{PLOT_TYPE}"
            + ".pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


def plot_network(
    M_start: np.ndarray,
    M_final: np.ndarray,
    C: np.ndarray,
    plot_name: str,
    ALGORITHM_PARAMETERS: dict[str, Any],
) -> None:
    """
    Plots the network before and after the optimization.

    Parameters
    ----------
    M_start : np.ndarray
        The initial transition matrix.
    M_final : np.ndarray
        The optimised transition matrix.
    C : np.ndarray
        The binary adjustment matrix.
    plot_name : str
        The name of the network plot if it is saved. Note that a suffix
        "_start" and "_final" will be added for the network plot of the initial
        and the optimised transition matrix, respectively.
    ALGORITHM_PARAMETERS : dict[str, Any]
        The algorithm parameters. The parameters ``SEED_LAYOUT``,
        ``COLOUR_MAP``, ``EDGE_FACTOR``,  ``GRAPH_COLOURS``,  ``SAVE_OUTPUT``
        and ``INSTANCE_OUTPUT_DIRECTORY`` are at least necessary. See
        ``main.py`` for parameter explanations. Note that
        ``INSTANCE_OUTPUT_DIRECTORY`` and ``CURRENT_INSTANCE_NAME`` are
        set in the methods of ``run_algorithm.py``.

    """

    stat_dist_start = stationary_distribution(M_start)
    stat_dist_final = stationary_distribution(M_final)

    # VMAX is max_stat_value rounded up to nearest multiple of 0.05.
    max_stat_value = np.max([stat_dist_start, stat_dist_final])
    VMAX = math.ceil(max_stat_value / 0.05) * 0.05

    G_unordered_start = nx.DiGraph()
    G_unordered_final = nx.DiGraph()

    NR_NODES = M_start.shape[0]

    for i in range(NR_NODES):
        for j in range(NR_NODES):
            # Because of width parameter ifs not necessary, but speeds-up creating
            # network. Network lay-out is also clearer with if-statements.
            if M_start[i, j] > 0:
                G_unordered_start.add_edge(
                    i, j, weight=M_start[i, j], adjustable=C[i, j]
                )
            if M_final[i, j] > 0:
                G_unordered_final.add_edge(
                    i, j, weight=M_final[i, j], adjustable=C[i, j]
                )

    # To obtain the correct ordering of the nodes (i.e., 0, 1, 2, 3, ...)
    G_start = nx.DiGraph()
    G_start.add_nodes_from(sorted(G_unordered_start.nodes(data=True)))
    G_start.add_edges_from(G_unordered_start.edges(data=True))

    G_final = nx.DiGraph()
    G_final.add_nodes_from(sorted(G_unordered_final.nodes(data=True)))
    G_final.add_edges_from(G_unordered_final.edges(data=True))

    # k and iterations so that distance between nodes is larger.
    pos = nx.spring_layout(
        G_start,
        seed=ALGORITHM_PARAMETERS["SEED_LAYOUT"],
        k=0.2,
        iterations=300,
    )

    fig0, ax0 = plt.subplots(figsize=(7, 5.8))

    # Draw nodes
    nodes_start = nx.draw_networkx_nodes(
        G_start,
        pos,
        vmin=0,
        vmax=VMAX,
        node_color=stat_dist_start,
        edgecolors="grey",
        cmap=ALGORITHM_PARAMETERS["COLOUR_MAP"],
        node_size=150,
        ax=ax0,
    )

    # Draw edges
    for u, v, d in G_start.edges(data=True):
        nx.draw_networkx_edges(
            G_start,
            pos,
            edgelist=[(u, v)],
            node_size=150,  # node_size for positioning edges
            alpha=0.8,
            width=d["weight"] * ALGORITHM_PARAMETERS["EDGE_FACTOR"],
            edge_color=ALGORITHM_PARAMETERS["GRAPH_COLOURS"][d["adjustable"]],
            connectionstyle="arc3,rad=0.05",
            ax=ax0,
        )

    # Draw node labels
    nx.draw_networkx_labels(G_start, pos, font_size=7, ax=ax0)
    plt.colorbar(nodes_start, ax=ax0)

    ax0.set_title(
        "The start network with nodes coloured by "
        "their stationary distribution value",
        fontsize=14,
    )

    plt.tight_layout()

    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        plt.savefig(
            ALGORITHM_PARAMETERS["INSTANCE_OUTPUT_DIRECTORY"]
            + plot_name
            + "_start.pdf",
            bbox_inches="tight",
        )
        plt.close(fig0)
    else:
        plt.show()

    fig1, ax1 = plt.subplots(figsize=(7, 5.8))

    # Draw nodes
    nodes_final = nx.draw_networkx_nodes(
        G_final,
        pos,
        vmin=0,
        vmax=VMAX,
        node_color=stat_dist_final,
        edgecolors="grey",
        cmap=ALGORITHM_PARAMETERS["COLOUR_MAP"],
        node_size=150,
        ax=ax1,
    )

    # Draw edges
    for u, v, d in G_final.edges(data=True):
        nx.draw_networkx_edges(
            G_final,
            pos,
            edgelist=[(u, v)],
            node_size=150,  # node_size for positioning edges
            alpha=0.8,
            width=d["weight"] * ALGORITHM_PARAMETERS["EDGE_FACTOR"],
            edge_color=ALGORITHM_PARAMETERS["GRAPH_COLOURS"][d["adjustable"]],
            connectionstyle="arc3,rad=0.05",
            ax=ax1,
        )

    # Draw node labels
    nx.draw_networkx_labels(G_final, pos, font_size=7, ax=ax1)
    plt.colorbar(nodes_final, ax=ax1)

    ax1.set_title(
        "The final network with nodes coloured by "
        "their stationary distribution value",
        fontsize=14,
    )

    plt.tight_layout()

    if ALGORITHM_PARAMETERS["SAVE_OUTPUT"]:
        plt.savefig(
            ALGORITHM_PARAMETERS["INSTANCE_OUTPUT_DIRECTORY"]
            + plot_name
            + "_final.pdf",
            bbox_inches="tight",
        )
        plt.close(fig1)
    else:
        plt.show()
