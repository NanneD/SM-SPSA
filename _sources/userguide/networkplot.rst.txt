Network plot
============

If ``PLOT_NETWORK=True``, network plots of the start and the optimised transition matrix are created. If ``SAVE_OUTPUT=True``, the plots are saved as pdf files with names ``NETWORK_PLOT_FILE_NAME_start`` and ``NETWORK_PLOT_FILE_NAME_final``.

The edges are coloured with the colours provided in ``GRAPH_COLOURS``, where the first and second colour represent the colour of the edges that cannot and can be adjusted, respectively. The nodes are coloured by the colour map specified in ``COLOUR_MAP`` according to their stationary distribution values. The edge weights are multiplied by ``EDGE_FACTOR`` to determine the edge width of the network plot.