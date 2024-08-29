.. _examples:

Examples
========

Example network instances
+++++++++++++++++++++++++

Five example network instances can be found in the ``data/example_networks/`` folder of the `GitHub repository <https://github.com/NanneD/SM-SPSA>`_. These were generated with the ``generate_data_instances_set.py`` script with the standard settings for the second method of generating random network instances (see the :ref:`API <generatedatainstancessetapi>` for explanations).

These networks range in size from 6x6 to 9x9. The basic settings of the ``smspsa/main.py`` script automatically runs the SM-SPSA algorithm on all five network instances. Which output is provided can be specified by the variables ``PLOT_NETWORK``, ``PLOT_MATRICES``, ``PLOT_OBJECTIVE`` and ``SAVE_OUTPUT`` (also see :ref:`output data<outputdata>`).

Example_network_0
+++++++++++++++++

The following shows the results of running SM-SPSA on "example_network_0" to illustrate how the SM-SPSA algorithm works. The "normal" version of the algorithm is used with the standard settings as provided in ``smspsa/main.py``. The objective is to maximise the stationary distribution of node 7. The centred mass heuristic is used and the resulting transition matrix is optimised with SM-SPSA.

Figure 1 shows the network plot of the transition matrix that will be optimised.

.. figure:: _static/Network_plot_example_network_centred_mass_0_start.pdf

   Figure 1: The network that will be optimised.

The final objective value is equal to 0.42, which represents an increase of 241% compared to the objective value of the initial transition matrix. Figure 2 shows the objective value at each iteration. Figure 3 shows the final network plot.

.. figure:: _static/example_network_centred_mass_0_objective_function.pdf
   
   Figure 2: The objective value plot of the optimisation of "example_network_0"

.. figure:: _static/Network_plot_example_network_centred_mass_0_final.pdf

   Figure 3: The optimised network.