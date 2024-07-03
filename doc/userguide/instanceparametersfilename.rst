INSTANCE_PARAMETERS_FILE_NAME
=============================

If ``ALGORITHM_RUN_TYPE="multiple"``, multiple network instances can be optimised consecutively. The required data about these network instances can be found in this table saved as a CSV file. The table contains six columns; ``Instance_name`` (the name of the instance), ``Random_M`` (whether the transition matrix was randomly generated or not), ``Random_C`` (whether the binary adjustment matrix was randomly generated or not), ``Seed`` (the seed used to generate the instance), ``Problem_size`` (the network size) and ``Objective_index`` (the objective index that should be optimised). The data in the columns ``Instance_name`` and ``Objective_index`` are used by the algorithm, but the other four data columns are convenient for your own reference. Note that all network instances that are part of this table will be optimised. The node numbers start counting at 0.

The table below shows an example instance parameter table. Example_network_1 has size 8x8 and node 4 will be optimised, for example.

.. list-table:: Example instance parameter table.
   :widths: 5 5 5 5 5 5
   :header-rows: 1

   * - Instance_name
     - Random_M
     - Random_C
     - Seed
     - Problem_size
     - Objective_index
   * - example_network_0
     - Yes
     - Yes
     - 50
     - 10
     - 8
   * - example_network_1
     - Yes
     - Yes
     - 51
     - 8
     - 4
   * - example_network_2
     - Yes
     - Yes
     - 52
     - 5
     - 3