RUNNING_TIME_FILE_NAME
======================

If ``SAVE_OUTPUT=True``, a CSV file is generated where each row represents the running time of SM-SPSA for each network instance. The file is named according to the variable ``RUNNING_TIME_FILE_NAME``.


In the example below, running SM-SPSA on example_network_0 took 17.12 seconds and on example_network_1 12.19 seconds.

.. list-table:: Example output running times.
   :widths: 5 5
   :header-rows: 1

   * - Instance_name
     - Running_time (sec)
   * - example_network_0
     - 17.12
   * - example_network_1
     - 12.19