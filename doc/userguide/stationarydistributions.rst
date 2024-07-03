Stationary distributions
========================

The stationary distribution at each iteration. This data set is saved as a CSV file with name ``Stationary_distributions_CURRENT_INSTANCE_NAME`` if ``SAVE_OUTPUT=True`` in a folder named according to ``INSTANCE_OUTPUT_DIRECTORY``. The ith row represents the stationary distribution of the (i-1)th iteration. In the example below, node 1 has a stationary distribution value of 0.40 and 0.48 in the first and second iteration, respectively. The stationary distribution at the 0th iteration (i.e., the first row), represents the start stationary distribution. Note that the node numbers start at 0.

.. list-table:: Example stationary distribution data set.
   :widths: 5 5 5
   :header-rows: 1

   * - 0
     - 1
     - 2
   * - 0.25
     - 0.30
     - 0.45
   * - 0.10
     - 0.40
     - 0.50
   * - 0.12
     - 0.48
     - 0.40