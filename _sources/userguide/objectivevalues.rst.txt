Objective values
================

The objective value at each iteration. This data set is saved as a CSV file with name ``Objective_values_CURRENT_INSTANCE_NAME`` if ``SAVE_OUTPUT=True`` in a folder named according to ``INSTANCE_OUTPUT_DIRECTORY``. The ith row represents the objective value of the (i-1)th iteration. In the example below, the objective values at the first and third iterations are equal to 0.322 and 0.389, respectively. The objective value at the 0th iteration (i.e., the first value), represents the start objective value.

.. list-table:: Example objective value data set.
   :widths: 5
   :header-rows: 1

   * - Objective_value
   * - 0.321
   * - 0.322
   * - 0.334
   * - 0.389