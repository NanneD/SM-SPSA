Gradient estimates
==================

The gradient estimate at each iteration. This data set is saved as a CSV file with name ``Gradient_estimates_CURRENT_INSTANCE_NAME`` if ``SAVE_OUTPUT=True`` in a folder named according to ``INSTANCE_OUTPUT_DIRECTORY``. The ith row represents the gradient estimates of the ith iteration. Note that the matrices have been flattened to a row so that the output is 2D instead of 3D. This means that, for example, the first row of a 3x3 matrix is represented by columns 0-2, the second row by columns 3-5 and the third row by columns 6-8, see the example below.

.. list-table:: Example gradient estimate data set.
   :widths: 5 5 5 5 5 5 5 5 5
   :header-rows: 1

   * - 0
     - 1
     - 2
     - 3
     - 4
     - 5
     - 6
     - 7
     - 8
   * - 0.11
     - 0.0
     - -0.11
     - 0.11
     - 0.11
     - 0.11
     - 0.0
     - 0.0
     - 0.11
   * - 0.012
     - 0.0
     - -0.012
     - 0.012
     - -0.012
     - 0.012
     - 0.0
     - 0.0
     - 0.012

This data set can be reshaped so that a matrix is obtained at each iteration. For example, the matrix at the second iteration is as follows:

.. list-table:: The gradient estimate matrix at the second iteration.
   :widths: 5 5 5
   :header-rows: 1

   * - 0
     - 1
     - 2
   * - 0.012
     - 0.0
     - -0.012
   * - 0.012
     - -0.012
     - 0.012
   * - 0.0
     - 0.0
     - 0.012