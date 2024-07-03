Transition matrices
===================

The transition matrix at each iteration. This data set is saved as a CSV file with name ``Transition_matrices_CURRENT_INSTANCE_NAME`` if ``SAVE_OUTPUT=True`` in a folder named according to ``INSTANCE_OUTPUT_DIRECTORY``. The ith row represents the transition matrix of the (i-1)th iteration. Note that the matrices have been flattened to a row so that the output is 2D instead of 3D. This means that, for example, the first row of a 3x3 matrix is represented by columns 0-2, the second row by columns 3-5 and the third row by columns 6-8, see the example below. The matrix at the 0th iteration (i.e., the first row), represents the transition matrix that is optimised.

.. list-table:: Example transition matrix data set.
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
   * - 0.12
     - 0.30
     - 0.58
     - 0.05
     - 0.35
     - 0.60
     - 0.75
     - 0.05
     - 0.20
   * - 0.11
     - 0.29
     - 0.60
     - 0.04
     - 0.36
     - 0.60
     - 0.73
     - 0.07
     - 0.20

This data set can be reshaped so that a matrix is obtained at each iteration. For example, the matrix at the first iteration is as follows:

.. list-table:: The transition matrix at the first iteration.
   :widths: 5 5 5
   :header-rows: 1

   * - 0
     - 1
     - 2
   * - 0.11
     - 0.29
     - 0.60
   * - 0.04
     - 0.36
     - 0.60
   * - 0.73
     - 0.07
     - 0.20