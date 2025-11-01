Matrices in the unconstrained space
===================================

The matrix in the unconstrained space at each iteration. This data set is saved as a CSV file with name ``Thetas_CURRENT_INSTANCE_NAME`` if ``SAVE_OUTPUT=True`` in a folder named according to ``INSTANCE_OUTPUT_DIRECTORY``. The ith row represents the matrix in the unconstrained space of the (i-1)th iteration. Note that the matrices have been flattened to a row so that the output is 2D instead of 3D. This means that, for example, the first row of a 3x3 matrix is represented by columns 0-2, the second row by columns 3-5 and the third row by columns 6-8, see the example below. The matrix at the 0th iteration (i.e., the first row), represents the inversely transformed transition matrix.

.. list-table:: Example data set of matrices in the unconstrained space.
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
   * - -6.907
     - 0.003
     - 0.001
     - -0.001
     - -6.907
     - 0.021
     - 0.032
     - -6.907
     - -0.045
   * - -6.907
     - 0.017
     - 0.015
     - 0.013
     - -6.907
     - 0.035
     - 0.046
     - -6.907
     - -0.031

This data set can be reshaped so that a matrix is obtained at each iteration. For example, the matrix at the first iteration is as follows:

.. list-table:: The matrix in the unconstrained space at the first iteration.
   :widths: 5 5 5
   :header-rows: 1

   * - 0
     - 1
     - 2
   * - -6.907
     - 0.017
     - 0.015
   * - 0.013
     - -6.907
     - 0.035
   * - 0.046
     - -6.907
     - -0.031