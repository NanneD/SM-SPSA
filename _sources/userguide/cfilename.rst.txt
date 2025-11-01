C_FILE_NAME
===========

The binary adjustment matrix of the network instance that is optimised saved as a CSV file. The binary adjustment matrix has value :math:`1` or :math:`0` at position :math:`(i,j)` if edge :math:`(i,j)` can be optimised or not, respectively. Note that the edges in a row can only be adjusted if at least two row-elements have value 1. The node numbers start counting at 0.

The table below shows an example binary adjustment matrix. In this case, edge (1,0) cannot be adjusted, whereas edge (0,1) can be adjusted.

.. list-table:: Example binary adjustment matrix.
   :widths: 5 5 5 5
   :header-rows: 1

   * - 0
     - 1
     - 2
     - 3
   * - 0
     - 1
     - 1
     - 1
   * - 0
     - 0
     - 0
     - 1
   * - 1
     - 1
     - 1
     - 1
   * - 0
     - 0
     - 0
     - 0