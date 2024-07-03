M_FILE_NAME
===========

The transition matrix of the network instance that is optimised saved as a CSV file. An element at position :math:`(i,j)` represents the transition probability from node :math:`i` to node :math:`j`. A transition matrix is a square and stochastic matrix. A stochastic matrix has row sums equal to one and all elements represent a probability, i.e., have a value between 0 and 1. The node numbers start counting at 0.

The table below shows an example transition matrix. In this case, the transition probability from node 1 to node 3 is equal to 0.3 and the transition probability from node 3 to node 2 is equal to 0.2, for example.

.. list-table:: Example transition matrix.
   :widths: 5 5 5 5
   :header-rows: 1

   * - 0
     - 1
     - 2
     - 3
   * - 0.1
     - 0.2
     - 0.6
     - 0.1
   * - 0.5
     - 0.1
     - 0.1
     - 0.3
   * - 0.8
     - 0.1
     - 0.1
     - 0.0
   * - 0.2
     - 0.1
     - 0.2
     - 0.5