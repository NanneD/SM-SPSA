.. _quickstart:

Quickstart
==========

For installation instructions, please see the :ref:`installation<installation>` section.

The SM-SPSA algorithm can be run through the main interface found in ``smspsa/main.py``. All algorithm parameters and input data can be set in this script. You can run the SM-SPSA algorithm by simply running ``smspsa/main.py`` either with an IDE or with the command line.

IDE Spyder
++++++++++
The IDE Spyder is automatically installed if you installed the ``SMSPSA`` environment by using the ``environment.yml`` file (see :ref:`installation instructions<installation>`). First, activate the ``SMSPSA`` environment by using the following command in the command line (assuming you use `Anaconda <https://anaconda.org>`_ or `Miniconda <https://docs.anaconda.com/miniconda/>`_):

.. code-block:: shell

   conda activate SMSPSA

Start Spyder by using the following command:

.. code-block:: shell

   spyder

Next, open the ``smspsa/main.py`` file in Spyder. You can then run the algorithm by pressing the "run file" button.

Command line
++++++++++++
First, open the command line and move to the SMSPSA folder on your computer (see the :ref:`installation instructions<installation>`). You can run the code through the command line by using:

.. code-block:: shell

	python smspsa/main.py

or, depending on your Python set-up:

.. code-block:: shell

	python3 smspsa/main.py

Input and output
++++++++++++++++
The SM-SPSA algorithm requires input data to work. An example set of network instances is provided in the ``data`` folder in the `GitHub repository <https://github.com/NanneD/SM-SPSA>`_, but you can also use your own networks. More information on the input data can be found in the :ref:`input data<inputdata>` section.

The algorithm can provide different types of output, such as plots. Which output is provided is determined by several variables that can be found in the ``smspsa/main.py`` file. If saving is turned on, the algorithm output is saved in the ``results`` folder, but this location can be changed in the ``smspsa/main.py`` script. When you run the code, an input validator automatically checks whether the provided parameters and data are valid. If invalid input is detected, an ``Exception`` is thrown with an explanation that specifies which input is invalid and how the problem can be solved. This makes experimenting with different parameters and input data easy. The required parameters are both explained in the ``smspsa/main.py`` script itself and in the :ref:`API<mainapi>`.

