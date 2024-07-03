Matrix plots
============

If ``PLOT_MATRICES=True``, plots of the adjustable elements of four matrices at each iteration are created: the gradient estimates, stationary distributions, matrices in the unconstrained space and the transition matrices. If ``SAVE_OUTPUT=True``, the plots are saved as pdf files with name ``CURRENT_INSTANCE_NAME_PLOT_TYPE``, where ``PLOT_TYPE`` is equal to ``gradient``, ``stat_dist``, ``theta`` and ``transition``, respectively, in a folder named according to ``INSTANCE_OUTPUT_DIRECTORY``. These plots are only available if ``ALGORITHM_RUN_TYPE="single"``.
