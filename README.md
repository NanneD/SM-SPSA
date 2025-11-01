![SM-SPSA logo](doc/_static/SMSPSA_logo.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# SM-SPSA

SM-SPSA (stochastic matrix simultaneous perturbation stochastic approximation) is an extension of the [SPSA algorithm](https://www.jhuapl.edu/SPSA/) to optimize a nonlinear objective function over the stationary distribution of a Markov chain. For more information, please visit the SM-SPSA website: https://nanned.github.io/SM-SPSA.

The code is written in Python 3.10.4.

## Installation

For installation instructions, please see: https://nanned.github.io/SM-SPSA/installation/installation.html.

## Documentation

For the user guide, including documentation and a quickstart, please see: https://nanned.github.io/SM-SPSA/userguide/userguide.html.

## Citing

If you would like to cite ``SM-SPSA``, please consider citing the following paper:
> Nanne A. Dieleman, Joost Berkhout, Bernd Heidergott (2025).
> A Pseudo-Gradient Approach for Model-free Markov Chain Optimization.
> Asia-Pacific Journal of Operational Research. doi: 10.1142/S0217595925500381.

Or, using the following BibTeX entry:

```bibtex
@article{Dieleman_Berkhout_Heidergott_2024,
	title = {A Pseudo-Gradient Approach for Model-free Markov Chain Optimization},
	author = {Dieleman, Nanne A. and Berkhout, Joost and Heidergott, Bernd},
	year = {2025},
        url = {https://doi.org/10.1142/S0217595925500381},
        journal = {Asia-Pacific Journal of Operational Research},
        doi = {10.1142/S0217595925500381},
} 
```

## License

The GNU General Public License v3 (GPL-3) license is used. For more information, please see the included LICENSE.md file.

## Contributing

If you would like to contribute to ``SM-SPSA`` in any way, please feel free to create an [issue](https://github.com/NanneD/SM-SPSA/issues) to discuss what you would like to add or change. Moreover, make sure that your code submission includes:
- tests
- type hints
- documentation
- docstrings for the added/changed methods, classes, etc. according to the NumPy docstrings format

To check whether the type hints and tests run smoothly, you can follow these steps:
1. Open the command line and move to the ``SMSPSA`` folder.
2. Run the tests by using the following command:
```
pytest smspsa/tests.py
```
3. Run the mypy checker by using:
```
mypy smspsa/
```