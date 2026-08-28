# mosaictools

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ymssp.2025.113381-blue)](https://doi.org/10.1016/j.ymssp.2025.113381)
[![PyPI - Version](https://img.shields.io/pypi/v/mosaictools?style=flat)](https://pypi.org/project/mosaictools/)

A Python package for surrogate modeling of modal properties using the
Mode-Shape-Adapted Input parameter domain Cutting (MOSAIC) method.

- Source code: [https://github.com/blazkurent/mosaictools](https://github.com/blazkurent/mosaictools)
- Bug reports: [https://github.com/blazkurent/mosaictools/issues](https://github.com/blazkurent/mosaictools/issues)

## Overview

The package is designed for structural dynamics workflows where mode degeneration effects
(for example mode crossing, veering, or coalescence) make direct surrogate modeling difficult.
MOSAIC addresses this by splitting the parameter domain into modal subdomains and fitting
local generalized Polynomial Chaos Expansion (gPCE) models.

Reference article: [A novel approach to surrogate modelling of modal properties: Mode-shape-adapted input parameter domain cutting](https://www.sciencedirect.com/science/article/pii/S0888327025010829)

## Installation

Install from PyPI:

```bash
pip install mosaictools
```

## Features

- Global multi-mode surrogate model through `Mosaic`.
- Automatic mode-wise subdomain discovery from eigenvector similarity (MAC-based clustering).
- Local frequency and eigenvector approximation in each subdomain.
- Built-in classifier support:
  - `classification_method='svc'` (scikit-learn `SVC`)
  - `classification_method='custom'` (any compatible classifier instance)
- K-fold cross-validation helper and modal error metrics.
- Model persistence to `.msic` files.

## Demo Notebook and Data

A complete demonstration is included in:

- [https://github.com/blazkurent/mosaictools/blob/main/notebooks/mosaic_package_demonstration.ipynb](https://github.com/blazkurent/mosaictools/blob/main/notebooks/mosaic_package_demonstration.ipynb)

The corresponding example arrays are in:

- [https://github.com/blazkurent/mosaictools/tree/main/mosaic_example_data](https://github.com/blazkurent/mosaictools/tree/main/mosaic_example_data)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
