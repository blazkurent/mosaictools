# mosaictools

`mosaictools` is a Python package for surrogate modeling of modal properties using the
Mode-Shape-Adapted Input parameter domain Cutting (MOSAIC) method.

## Overview

The package is designed for structural dynamics workflows where mode degeneration effects
(for example mode crossing, veering, or coalescence) make direct surrogate modeling difficult.
MOSAIC addresses this by splitting the parameter domain into modal subdomains and fitting
local generalized Polynomial Chaos Expansion (gPCE) models.

Reference preprint: [MOSAIC method](https://doi.org/10.2139/ssrn.5072693)

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

- `demo/notebooks/mosaic_example_notebook.ipynb`

The corresponding example arrays are in:

- `demo/data/parameters.npy`
- `demo/data/frequencies.npy`
- `demo/data/eigenvectors.npy`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).