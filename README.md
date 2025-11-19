STR and BUIP Econometric Estimation

A Python framework for Smooth Transition Regression (STR) and Behavioral Uncovered Interest Parity (BUIP) modeling. This repository contains robust, optimized implementations of non-linear time series models widely used in macro-econometrics.

Features

Smooth Transition Regression (STR): Logistic (LSTR) and Exponential (ESTR) transition functions.

Grid Search Optimization: Vectorized initialization to avoid local minima.

Robust Estimation: Non-linear Least Squares (NLS) with HAC (Newey-West) standard errors.

Diagnostics: Luukkonen et al. (1988) linearity tests and residual analysis.

Installation

Clone the repository:

git clone [https://github.com/lorenzobiondiecon/str-buip-estimation.git](https://github.com/lorenzobiondiecon/str-buip-estimation.git)
cd str-buip-estimation


Install dependencies:

pip install -r requirements.txt


Usage

Place your panel data in the data/ folder (see src/config.py for paths) and run the main pipeline:

python run_estimation.py


Methodology Note

The STR model is specified generally as:
$$ y_t = \phi_{lin}' \mathbf{x}{t} + G(\gamma, c; z_t) \cdot \phi{non}' \mathbf{x}_{t} + u_t $$

Where:

$\mathbf{x}_t$: Vector of regressors (can be distinct for linear/non-linear parts).

$G(\cdot)$: Transition function (Logistic by default).

$z_t$: Transition variable.

Correction from previous versions: This implementation ensures consistency between the initialization phase (Grid Search) and the estimation phase (NLS). Both phases now use the exact same design matrix construction, resolving potential specification biases.

