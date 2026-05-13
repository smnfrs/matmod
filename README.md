# matmod

A Python framework for building and simulating macroeconomic models using symbolic mathematics.

Macroeconomic models are systems of simultaneous equations — prices depend on wages, wages depend on output, output depends on prices. Writing these as code usually means choosing between readable math and runnable simulations. matmod lets you do both: write equations as strings that look like the textbook, and the solver handles the circular dependencies for you.

## What it does

- **Equations as strings** -- write `'(eye(n) - a)**-1 * y'` and matmod parses it into a SymPy expression, resolving variable references automatically
- **Scalar and matrix variables** -- the same API works for single numbers and for vectors or matrices, so a one-sector toy model and a multi-sector IO model use the same code patterns
- **Gauss-Seidel solver** -- within each time period, the solver iterates through all equations until values converge, handling circular dependencies without requiring the user to re-order anything
- **Built-in analysis** -- once a model is defined, you can inspect its dependency structure, compute symbolic or numeric Jacobians, check eigenvalue stability, and generate calibration reports

## Installation

```bash
git clone https://github.com/smnfrs/matmod.git
cd matmod
pip install -e .
```

Requires Python 3.10+. Dependencies (sympy, numpy, pandas, matplotlib) are installed automatically.

## Quick example

A minimal model with capital accumulation:

```python
from matmod import Variable, LagVariable, Equation, Model

model = Model(t_length=100, iterations=20)

# Variables
K = model.add_var('K', 100.0, 'Capital stock', scalar=True)
K_lag = K.lag('K_lag')
g = model.add_var('g', 0.03, 'Growth rate', scalar=True)

# Equation: capital grows at rate g each period
model.add_eq('accumulation', 'K_lag * (1 + g)', K, K_lag, g)

# Simulate
model.run_all()
print(f"Capital after 100 periods: {float(K.value):.2f}")
```

## How it works

Each **variable** holds two things: a SymPy symbol (used when parsing equations) and a numeric value (used during simulation). You define relationships between variables by writing **equations** as strings — matmod parses these into symbolic expressions, making functions like `eye()`, `diag()`, `inv()`, and `log()` available automatically. Any keyword arguments passed to the Model constructor (e.g. `n=2`) become constants you can reference in equation strings.

When you call `model.run_all()`, the solver steps through time. At each period it substitutes the current numeric values into every equation, iterates until convergence (Gauss-Seidel method), then saves the results to each variable's history. Lagged variables automatically pull from this history, so dynamic models with feedback loops work out of the box.

After simulation, every variable has a complete time series you can query with methods like `total_history()`, `growth_rate()`, and `plot()`.

## Analysis tools

Once a model is defined, you can ask structural questions about it:

- **Dependency graph** -- which variables feed into which equations
- **Circular dependencies** -- detect feedback loops using Tarjan's SCC algorithm
- **Determination check** -- does the number of equations match the number of unknowns (with automatic expansion of matrix variables to scalar elements)
- **Jacobian** -- symbolic partial derivatives of all equations with respect to all endogenous variables, optionally evaluated numerically at current values
- **Stability analysis** -- eigenvalues of the Jacobian and spectral radius to assess whether the model converges or explodes
- **Calibration report** -- a summary table showing every variable's classification, current value, and cycle membership

## Tests

```bash
python tests/test_core.py       # 16 tests: variables, equations, solver, performance
python tests/test_analysis.py   # 15 tests: dependency analysis, Jacobian, stability
```

## License

MIT
