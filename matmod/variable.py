"""Variable classes for the IO-SFC model.

This module provides the Variable and LagVariable classes. Each variable
stores a *numeric* value (a Python float for scalars, a 2-D NumPy array for
vectors/matrices) used during simulation, plus a SymPy ``symbol`` used only by
the symbolic analysis layer (Jacobian, determination checks, pretty-printing).

The numeric value is the hot-path representation: the Gauss-Seidel solver reads
and writes it every iteration. The symbol is touched only by analysis methods,
which run once, off the hot path.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
import sympy as sp


def _is_empty(val) -> bool:
    """True if ``val`` represents 'no value yet' (None, [], or a 0-size array)."""
    if val is None:
        return True
    if isinstance(val, list) and len(val) == 0:
        return True
    if hasattr(val, 'shape') and 0 in getattr(val, 'shape', ()):
        return True
    return False


class Variable:
    """A model variable storing a numeric value and a SymPy symbol.

    Attributes:
        name: Short identifier for the variable
        value: Current numeric value (float for scalars, 2-D np.ndarray otherwise)
        scalar: Whether this is a scalar (True) or vector/matrix (False)
        symbol: SymPy Symbol or MatrixSymbol (used by the analysis layer only)
        desc: Human-readable description
        time: Current time period
        hist: List of historical numeric values
        iterations: Numeric values during the current period's iteration process
    """

    # Class variable to store all variables
    _all_vars = []

    @classmethod
    def _clean_vars(cls):
        """Clean the _all_vars list. Call before defining a new model."""
        cls._all_vars = []

    def __init__(self, name: str,
                 value: Union[int, float, List, None, sp.Matrix, np.ndarray] = None,
                 desc: str = None,
                 scalar: bool = True,
                 time: int = 0):
        """Initialize a Variable.

        Args:
            name: Short identifier (e.g., 'K', 'x', 'P')
            value: Initial value - number, list, NumPy array, or SymPy Matrix
            desc: Human-readable description
            scalar: True if scalar, False if vector/matrix
            time: Initial time period (default 0)
        """
        self.name = name
        self.value = self.instantiate_numpy(value, scalar)
        self.scalar = scalar
        self.symbol = self.instantiate_symbol(name, self.value, scalar)
        self.desc = desc
        self.time = time

        # Create time series history; seed it only with a real starting value
        self.hist = []
        if not _is_empty(value):
            self.hist.append(self.value)

        # Add instance to class variable
        self.__class__._all_vars.append(self)

        # List to store iteration values during solving
        self.iterations = [self.value]
        # Flag to reset iterations at beginning of new period
        self._new_period = False

    def __repr__(self):
        """Print method for the variable class."""
        return f" {self.desc}: \n{self.value}"

    def _iterate(self, newval):
        """Iterate the variable forward but don't save to history.

        Used during the Gauss-Seidel iteration process within a period.
        """
        self.value = newval
        if self._new_period:
            self.iterations = []
            self._new_period = False
        self.iterations.append(newval)

    def _update(self, newval):
        """Save the most recent value to history and prepare for next period.

        Called at the end of each period after convergence.
        If this is a calculated variable (empty hist), prepend NaN for t=0.
        """
        # If hist is empty, this is a calculated variable - prepend NaN for t=0
        if len(self.hist) == 0:
            if self.scalar:
                self.hist.append(float('nan'))
            else:
                self.hist.append(np.full(np.asarray(newval).shape, np.nan))

        self.value = newval
        self.iterations.append(newval)
        self.hist.append(newval)
        self._new_period = True

    def exogenous_update(self, newval):
        """Update an exogenous variable during simulation."""
        self.value = self.instantiate_numpy(newval, self.scalar)
        self.hist.append(self.value)

    def instantiate_symbol(self, name: str,
                          val: Optional[Union[int, float, List]],
                          scalar: bool) -> Union[sp.Symbol, sp.MatrixSymbol]:
        """Create a SymPy symbol of the appropriate shape.

        Shape is taken from the numeric value so the analysis layer (Jacobian
        expansion) sees the correct matrix dimensions.
        """
        if scalar:
            return sp.Symbol(name)
        arr = np.asarray(val)
        rows = arr.shape[0] if arr.ndim >= 1 else 1
        cols = arr.shape[1] if arr.ndim >= 2 else 1
        return sp.MatrixSymbol(name, rows, cols)

    def instantiate_numpy(self, val: Optional[Union[int, float, List]],
                          scalar: bool) -> Union[float, np.ndarray]:
        """Convert input to a Python float (scalar) or 2-D NumPy array.

        Vectors are stored as column vectors with shape (n, 1) so that they
        compose correctly with NumPy matrix multiplication and with the
        SymPy-generated element indexing used by lambdified equations.
        """
        # Normalise SymPy matrices to nested lists first
        if isinstance(val, sp.MatrixBase):
            val = val.tolist()

        # Empty / none case
        if _is_empty(val):
            return 0.0 if scalar else np.empty((1, 0), dtype=float)

        # Scalars
        if scalar:
            if isinstance(val, (list, tuple, np.ndarray)):
                flat = np.asarray(val, dtype=float).reshape(-1)
                if flat.size != 1:
                    raise ValueError(f"Scalar must have 1 element, got {flat.size}")
                return float(flat[0])
            return float(val)

        # Vectors and matrices
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 1:
            # Treat a flat list as a column vector
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] > 1:
            # Treat a 1xn row as a column vector
            arr = arr.T
        elif arr.ndim > 2:
            raise ValueError(
                f"Can only work with one or two dimensional arrays, "
                f"received {arr.ndim}-dimensional"
            )
        return arr

    # Kept for backwards compatibility; no longer used by the hot path.
    def instantiate_sympy(self, val, scalar):
        """Legacy: convert input to a SymPy value/matrix."""
        if isinstance(val, (sp.Matrix, np.ndarray)):
            val = np.asarray(val).tolist()
        if _is_empty(val):
            return sp.Integer(0) if scalar else sp.Matrix([[]])
        if scalar:
            if isinstance(val, list):
                if len(val) != 1:
                    raise ValueError(f"Scalar must have 1 element, got {len(val)}")
                val = val[0]
            return sp.Float(val)
        if not isinstance(val, list):
            raise ValueError("Sympy matrix object requires a list")
        return sp.Matrix(val)

    def _is_nan(self, h) -> bool:
        """True if a history entry is the NaN placeholder."""
        if np.isscalar(h) or (hasattr(h, 'ndim') and np.asarray(h).ndim == 0):
            try:
                return bool(np.isnan(float(h)))
            except (TypeError, ValueError):
                return False
        arr = np.asarray(h, dtype=float)
        return arr.size > 0 and bool(np.isnan(arr).any())

    def create_time_series(self):
        """Format variable history into a long-format time series DataFrame."""
        rows = []
        for time, time_val in enumerate(self.hist):
            if self.scalar:
                rows.append({'variable': self.symbol, 'row': None,
                             'column': None, 'time': time, 'value': time_val})
            else:
                arr = np.asarray(time_val, dtype=float)
                n_rows = arr.shape[0]
                n_cols = arr.shape[1] if arr.ndim > 1 else 1
                for i in range(n_rows):
                    for j in range(n_cols):
                        rows.append({
                            'variable': self.symbol,
                            'row': i,
                            'column': j,
                            'time': time,
                            'value': arr[i, j] if arr.ndim > 1 else arr[i],
                        })
        return pd.DataFrame(
            rows, columns=['variable', 'row', 'column', 'time', 'value'])

    def lag(self, name: str, periods: int = 1):
        """Create a LagVariable that tracks this variable's history.

        Args:
            name: Name to use for the lag variable (recommended: same as
                  the variable name in global namespace, e.g., 'K_lag1')
            periods: Number of periods to lag (default: 1)

        Returns:
            A LagVariable instance
        """
        return LagVariable(self, name, periods)

    # =========== Time Series Methods ===========

    def total(self):
        """Sum across all elements (industries) to get aggregate total.

        Returns:
            For vectors: sum of all elements (float)
            For scalars: the value itself (float)
        """
        if self.scalar:
            return float(self.value)
        return float(np.sum(self.value))

    def total_history(self):
        """Return time series of totals across all periods.

        Returns:
            List of totals, one per time period (NaN for missing values)
        """
        totals = []
        for h in self.hist:
            if self._is_nan(h):
                totals.append(float('nan'))
            elif self.scalar:
                totals.append(float(h))
            else:
                totals.append(float(np.sum(np.asarray(h, dtype=float))))
        return totals

    def sector_history(self, sector_idx=0):
        """Return time series for a specific sector/element.

        Args:
            sector_idx: Index of sector (0-based)

        Returns:
            List of values for that sector across all periods (NaN for missing)
        """
        result = []
        for h in self.hist:
            if self._is_nan(h):
                result.append(float('nan'))
            elif self.scalar:
                result.append(float(h))
            else:
                result.append(float(np.asarray(h, dtype=float).reshape(-1)[sector_idx]))
        return result

    def growth_rate(self, total=True):
        """Return period-over-period growth rates.

        Args:
            total: If True, compute growth of aggregate

        Returns:
            List of growth rates (length = len(hist) - 1), NaN where undefined
        """
        import math

        values = self.total_history() if total else self.sector_history(0)

        rates = []
        for i in range(1, len(values)):
            if math.isnan(values[i-1]) or math.isnan(values[i]):
                rates.append(float('nan'))
            elif values[i-1] != 0:
                rates.append((values[i] - values[i-1]) / values[i-1])
            else:
                rates.append(float('inf') if values[i] != 0 else 0)

        return rates

    def plot(self, by_sector=True, total=False, ax=None, **kwargs):
        """Plot this variable's time series.

        Args:
            by_sector: If True, plot each sector as separate line
            total: If True, plot aggregate total
            ax: Matplotlib axes (creates new if None)
            **kwargs: Passed to plt.plot()

        Returns:
            matplotlib axes object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        times = range(len(self.hist))

        if self.scalar or total:
            values = self.total_history()
            label = f'{self.name} (Total)' if not self.scalar else self.name
            ax.plot(times, values, label=label, **kwargs)
        elif by_sector:
            n_sectors = np.asarray(self.value).shape[0]
            for i in range(n_sectors):
                values = self.sector_history(i)
                ax.plot(times, values, label=f'{self.name}[{i}]', **kwargs)

        ax.set_xlabel('Time')
        ax.set_ylabel(self.name)
        ax.set_title(self.desc)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class LagVariable(Variable):
    """A Variable that provides lagged values from another Variable's history.

    Note: The value property returns hist[-periods] which is correct DURING model
    execution (between update() calls). After a simulation completes, hist[-1] is
    the current value, so querying value post-simulation returns the current value,
    not the lagged value. For post-simulation analysis, access hist directly.
    """

    def __init__(self, variable: Variable, name: str, periods: int = 1):
        """Initialize a LagVariable.

        Args:
            variable: The Variable instance to lag
            name: Name for this lagged variable
            periods: Number of periods to lag (default: 1)
        """
        self.source_variable = variable
        self.periods = periods

        # Get the initial lagged value
        initial_value = self._get_lagged_value()

        # Initialize the Variable base class
        if hasattr(initial_value, 'shape') and np.asarray(initial_value).ndim >= 1:
            init = np.asarray(initial_value, dtype=float).tolist()
        else:
            init = float(initial_value)

        super().__init__(
            name=name,
            value=init,
            desc=f"{variable.desc} (t - {periods})",
            scalar=variable.scalar
        )

    def _get_lagged_value(self):
        """Get the current lagged value from source variable's history.

        Returns the source's current value if its history is empty (calculated
        variable not yet computed).
        """
        if len(self.source_variable.hist) == 0:
            return self.source_variable.value
        elif self.periods <= len(self.source_variable.hist):
            return self.source_variable.hist[-self.periods]
        else:
            return self.source_variable.hist[0]

    @property
    def value(self):
        """Override value property to always return current lagged value."""
        return self._get_lagged_value()

    @value.setter
    def value(self, newval):
        """Setter for value - needed because parent class tries to set it.
        Does nothing as LagVariables derive their value from source."""
        pass

    # Override methods that shouldn't work on LagVariables
    def _iterate(self, newval):
        """LagVariables don't iterate independently."""
        pass

    def _update(self, newval):
        """LagVariables don't update independently."""
        pass

    def exogenous_update(self, newval):
        """LagVariables can't be updated exogenously."""
        raise AttributeError("LagVariables cannot be updated exogenously")
