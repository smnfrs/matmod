"""Variable classes for the IO-SFC model.

This module provides the Variable and LagVariable classes for storing
both symbolic (SymPy) and numeric representations of model variables.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
import sympy as sp


class Variable:
    """A class for creating variables that stores both a SymPy symbol
    and values for use in economic models.

    Attributes:
        name: Short identifier for the variable
        value: Current value (SymPy Matrix or scalar)
        scalar: Whether this is a scalar (True) or vector/matrix (False)
        symbol: SymPy symbol or MatrixSymbol for symbolic computation
        desc: Human-readable description
        time: Current time period
        hist: List of historical values
        iterations: Values during current period's iteration process
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
            value: Initial value - can be number, list, or matrix
            desc: Human-readable description
            scalar: True if scalar, False if vector/matrix
            time: Initial time period (default 0)
        """
        self.name = name
        self.value = self.instantiate_sympy(value, scalar)
        self.scalar = scalar
        self.symbol = self.instantiate_symbol(name, self.value, scalar)
        self.desc = desc
        self.time = time

        # Create time series history and add non-empty starting values
        self.hist = []
        # Check for empty values more robustly (including empty SymPy matrices)
        is_empty = (value is None or (isinstance(value, list) and len(value) == 0) or
                   (hasattr(self.value, 'shape') and 0 in self.value.shape))
        if not is_empty:
            self.hist.append(self.value)

        # Add instance to class variable
        self.__class__._all_vars.append(self)

        # List to store iteration values during solving
        self.iterations = [self.value]
        # Flag to reset iterations at beginning of new period
        self._new_period = False

    def __repr__(self):
        """Print method for the variable class."""
        value_str = sp.pretty(self.value, use_unicode=True)
        return f" {self.desc}: \n{value_str}"

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
        if len(self.hist) == 0 and not self.scalar:
            nan_val = sp.Matrix([[sp.nan]] * newval.shape[0])
            self.hist.append(nan_val)
        elif len(self.hist) == 0 and self.scalar:
            self.hist.append(sp.nan)

        self.value = newval
        self.iterations.append(newval)
        self.hist.append(newval)
        self._new_period = True

    def exogenous_update(self, newval):
        """Update an exogenous variable during simulation."""
        self.value = self.instantiate_sympy(newval, self.scalar)
        self.hist.append(self.value)

    def instantiate_symbol(self, name: str,
                          val: Optional[Union[int, float, List]],
                          scalar: bool) -> Union[sp.Symbol, sp.MatrixSymbol]:
        """Create a SymPy symbol of the appropriate shape."""
        if scalar:
            symbol = sp.Symbol(name)
        else:
            symbol = sp.MatrixSymbol(name, val.shape[0], val.shape[1])
        return symbol

    def instantiate_sympy(self, val: Optional[Union[int, float, List]],
                         scalar: bool) -> Union[sp.Expr, sp.Matrix]:
        """Convert input to SymPy array or value with specified shape."""
        # First convert sympy matrices or numpy vectors to list
        if isinstance(val, (sp.Matrix, np.ndarray)):
            val = val.tolist()

        # Handle empty/none case
        if val is None or (isinstance(val, list) and len(val) == 0):
            return sp.Integer(0) if scalar else sp.Matrix([[]])

        # Handle scalars
        if scalar:
            if isinstance(val, list):
                if len(val) != 1:
                    raise ValueError(f"Scalar must have 1 element, got {len(val)}")
                val = val[0]
            return sp.Float(val)

        # Raise error if expecting a matrix and receive something other than a list
        if not isinstance(val, list):
            raise ValueError(f"Sympy matrix object requires a list")

        return sp.Matrix(val)

    def instantiate_numpy(self, val: Optional[Union[int, float, List]],
                         scalar: bool) -> Union[int, float, np.ndarray]:
        """Convert input to number or numpy array."""
        # Handle empty/none case
        if val is None or (isinstance(val, list) and len(val) == 0):
            return 0 if scalar else np.array([[]])

        # Handle scalars
        if scalar:
            if isinstance(val, list):
                if len(val) != 1:
                    raise ValueError(f"Scalar must have 1 element, got {len(val)}")
            return val
        # Handle vectors and matrices
        else:
            # Raise error if expecting a matrix and receive something other than a list
            if not isinstance(val, list):
                raise ValueError(f"Numpy array requires a list")

            # Turn vectors into column vectors
            arr = np.array(val)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            # Also handle row vectors (1xn arrays)
            elif arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr.T
            # Error for more than two dimensional array
            elif arr.ndim > 2:
                raise ValueError(f"Can only work with one or two dimensional arrays, received {arr.ndim}-dimensional")

            return arr

    def create_time_series(self):
        """Format variable history into a time series DataFrame in long format."""
        var_df = pd.DataFrame(columns=['variable', 'row', 'column', 'time', 'value'])

        # Types of sympy matrices
        matrix_types = (sp.Matrix, sp.ImmutableDenseMatrix,
                       sp.ImmutableSparseMatrix, sp.MutableDenseMatrix)

        # Loop through the history list
        for time, time_val in enumerate(self.hist):
            # Check if value is a matrix or a single number
            if isinstance(self.hist[0], matrix_types):
                # Loop through matrix and extract values
                for num, val in enumerate(time_val):
                    new_row = {
                        'variable': self.symbol,
                        'element': self.symbol[num],
                        'row': self.symbol[num].i,
                        'column': self.symbol[num].j,
                        'time': time,
                        'value': val
                    }
                    var_df.loc[len(var_df)] = new_row
            else:
                new_row = {'variable': self.symbol, 'time': time, 'value': time_val}
                var_df.loc[len(var_df)] = new_row

        return var_df

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
        # Sum all elements of the matrix/vector
        return float(sum(self.value))

    def total_history(self):
        """Return time series of totals across all periods.

        Returns:
            List of totals, one per time period (NaN for missing values)
        """
        totals = []
        for h in self.hist:
            if self.scalar:
                # Handle sympy nan
                if h == sp.nan or (hasattr(h, 'is_nan') and h.is_nan):
                    totals.append(float('nan'))
                else:
                    totals.append(float(h))
            else:
                # Check if any element is nan
                if any(elem == sp.nan for elem in h):
                    totals.append(float('nan'))
                else:
                    totals.append(float(sum(h)))
        return totals

    def sector_history(self, sector_idx=0):
        """Return time series for a specific sector/element.

        Args:
            sector_idx: Index of sector (0-based)

        Returns:
            List of values for that sector across all periods (NaN for missing)
        """
        if self.scalar:
            result = []
            for h in self.hist:
                if h == sp.nan or (hasattr(h, 'is_nan') and h.is_nan):
                    result.append(float('nan'))
                else:
                    result.append(float(h))
            return result

        result = []
        for h in self.hist:
            # Handle empty matrices (shouldn't happen with NaN approach, but defensive)
            if hasattr(h, 'shape') and 0 in h.shape:
                result.append(float('nan'))
            # Handle NaN values
            elif h[sector_idx] == sp.nan:
                result.append(float('nan'))
            else:
                result.append(float(h[sector_idx]))
        return result

    def growth_rate(self, total=True):
        """Return period-over-period growth rates.

        Args:
            total: If True, compute growth of aggregate

        Returns:
            List of growth rates (length = len(hist) - 1), NaN where undefined
        """
        import math

        if total:
            values = self.total_history()
        else:
            values = self.sector_history(0)

        rates = []
        for i in range(1, len(values)):
            # If either value is NaN, growth rate is NaN
            if math.isnan(values[i-1]) or math.isnan(values[i]):
                rates.append(float('nan'))
            elif values[i-1] != 0:
                rate = (values[i] - values[i-1]) / values[i-1]
                rates.append(rate)
            else:
                rate = float('inf') if values[i] != 0 else 0
                rates.append(rate)

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
            # Plot each sector
            n_sectors = self.value.shape[0]
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
        super().__init__(
            name=name,
            value=initial_value.tolist() if hasattr(initial_value, 'shape') else float(initial_value),
            desc=f"{variable.desc} (t - {periods})",
            scalar=variable.scalar
        )

    def _get_lagged_value(self):
        """Get the current lagged value from source variable's history.

        Returns NaN matrix if history is empty (calculated variable not yet computed).
        """
        if len(self.source_variable.hist) == 0:
            # Source has no history yet - return its current value (may be empty matrix)
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
