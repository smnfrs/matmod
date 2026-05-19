"""Model class for the IO-SFC model.

This module provides the Model class that contains variables and equations,
and runs the simulation using Gauss-Seidel iteration.
"""

import pandas as pd
import numpy as np
from .variable import Variable
from .equation import Equation


class Model:
    """The main model class that contains variables, equations, and runs simulation.

    The model uses Gauss-Seidel iteration to solve the simultaneous equations
    within each time period, then advances to the next period.

    Attributes:
        t_length: Number of time periods to simulate
        iterations: Number of Gauss-Seidel iterations per period
        variables: Dict of variables in the model
        equations: Dict of equations in the model
        constants: Dict of constants for use in equations
        time: Current time period
    """

    # Default variables for quick plotting
    DEFAULT_KEY_VARS = ['X', 'I', 'C', 'K', 'U', 'Pi', 'W', 'B']

    def __init__(self, t_length: int, iterations: int = 50,
                 add_all: bool = False, **constants):
        """Initialize a Model.

        Args:
            t_length: Number of time periods to simulate
            iterations: Number of Gauss-Seidel iterations per period (default 50)
            add_all: If True, automatically add all Variable and Equation
                     instances to the model
            **constants: Constants for use within equations
        """
        self.t_length = t_length
        self.iterations = iterations
        self.variables = {}
        self.equations = {}
        self.constants = constants
        self.time = 0

        if add_all:
            self.add_variables(*Variable._all_vars)
            self.add_equations(*Equation._all_eqs)

    def run1(self):
        """Run one time period of the model.

        Performs Gauss-Seidel iteration, then updates all variables.
        """
        # Iterate the model for the number of iterations set
        for _ in range(self.iterations - 1):
            for eq in self.equations.values():
                eq.iterate()

        # Update the variables with the final values
        for eq in self.equations.values():
            eq.update()

        # Update the time variable
        self.time += 1

    def run_all(self):
        """Run the model for all time periods."""
        while self.time < self.t_length:
            self.run1()

    def add_variables(self, *variables: Variable):
        """Add predefined variables to the model."""
        self.variables.update({var.name: var for var in variables})

    def add_equations(self, *equations: Equation):
        """Add predefined equations to the model.

        Sets model reference on each equation to self so they can access model.constants.
        Initializes equations if they were created without a model.
        """
        for eq in equations:
            eq.model = self
            eq._initialize()  # Initialize now that model is available
            self.equations[eq.name] = eq

    def add_var(self, name: str, value=None, desc: str = None,
                scalar: bool = True, time: int = 0):
        """Create a new variable and add it to the model.

        The variable is only stored in model.variables, not global namespace.
        """
        var = Variable(name, value, desc, scalar, time)
        self.variables[var.name] = var
        return var

    def add_eq(self, name: str, equation: str, output: Variable,
               *inputs: Variable):
        """Create a new equation and add it to the model.

        The equation is only stored in model.equations, not global namespace.
        Constants are accessed from model.constants via the equation's model reference.
        Any input variables not yet in model.variables are registered automatically
        (this catches LagVariables created via var.lag()).
        """
        # Register any unregistered input variables (e.g. LagVariables)
        for var in inputs:
            if var.name not in self.variables:
                self.variables[var.name] = var
        eq = Equation(name, equation, output, *inputs, model=self)
        self.equations[eq.name] = eq
        return eq

    def unpack(self, vars: bool = True, eqs: bool = True):
        """Add model variables/equations to global namespace."""
        if vars:
            globals().update(self.variables)
        if eqs:
            globals().update(self.equations)

    def pack(self, vars: bool = True, eqs: bool = True):
        """Remove model variables/equations from global namespace."""
        if vars:
            for key in self.variables.keys():
                if key in globals():
                    del globals()[key]

        if eqs:
            for key in self.equations.keys():
                if key in globals():
                    del globals()[key]

    def create_df(self):
        """Create a long-format DataFrame of all variable time series."""
        df = pd.DataFrame()
        for var in self.variables.values():
            var_df = var.create_time_series()
            df = pd.concat([df, var_df])
        return df

    def create_lags(self, *variables: Variable):
        """Create lagged variables for each of the variables passed.

        TODO: Implement this method.
        """
        pass

    # =========== Information methods ===========

    def _get_end_vars(self):
        """Get list of endogenous variables (those with equations)."""
        return [equation.output for equation in self.equations.values()]

    def _get_used_vars(self):
        """Get list of all variables used as inputs in equations."""
        return [var for eq in self.equations.values() for var in eq.inputs]

    @property
    def var_info(self):
        """Return DataFrame with information about variables.

        Shows whether each variable is used in equations and whether
        it is endogenous (has an equation defining it).
        """
        names = {'Variables': [var.name for var in self.variables.values()]}
        used = {'Used?': [var in self._get_used_vars()
                         for var in self.variables.values()]}
        endog = {'Endogenous': [var in self._get_end_vars()
                               for var in self.variables.values()]}

        vars_dict = names | used | endog

        return pd.DataFrame(vars_dict)

    # =========== Plotting Methods ===========

    def plot_variable(self, var_name, sectors=None, total=False, ax=None, **kwargs):
        """Plot time series for a single variable.

        Args:
            var_name: Name of variable to plot
            sectors: List of sector indices to include (None = all)
            total: If True, plot aggregate total
            ax: Matplotlib axes (creates new if None)
            **kwargs: Passed to plt.plot()

        Returns:
            matplotlib axes object
        """
        import matplotlib.pyplot as plt

        if var_name not in self.variables:
            raise ValueError(f"Variable '{var_name}' not found in model")

        var = self.variables[var_name]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        times = range(len(var.hist))

        if var.scalar or total:
            values = var.total_history()
            ax.plot(times, values, label=var_name, **kwargs)
        else:
            n_sectors = var.value.shape[0]
            plot_sectors = sectors if sectors is not None else range(n_sectors)
            for i in plot_sectors:
                values = var.sector_history(i)
                ax.plot(times, values, label=f'{var_name}[{i}]', **kwargs)

        ax.set_xlabel('Time')
        ax.set_ylabel(var_name)
        ax.set_title(var.desc)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_key_variables(self, var_names=None, total=True, figsize=(14, 10)):
        """Quick plot of key model variables with sensible defaults.

        This is the primary plotting method for quick model inspection.

        Args:
            var_names: List of variable names (default: DEFAULT_KEY_VARS)
            total: If True, plot aggregates (default: True)
            figsize: Figure size tuple

        Returns:
            matplotlib figure object
        """
        import matplotlib.pyplot as plt

        if var_names is None:
            var_names = [v for v in self.DEFAULT_KEY_VARS if v in self.variables]

        n_vars = len(var_names)
        n_cols = 2
        n_rows = (n_vars + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for idx, var_name in enumerate(var_names):
            if var_name in self.variables:
                self.plot_variable(var_name, total=total, ax=axes[idx])
            else:
                axes[idx].text(0.5, 0.5, f'{var_name} not found',
                              ha='center', va='center')
                axes[idx].set_title(var_name)

        # Hide unused subplots
        for idx in range(n_vars, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('Key Model Variables', fontsize=14, fontweight='bold')
        fig.tight_layout()

        return fig

    def plot_multiple(self, var_names, normalize=False, total=True, ax=None):
        """Plot multiple variables on same axes for comparison.

        Args:
            var_names: List of variable names
            normalize: If True, normalize to first non-NaN value = 100
            total: If True, plot aggregates
            ax: Matplotlib axes (creates new if None)

        Returns:
            matplotlib axes object
        """
        import matplotlib.pyplot as plt
        import math

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for var_name in var_names:
            if var_name not in self.variables:
                continue
            var = self.variables[var_name]
            values = var.total_history() if total else var.sector_history(0)

            if normalize:
                # Find first non-NaN value for normalization
                base_val = None
                for v in values:
                    if not math.isnan(v) and v != 0:
                        base_val = v
                        break
                if base_val is not None:
                    values = [100 * v / base_val if not math.isnan(v) else float('nan')
                              for v in values]

            times = range(len(values))
            ax.plot(times, values, label=var_name)

        ax.set_xlabel('Time')
        ax.set_ylabel('Value' + (' (normalized)' if normalize else ''))
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def summary_stats(self):
        """Return summary statistics for all endogenous variables.

        Returns:
            DataFrame with mean, std, min, max, growth rate for each variable
            (uses nan-aware functions to handle missing initial values)
        """
        stats = []
        for var in self._get_end_vars():
            values = np.array(var.total_history())
            growth = np.array(var.growth_rate())

            stats.append({
                'variable': var.name,
                'mean': np.nanmean(values),
                'std': np.nanstd(values),
                'min': np.nanmin(values),
                'max': np.nanmax(values),
                'final': values[-1],
                'avg_growth': np.nanmean(growth) if len(growth) > 0 else np.nan
            })

        return pd.DataFrame(stats)

    # =========== Matrix Expansion Helpers ===========

    def _get_variable_size(self, var):
        """Get the number of scalar elements in a variable.

        Returns:
            int: Number of scalar elements (1 for scalars, n for n×1 vectors, n*m for n×m matrices)
        """
        import sympy as sp
        if var.scalar:
            return 1
        if hasattr(var.value, 'shape'):
            shape = var.value.shape
            return shape[0] * shape[1] if len(shape) > 1 else shape[0]
        return 1

    def _expand_to_scalars(self):
        """Expand all matrix variables and equations to scalar form.

        Returns:
            dict: {
                'endo_vars': list of (name, index, symbol) for endogenous scalars,
                'equations': list of (eq_name, index, expr, output_var_name, output_idx) tuples,
                'n_scalar_eqs': int,
                'n_scalar_endo': int,
                'element_symbols': dict mapping MatrixElement -> Symbol,
                'reverse_mapping': dict mapping Symbol -> (var_name, i, j)
            }
        """
        import sympy as sp

        # First, build element symbols (shared across all uses)
        element_symbols, reverse_mapping = self._build_element_symbols()

        # Expand endogenous variables to scalars
        scalar_endo = []  # (var_name, index, symbol_element)
        for eq in self.equations.values():
            var = eq.output
            if var.scalar:
                scalar_endo.append((var.name, None, var.symbol))
            else:
                # For matrix variables, use the pre-built element symbols
                if hasattr(var.value, 'shape'):
                    n_rows = var.value.shape[0]
                    n_cols = var.value.shape[1] if len(var.value.shape) > 1 else 1
                    for i in range(n_rows):
                        for j in range(n_cols):
                            idx = (i, j) if n_cols > 1 else i
                            # Get the symbol from element_symbols
                            if n_cols > 1:
                                mat_elem = var.symbol[i, j]
                            else:
                                mat_elem = var.symbol[i, 0]
                            elem_symbol = element_symbols.get(mat_elem)
                            if elem_symbol is None:
                                # Fallback: create new symbol
                                elem_symbol = sp.Symbol(f"{var.name}_{i}_{j}" if n_cols > 1 else f"{var.name}_{i}", real=True)
                            scalar_endo.append((var.name, idx, elem_symbol))

        # Expand equations to scalar form
        scalar_eqs = []  # (eq_name, index, expr_element, output_var_name, output_idx)
        for eq in self.equations.values():
            expr = eq.expr
            output = eq.output

            if output.scalar:
                scalar_eqs.append((eq.name, None, expr, output.name, None))
            else:
                # Matrix expression - extract each element
                if hasattr(expr, 'shape'):
                    n_rows = expr.shape[0]
                    n_cols = expr.shape[1] if len(expr.shape) > 1 else 1
                    for i in range(n_rows):
                        for j in range(n_cols):
                            idx = (i, j) if n_cols > 1 else i
                            # Get the element expression
                            if n_cols > 1:
                                elem_expr = expr[i, j]
                            else:
                                elem_expr = expr[i]
                            scalar_eqs.append((eq.name, idx, elem_expr, output.name, idx))
                else:
                    # Single element returned from matrix equation
                    scalar_eqs.append((eq.name, None, expr, output.name, None))

        return {
            'endo_vars': scalar_endo,
            'equations': scalar_eqs,
            'n_scalar_eqs': len(scalar_eqs),
            'n_scalar_endo': len(scalar_endo),
            'element_symbols': element_symbols,
            'reverse_mapping': reverse_mapping
        }

    def _build_element_symbols(self):
        """Build a mapping from MatrixSymbol elements to substitutable symbols.

        For proper differentiation, we need individual symbols for each matrix element.

        Returns:
            tuple: (element_symbols dict, reverse_mapping dict)
        """
        import sympy as sp

        element_symbols = {}  # MatrixSymbol[i,j] -> Symbol
        reverse_mapping = {}  # Symbol -> (var_name, i, j)

        for var in self.variables.values():
            if var.scalar:
                element_symbols[var.symbol] = var.symbol
                reverse_mapping[var.symbol] = (var.name, None, None)
            else:
                if hasattr(var.value, 'shape'):
                    n_rows = var.value.shape[0]
                    n_cols = var.value.shape[1] if len(var.value.shape) > 1 else 1
                    for i in range(n_rows):
                        for j in range(n_cols):
                            if n_cols > 1:
                                elem = var.symbol[i, j]
                                sym = sp.Symbol(f"{var.name}_{i}_{j}", real=True)
                            else:
                                elem = var.symbol[i, 0] if hasattr(var.symbol, '__getitem__') else var.symbol[i]
                                sym = sp.Symbol(f"{var.name}_{i}", real=True)
                            element_symbols[elem] = sym
                            reverse_mapping[sym] = (var.name, i, j if n_cols > 1 else 0)

        return element_symbols, reverse_mapping

    def _get_element_value(self, var, i, j=0):
        """Get the numeric value of a variable element."""
        if var.scalar:
            return float(var.value)
        arr = var.value  # 2-D ndarray by the Variable invariant
        return float(arr[i, j] if arr.shape[1] > 1 else arr[i, 0])

    # =========== Dependency Analysis Methods (Phase 1) ===========

    def dependency_graph(self):
        """Build a dependency graph from model equations.

        Returns:
            dict[str, list[str]]: Maps output_var → [input_vars] for each equation
        """
        graph = {}
        for eq in self.equations.values():
            graph[eq.output_name] = eq.input_names
        return graph

    def classify_variables(self):
        """Classify variables as endogenous, exogenous, or unused.

        Returns:
            dict[str, list[str]]: {'endogenous': [...], 'exogenous': [...], 'unused': [...]}
        """
        # Endogenous = have an equation defining them
        endogenous = [eq.output_name for eq in self.equations.values()]

        # All variables used as inputs
        used_as_input = set()
        for eq in self.equations.values():
            used_as_input.update(eq.input_names)

        # Exogenous = used as input but not endogenous
        exogenous = [name for name in used_as_input if name not in endogenous]

        # Unused = in model.variables but never used
        all_var_names = set(self.variables.keys())
        used_vars = set(endogenous) | used_as_input
        unused = [name for name in all_var_names if name not in used_vars]

        return {
            'endogenous': sorted(endogenous),
            'exogenous': sorted(exogenous),
            'unused': sorted(unused)
        }

    def dependency_matrix(self):
        """Create an incidence matrix showing variable usage in equations.

        Returns:
            pd.DataFrame: Rows are equations, columns are variables,
                         values are 1 if variable is used in equation, 0 otherwise.
                         Output variable is marked with -1.
        """
        # Get all variable names (inputs and outputs)
        all_vars = set()
        for eq in self.equations.values():
            all_vars.add(eq.output_name)
            all_vars.update(eq.input_names)
        all_vars = sorted(all_vars)

        # Build the matrix
        data = []
        eq_names = []
        for eq in self.equations.values():
            row = {}
            for var in all_vars:
                if var == eq.output_name:
                    row[var] = -1  # Output variable
                elif var in eq.input_names:
                    row[var] = 1   # Input variable
                else:
                    row[var] = 0   # Not used
            data.append(row)
            eq_names.append(eq.name)

        return pd.DataFrame(data, index=eq_names)

    # =========== Analysis Methods (Phase 2) ===========

    def check_determination(self, expand_matrices=True):
        """Check if the model is exactly determined.

        Args:
            expand_matrices: If True, count scalar elements of matrix variables/equations.
                           If False, count matrix variables as single entities.

        Returns:
            dict: {
                'determined': bool,
                'n_equations': int,
                'n_endogenous': int,
                'n_matrix_equations': int,  # Original count before expansion
                'n_matrix_endogenous': int,
                'status': str,  # 'exactly_determined', 'underdetermined', 'overdetermined'
                'issues': list[str]
            }
        """
        classification = self.classify_variables()
        n_matrix_equations = len(self.equations)
        n_matrix_endogenous = len(classification['endogenous'])

        issues = []

        # Check for variables that appear as output but not in model.variables
        for eq in self.equations.values():
            if eq.output_name not in self.variables:
                issues.append(f"Output '{eq.output_name}' not in model.variables")

        # Check for input variables not defined anywhere
        for eq in self.equations.values():
            for input_name in eq.input_names:
                if input_name not in self.variables and input_name not in classification['endogenous']:
                    issues.append(f"Input '{input_name}' not defined in model")

        if expand_matrices:
            # Count scalar elements
            expanded = self._expand_to_scalars()
            n_equations = expanded['n_scalar_eqs']
            n_endogenous = expanded['n_scalar_endo']
        else:
            n_equations = n_matrix_equations
            n_endogenous = n_matrix_endogenous

        if n_equations == n_endogenous:
            status = 'exactly_determined'
            determined = True
        elif n_equations < n_endogenous:
            status = 'underdetermined'
            determined = False
            issues.append(f"More endogenous variables ({n_endogenous}) than equations ({n_equations})")
        else:
            status = 'overdetermined'
            determined = False
            issues.append(f"More equations ({n_equations}) than endogenous variables ({n_endogenous})")

        return {
            'determined': determined and len(issues) == 0,
            'n_equations': n_equations,
            'n_endogenous': n_endogenous,
            'n_matrix_equations': n_matrix_equations,
            'n_matrix_endogenous': n_matrix_endogenous,
            'status': status,
            'issues': issues
        }

    def find_circular_dependencies(self):
        """Find strongly connected components (cycles) in the dependency graph.

        Uses Tarjan's algorithm to find all groups of variables that form
        feedback loops.

        Returns:
            list[list[str]]: List of cycles, each cycle is a list of variable names.
                            Only returns cycles with more than one variable.
        """
        graph = self.dependency_graph()

        # Tarjan's algorithm implementation
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True

            # Consider successors (variables this node depends on that are also outputs)
            for successor in graph.get(node, []):
                if successor in graph:  # Only follow edges to endogenous variables
                    if successor not in index:
                        strongconnect(successor)
                        lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                    elif on_stack.get(successor, False):
                        lowlinks[node] = min(lowlinks[node], index[successor])

            # If node is a root node, pop the stack and generate an SCC
            if lowlinks[node] == index[node]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == node:
                        break
                if len(scc) > 1:  # Only return non-trivial cycles
                    sccs.append(sorted(scc))

        for node in graph:
            if node not in index:
                strongconnect(node)

        return sccs

    def jacobian_symbolic(self, expand_matrices=True, as_dataframe=False):
        """Compute the symbolic Jacobian matrix of the model equations.

        The Jacobian J[i,j] = ∂eq_i/∂var_j represents how each equation's
        output responds to changes in each endogenous variable.

        Args:
            expand_matrices: If True, expand matrix equations to scalar elements.
            as_dataframe: If True, return a pandas DataFrame with labeled rows/columns.

        Returns:
            sp.Matrix or pd.DataFrame: Symbolic Jacobian matrix.
            Rows are equations, columns are endogenous variables.
        """
        import sympy as sp

        if not expand_matrices:
            # Original behavior (doesn't work well for matrices)
            classification = self.classify_variables()
            endogenous = classification['endogenous']
            endo_symbols = []
            for name in endogenous:
                if name in self.variables:
                    var = self.variables[name]
                    endo_symbols.append(var.symbol)
                else:
                    endo_symbols.append(sp.Symbol(name))

            jacobian_rows = []
            for eq in self.equations.values():
                row = []
                for symbol in endo_symbols:
                    try:
                        deriv = sp.diff(eq.expr, symbol)
                        row.append(deriv)
                    except Exception:
                        row.append(sp.Integer(0))
                jacobian_rows.append(row)

            if not jacobian_rows:
                return sp.Matrix([])
            return sp.Matrix(jacobian_rows)

        # Expand matrices to scalars (includes element_symbols)
        expanded = self._expand_to_scalars()
        element_symbols = expanded['element_symbols']
        reverse_mapping = expanded['reverse_mapping']

        # Get list of endogenous scalar symbols (in order)
        endo_scalar_symbols = []
        endo_labels = []
        for var_name, idx, elem_sym in expanded['endo_vars']:
            endo_scalar_symbols.append(elem_sym)
            if idx is None:
                endo_labels.append(var_name)
            elif isinstance(idx, tuple):
                endo_labels.append(f"{var_name}[{idx[0]},{idx[1]}]")
            else:
                endo_labels.append(f"{var_name}[{idx}]")

        # Build Jacobian row by row (one row per scalar equation)
        jacobian_rows = []
        eq_labels = []

        for eq_name, eq_idx, expr, out_name, out_idx in expanded['equations']:
            # Substitute MatrixSymbol elements with regular symbols for differentiation
            expr_substituted = expr
            for mat_elem, scalar_sym in element_symbols.items():
                try:
                    expr_substituted = expr_substituted.subs(mat_elem, scalar_sym)
                except:
                    pass

            row = []
            for endo_sym in endo_scalar_symbols:
                try:
                    deriv = sp.diff(expr_substituted, endo_sym)
                    row.append(deriv)
                except Exception:
                    row.append(sp.Integer(0))
            jacobian_rows.append(row)

            if eq_idx is None:
                eq_labels.append(eq_name)
            elif isinstance(eq_idx, tuple):
                eq_labels.append(f"{eq_name}[{eq_idx[0]},{eq_idx[1]}]")
            else:
                eq_labels.append(f"{eq_name}[{eq_idx}]")

        if not jacobian_rows:
            if as_dataframe:
                return pd.DataFrame()
            return sp.Matrix([])

        # Store metadata for later use
        self._jacobian_metadata = {
            'row_labels': eq_labels,
            'col_labels': endo_labels,
            'element_symbols': element_symbols,
            'reverse_mapping': reverse_mapping
        }

        if as_dataframe:
            return pd.DataFrame(jacobian_rows, index=eq_labels, columns=endo_labels)

        return sp.Matrix(jacobian_rows)

    def jacobian_numeric(self, expand_matrices=True, as_dataframe=False):
        """Compute the numeric Jacobian matrix at current variable values.

        Args:
            expand_matrices: If True, expand matrix equations to scalar elements.
            as_dataframe: If True, return a pandas DataFrame with labeled rows/columns.

        Returns:
            np.ndarray or pd.DataFrame: Numeric Jacobian matrix evaluated at current values.
            Rows are equations, columns are endogenous variables.
        """
        import sympy as sp

        J_sym = self.jacobian_symbolic(expand_matrices=expand_matrices)

        if J_sym.shape == (0, 0):
            if as_dataframe:
                return pd.DataFrame()
            return np.array([])

        # Build substitution dict
        subs_dict = {}

        if expand_matrices and hasattr(self, '_jacobian_metadata'):
            # Use element symbols
            element_symbols = self._jacobian_metadata['element_symbols']
            reverse_mapping = self._jacobian_metadata['reverse_mapping']

            for scalar_sym, (var_name, i, j) in reverse_mapping.items():
                if var_name in self.variables:
                    var = self.variables[var_name]
                    if i is None:
                        subs_dict[scalar_sym] = float(var.value)
                    else:
                        subs_dict[scalar_sym] = self._get_element_value(var, i, j)
        else:
            # Non-expanded path: only scalar substitutions are well-defined
            # (matrix variables require expand_matrices=True).
            for var in self.variables.values():
                if var.scalar:
                    subs_dict[var.symbol] = float(var.value)

        # Evaluate the Jacobian numerically
        try:
            J_num = J_sym.subs(subs_dict).evalf()
            J_arr = np.array(J_num.tolist(), dtype=float)

            if as_dataframe and hasattr(self, '_jacobian_metadata'):
                row_labels = self._jacobian_metadata['row_labels']
                col_labels = self._jacobian_metadata['col_labels']
                return pd.DataFrame(J_arr, index=row_labels, columns=col_labels)

            return J_arr
        except Exception as e:
            # If evaluation fails, return NaN
            n = J_sym.shape[0]
            J_arr = np.full((n, n), np.nan)

            if as_dataframe and hasattr(self, '_jacobian_metadata'):
                row_labels = self._jacobian_metadata['row_labels']
                col_labels = self._jacobian_metadata['col_labels']
                return pd.DataFrame(J_arr, index=row_labels, columns=col_labels)

            return J_arr

    def stability_analysis(self, expand_matrices=True):
        """Analyze the stability of the model using eigenvalue analysis.

        Args:
            expand_matrices: If True, expand matrix equations to scalar elements.

        Returns:
            dict: {
                'eigenvalues': np.ndarray,
                'stable': bool,  # True if spectral radius < 1
                'spectral_radius': float,
                'max_real': float,  # Maximum real part of eigenvalues
                'jacobian_shape': tuple
            }
        """
        J = self.jacobian_numeric(expand_matrices=expand_matrices)

        if J.size == 0:
            return {
                'eigenvalues': np.array([]),
                'stable': True,
                'max_real': 0.0,
                'spectral_radius': 0.0,
                'jacobian_shape': (0, 0)
            }

        # Check for NaN
        if np.any(np.isnan(J)):
            return {
                'eigenvalues': np.array([np.nan]),
                'stable': False,
                'max_real': np.nan,
                'spectral_radius': np.nan,
                'jacobian_shape': J.shape,
                'error': 'Jacobian contains NaN values'
            }

        try:
            eigenvalues = np.linalg.eigvals(J)
            max_real = np.max(np.abs(eigenvalues.real))

            # For discrete-time models, stability requires |eigenvalues| < 1
            spectral_radius = np.max(np.abs(eigenvalues))
            stable = spectral_radius < 1

            return {
                'eigenvalues': eigenvalues,
                'stable': stable,
                'max_real': max_real,
                'spectral_radius': spectral_radius,
                'jacobian_shape': J.shape
            }
        except Exception as e:
            return {
                'eigenvalues': np.array([np.nan]),
                'stable': False,
                'max_real': np.nan,
                'spectral_radius': np.nan,
                'jacobian_shape': J.shape,
                'error': str(e)
            }

    # =========== Calibration Methods (Phase 3) ===========

    def topological_order(self):
        """Find a topological ordering of equations if one exists.

        Returns:
            list[str] | None: Ordered list of equation names, or None if cycles exist
        """
        graph = self.dependency_graph()
        classification = self.classify_variables()
        endogenous = set(classification['endogenous'])

        # Build in-degree count (only counting endogenous dependencies)
        in_degree = {node: 0 for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                if dep in endogenous:
                    in_degree[node] += 1

        # Kahn's algorithm
        queue = [node for node, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            # For each node that depends on this one, decrease in-degree
            for other_node, deps in graph.items():
                if node in deps and other_node in in_degree:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)

        # If we processed all nodes, we found a valid ordering
        if len(order) == len(graph):
            return order
        else:
            return None  # Cycles exist

    def find_fixed_point(self, max_iter=1000, tol=1e-8, verbose=False):
        """Iterate equations until convergence to find a fixed point.

        WARNING: This finds a fixed point of the ITERATION PROCESS, not a
        long-run steady state. If your model has growth (e.g., G > 0), there
        is no steady state, but this method will still converge to a consistent
        set of values for one period.

        For true steady-state analysis, you need to either:
        1. Set growth rates to zero
        2. Use a model-specific calibration that solves analytically

        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            verbose: If True, print iteration progress

        Returns:
            dict: {
                'converged': bool,
                'iterations': int,
                'max_change': float,
                'values': dict[str, any]  # Final values for all endogenous variables
            }
        """
        # Store initial values to restore later
        initial_values = {name: var.value for name, var in self.variables.items()}

        converged = False
        iteration = 0
        max_change = float('inf')

        for iteration in range(max_iter):
            max_change = 0.0

            for eq in self.equations.values():
                old_value = eq.output.value

                # Calculate new value
                new_value = eq.calc()

                # Calculate change (works for float or ndarray values)
                try:
                    change = float(np.max(np.abs(np.subtract(new_value, old_value))))
                    max_change = max(max_change, change)
                except (TypeError, ValueError):
                    pass

                # Update value (but not history)
                eq.output.value = new_value

            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: max_change = {max_change:.2e}")

            if max_change < tol:
                converged = True
                break

        # Collect final values
        final_values = {}
        for eq in self.equations.values():
            final_values[eq.output_name] = eq.output.value

        # Restore initial values if we don't want to modify the model
        for name, val in initial_values.items():
            if name in self.variables:
                self.variables[name].value = val

        return {
            'converged': converged,
            'iterations': iteration + 1,
            'max_change': max_change,
            'values': final_values
        }

    # Alias for backwards compatibility
    def find_steady_state(self, max_iter=1000, tol=1e-8, verbose=False):
        """Alias for find_fixed_point(). See that method for documentation."""
        return self.find_fixed_point(max_iter, tol, verbose)

    def calibration_report(self, expand_matrices=True):
        """Generate a comprehensive calibration report.

        Args:
            expand_matrices: If True, count scalar elements of matrix variables.

        Returns:
            pd.DataFrame: Report with variable classification, current values,
                         and cycle membership
        """
        classification = self.classify_variables()
        cycles = self.find_circular_dependencies()
        determination = self.check_determination(expand_matrices=expand_matrices)

        # Build cycle membership lookup
        cycle_membership = {}
        for i, cycle in enumerate(cycles):
            for var in cycle:
                if var in cycle_membership:
                    cycle_membership[var].append(i + 1)
                else:
                    cycle_membership[var] = [i + 1]

        # Build report data
        data = []

        # Add endogenous variables
        for name in classification['endogenous']:
            var = self.variables.get(name)
            if var:
                try:
                    if var.scalar:
                        value_str = f"{float(var.value):.4f}"
                    else:
                        value_str = str([f"{v:.4f}"
                                         for v in var.value.reshape(-1)])
                except (TypeError, ValueError):
                    value_str = str(var.value)
            else:
                value_str = "N/A"

            cycles_str = ",".join(map(str, cycle_membership.get(name, []))) or "-"

            data.append({
                'variable': name,
                'type': 'endogenous',
                'value': value_str,
                'in_cycle': cycles_str
            })

        # Add exogenous variables
        for name in classification['exogenous']:
            var = self.variables.get(name)
            if var:
                try:
                    if var.scalar:
                        value_str = f"{float(var.value):.4f}"
                    else:
                        value_str = str([f"{v:.4f}"
                                         for v in var.value.reshape(-1)])
                except (TypeError, ValueError):
                    value_str = str(var.value)
            else:
                value_str = "N/A"

            data.append({
                'variable': name,
                'type': 'exogenous',
                'value': value_str,
                'in_cycle': '-'
            })

        # Add unused variables
        for name in classification['unused']:
            var = self.variables.get(name)
            if var:
                try:
                    if var.scalar:
                        value_str = f"{float(var.value):.4f}"
                    else:
                        value_str = str([f"{v:.4f}"
                                         for v in var.value.reshape(-1)])
                except (TypeError, ValueError):
                    value_str = str(var.value)
            else:
                value_str = "N/A"

            data.append({
                'variable': name,
                'type': 'unused',
                'value': value_str,
                'in_cycle': '-'
            })

        df = pd.DataFrame(data)

        # Print summary
        print("=" * 60)
        print("CALIBRATION REPORT")
        print("=" * 60)
        print(f"\nModel Status: {determination['status']}")
        if expand_matrices and 'n_matrix_equations' in determination:
            print(f"  Matrix equations: {determination['n_matrix_equations']}")
            print(f"  Matrix endogenous: {determination['n_matrix_endogenous']}")
            print(f"  Scalar equations (expanded): {determination['n_equations']}")
            print(f"  Scalar endogenous (expanded): {determination['n_endogenous']}")
        else:
            print(f"  Equations: {determination['n_equations']}")
            print(f"  Endogenous variables: {determination['n_endogenous']}")
        print(f"  Exogenous variables: {len(classification['exogenous'])}")
        print(f"  Unused variables: {len(classification['unused'])}")

        if cycles:
            print(f"\nCircular Dependencies ({len(cycles)} cycles):")
            for i, cycle in enumerate(cycles):
                print(f"  Cycle {i+1}: {' → '.join(cycle)} → {cycle[0]}")
        else:
            print("\nNo circular dependencies found.")

        if determination['issues']:
            print("\nIssues:")
            for issue in determination['issues']:
                print(f"  - {issue}")

        print("=" * 60)

        return df
