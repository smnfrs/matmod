"""Equation class for the IO-SFC model.

This module provides the Equation class for defining relationships
between variables using string expressions that are parsed into SymPy.
"""

import re
import sympy as sp
from .variable import Variable


def _diag_wrapper(X):
    """
    Create diagonal matrix from vector without unpacking operator.

    Replaces sp.diag(*X) with sp.diag(X) syntax.

    Args:
        X: SymPy Matrix (vector), MatrixSymbol, or scalar

    Returns:
        Diagonal matrix as SymPy Matrix

    Raises:
        ValueError: If X is a matrix (not a vector)
    """
    if isinstance(X, sp.MatrixSymbol):
        # Handle MatrixSymbol (symbolic matrix)
        if X.shape[1] == 1:  # Column vector (n, 1)
            n = X.shape[0]
            return sp.diag(*[X[i, 0] for i in range(n)])
        elif X.shape[0] == 1:  # Row vector (1, n)
            n = X.shape[1]
            return sp.diag(*[X[0, i] for i in range(n)])
        else:
            raise ValueError(f"diag() requires a vector, got matrix with shape {X.shape}")
    elif isinstance(X, sp.MatrixBase):
        # Handle concrete Matrix
        if X.shape[1] == 1:  # Column vector (n, 1)
            return sp.diag(*X)
        elif X.shape[0] == 1:  # Row vector (1, n)
            return sp.diag(*X.T)
        else:
            raise ValueError(f"diag() requires a vector, got matrix with shape {X.shape}")
    else:
        # Handle scalar or symbolic expressions
        return sp.diag(X)


class Equation:
    """Defines a relationship between variables using a string expression.

    The equation string is parsed into a SymPy expression, which can then
    be evaluated numerically by substituting variable values.

    Attributes:
        name: Identifier for the equation
        equation_str: The original string expression
        output: Variable that stores the result
        inputs: Tuple of input Variables
        constants: Dict of constants used in the equation
        expr: Parsed SymPy expression
    """

    # Class variable to store all equations
    _all_eqs = []

    @classmethod
    def _clean_vars(cls):
        """Clean the _all_eqs list. Call before defining a new model."""
        cls._all_eqs = []

    def __init__(self, name: str, equation: str, output: Variable,
                 *inputs: Variable, model=None):
        """Initialize an Equation.

        Args:
            name: Identifier for this equation (e.g., 'prices', 'output')
            equation: String representation of equation (e.g., 'x + y*z')
            output: Variable that will store the result
            *inputs: Variable objects that are inputs to the equation
            model: Model instance (provides access to constants via model.constants)
        """
        self.name = name
        self.equation_str = equation
        self.output = output
        self.inputs = inputs
        self.model = model

        # Lazy parsing - will parse on first access to expr property
        self._expr = None
        self._parsed = False
        self._initialized = False

        # Placeholder for lambdified function (not currently used)
        self.lambdified_func = None
        self.output_val = None

        # Add instance to class variable first
        self.__class__._all_eqs.append(self)

        # Initialize if we have a model, otherwise defer until model is set
        if self.model is not None:
            self._initialize()

    def _initialize(self):
        """Initialize the equation after model is set.

        This includes parsing, calculating initial value, updating symbols, and iterating.
        Called when model is set or on first use if not already initialized.
        """
        if self._initialized:
            return

        # Calculate initial value and save to output
        self.output_val = self.calc()

        # Update the symbol of the output to be the correct dimensions
        self.output.symbol = self.update_symbol()

        # Iterate the first time for the output variable to be used in other equations
        self.iterate()

        self._initialized = True

    @property
    def expr(self):
        """Get the parsed expression, parsing if needed.

        Uses lazy parsing to allow equations to be created before model assignment.
        When accessed, parses the equation using constants from self.model.constants.
        """
        if not self._parsed:
            self._expr = self.parse()
            self._parsed = True
        return self._expr

    def parse(self):
        """Parse the equation string into a SymPy expression.

        Replaces variable names with their symbols and evaluates
        in a namespace with common SymPy functions.
        """
        # Build a dict of input variables
        inputs_dict = {}
        for var in self.inputs:
            inputs_dict[var.name] = var

        # Create namespace with essential sympy functions and constants
        namespace = {
            'inputs_dict': inputs_dict,
            'sp': sp,
            # Common sympy functions
            'eye': sp.eye,
            'diag': _diag_wrapper,
            'Matrix': sp.Matrix,
            'zeros': sp.zeros,
            'ones': sp.ones,
            'Identity': sp.Identity,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'exp': sp.exp,
            'log': sp.log,
            'sqrt': sp.sqrt,
            'pi': sp.pi,
            'E': sp.E,
            'I': sp.I,
            # Matrix functions
            'transpose': lambda x: x.T,
            'inv': lambda x: x**-1 if hasattr(x, '__pow__') else sp.Matrix(x).inv(),
            'det': sp.det,
            'trace': sp.trace,
        }

        # Add constants from model if available
        if self.model is not None and hasattr(self.model, 'constants'):
            namespace.update(self.model.constants)

        # Replace variable names in the string with their symbols
        modified_str = self.equation_str

        for var in self.inputs:
            # Use regex to replace whole variable names
            # The pattern uses word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(var.name) + r'\b'
            modified_str = re.sub(
                pattern,
                f'inputs_dict["{var.name}"].symbol',
                modified_str
            )

        # Evaluate to create the sympy expression
        try:
            expr = eval(modified_str, namespace)
        except NameError as e:
            print(f"Error: {e}")
            print(f"Available names in namespace: {list(namespace.keys())}")
            print(f"Modified string: {modified_str}")
            raise

        return expr

    def calc(self):
        """Substitute input values into expression and return the result."""
        symbol2value = {}
        for var in self.inputs:
            symbol2value[var.symbol] = var.value

        result = self.expr.subs(symbol2value).doit()

        # Handle scalar results (not a matrix)
        if not hasattr(result, '__len__'):
            return result

        # Handle 1-element matrices - extract the scalar
        if len(result) == 1:
            return result[0]
        else:
            return result

    def update_symbol(self):
        """Update the symbolic representation to match output dimensions.

        Important when the variable has been declared empty.
        """
        symbol = self.output.instantiate_symbol(
            self.output.name,
            self.output_val,
            self.output.scalar
        )
        return symbol

    def iterate(self):
        """Run an iteration and add result to output's iteration list.

        Used during Gauss-Seidel solving within a period.
        """
        self.output_val = self.calc()
        self.output._iterate(self.output_val)

    def update(self):
        """Update the output variable at end of period.

        The iteration is run one last time and result is saved to history.
        """
        self.output_val = self.calc()
        self.output._update(self.output_val)

    def __repr__(self):
        """Print method showing the equation."""
        expr_str = sp.pretty(self.expr, use_unicode=True)
        return f"{self.output.name}, {self.output.desc} = \n {expr_str}"

    @property
    def input_names(self):
        """Return list of input variable names."""
        return [var.name for var in self.inputs]

    @property
    def output_name(self):
        """Return name of output variable."""
        return self.output.name

    # =========== Legacy/utility methods ===========

    def run(self):
        """Update output value and symbol (legacy method)."""
        self.output.value = self.output_val
        self.output.symbol = self.output.instantiate_symbol(
            self.output.name,
            self.output_val,
            self.output.scalar
        )

    def lambdify(self):
        """Create lambdified function with sympy backend."""
        input_symbols = [var.symbol for var in self.inputs]
        self.lambdified_func = sp.lambdify(
            input_symbols,
            self.expr,
            modules='sympy'
        )
