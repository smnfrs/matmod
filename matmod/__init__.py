"""matmod — Macroeconomic Model Framework

Build and simulate macroeconomic models using symbolic mathematics.

Define variables, write equations as strings, and let the solver handle
the rest. Supports scalar and matrix variables, lagged values, Gauss-Seidel
iteration, and built-in analysis tools (Jacobian, stability, dependency graphs).

Classes:
    Variable: Symbolic and numeric variable with history tracking
    LagVariable: Time-lagged access to a Variable's history
    Equation: Relationship between variables, parsed from a string expression
    Model: Solver engine using Gauss-Seidel iteration
"""

from .variable import Variable, LagVariable
from .equation import Equation
from .model import Model

__all__ = ['Variable', 'LagVariable', 'Equation', 'Model']
__version__ = '0.1.0'
