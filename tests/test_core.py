"""
Core framework tests for matmod.

Tests Variable, LagVariable, Equation, and Model classes
with both scalar and matrix variables.

Usage:
    python tests/test_core.py
"""

import time
import numpy as np
import sympy as sp

from matmod import Variable, LagVariable, Equation, Model


class CoreTests:
    """Test suite for core matmod framework."""

    def __init__(self):
        self.results = {}

    def _report(self, test_name, passed, message=""):
        self.results[test_name] = {'passed': passed, 'message': message}
        return passed

    # ==================== Variable Tests ====================

    def test_scalar_variable(self):
        """Test scalar variable creation and value storage."""
        Variable._clean_vars()
        v = Variable('x', 3.14, 'test scalar', scalar=True)

        is_float = isinstance(v.value, float)
        correct_val = abs(float(v.value) - 3.14) < 1e-10
        has_hist = len(v.hist) == 1
        is_scalar = v.scalar is True

        passed = is_float and correct_val and has_hist and is_scalar
        return self._report('scalar_variable', passed,
                          f"type={type(v.value).__name__}, val={v.value}")

    def test_matrix_variable(self):
        """Test matrix/vector variable creation."""
        Variable._clean_vars()
        v = Variable('P', [1.5, 2.0], 'price vector', scalar=False)

        is_matrix = isinstance(v.value, np.ndarray)
        correct_shape = v.value.shape == (2, 1)
        correct_vals = float(v.value[0, 0]) == 1.5 and float(v.value[1, 0]) == 2.0

        passed = is_matrix and correct_shape and correct_vals
        return self._report('matrix_variable', passed,
                          f"shape={v.value.shape}, vals={v.value.T}")

    def test_empty_variable(self):
        """Test variable creation with no initial value."""
        Variable._clean_vars()
        v = Variable('y', None, 'empty', scalar=True)

        is_zero = v.value == 0
        empty_hist = len(v.hist) == 0

        passed = is_zero and empty_hist
        return self._report('empty_variable', passed,
                          f"value={v.value}, hist_len={len(v.hist)}")

    def test_variable_history(self):
        """Test that _iterate and _update manage history correctly."""
        Variable._clean_vars()
        v = Variable('x', 1.0, 'test', scalar=True)

        v._iterate(sp.Float(2.0))
        v._iterate(sp.Float(3.0))
        iterations_correct = len(v.iterations) == 3  # initial + 2

        v._update(sp.Float(3.0))
        hist_correct = len(v.hist) == 2  # initial + 1 update

        passed = iterations_correct and hist_correct
        return self._report('variable_history', passed,
                          f"iterations={len(v.iterations)}, hist={len(v.hist)}")

    def test_variable_registry(self):
        """Test that _all_vars tracks instances and _clean_vars clears."""
        Variable._clean_vars()
        a = Variable('a', 1, 'a', scalar=True)
        b = Variable('b', 2, 'b', scalar=True)

        count_before = len(Variable._all_vars)
        Variable._clean_vars()
        count_after = len(Variable._all_vars)

        passed = count_before == 2 and count_after == 0
        return self._report('variable_registry', passed,
                          f"before={count_before}, after={count_after}")

    # ==================== LagVariable Tests ====================

    def test_lag_variable(self):
        """Test LagVariable returns correct lagged value during simulation."""
        Variable._clean_vars()
        Equation._clean_vars()

        # Build a simple model: x doubles each period
        model = Model(t_length=3, iterations=5)
        x = model.add_var('x', 10.0, 'state', scalar=True)
        x_lag = x.lag('x_lag')
        model.add_eq('double', 'x_lag * 2', x, x_lag)
        model.run_all()

        # hist should be [10, 20, 40, 80]
        hist = [float(h) for h in x.hist]
        correct_hist = (
            abs(hist[0] - 10) < 1e-10 and
            abs(hist[1] - 20) < 1e-10 and
            abs(hist[2] - 40) < 1e-10 and
            abs(hist[3] - 80) < 1e-10
        )

        passed = correct_hist
        return self._report('lag_variable', passed, f"hist={hist}")

    def test_lag_update_blocked(self):
        """Test that LagVariables cannot be updated directly."""
        Variable._clean_vars()
        x = Variable('x', 5.0, 'test', scalar=True)
        x_lag = x.lag('x_lag')

        try:
            x_lag.exogenous_update(99.0)
            passed = False
            msg = "Should have raised AttributeError"
        except AttributeError:
            passed = True
            msg = "Correctly blocked"

        return self._report('lag_update_blocked', passed, msg)

    # ==================== Equation Tests ====================

    def test_scalar_equation(self):
        """Test equation evaluation with scalar variables."""
        Variable._clean_vars()
        Equation._clean_vars()

        x = Variable('x', 5.0, 'input', scalar=True)
        y = Variable('y', None, 'output', scalar=True)

        model = Model(t_length=1)
        model.add_variables(x, y)
        eq = Equation('eq_y', '2 * x + 1', y, x, model=model)

        result = float(eq.calc())
        passed = abs(result - 11.0) < 1e-10
        return self._report('scalar_equation', passed, f"2*5+1 = {result}")

    def test_matrix_equation(self):
        """Test equation evaluation with matrix variables."""
        Variable._clean_vars()
        Equation._clean_vars()

        a = Variable('a', [[0.1, 0.2], [0.15, 0.1]], 'IO coefficients', scalar=False)
        y = Variable('y', [[100], [80]], 'final demand', scalar=False)
        x = Variable('x', None, 'output', scalar=False)

        model = Model(t_length=1, n=2)
        model.add_variables(a, y, x)
        eq = Equation('eq_x', 'inv(eye(n) - a) * y', x, a, y, model=model)

        result = eq.calc()
        # (I - a)^-1 * y should give output > final demand
        x0, x1 = float(result[0, 0]), float(result[1, 0])
        passed = x0 > 100 and x1 > 80
        return self._report('matrix_equation', passed, f"x=[{x0:.1f}, {x1:.1f}]")

    def test_equation_with_constants(self):
        """Test that model constants are available in equation strings."""
        Variable._clean_vars()
        Equation._clean_vars()

        x = Variable('x', [10.0, 20.0], 'input', scalar=False)
        y = Variable('y', None, 'output', scalar=False)

        model = Model(t_length=1, n=2)
        model.add_variables(x, y)
        eq = Equation('eq_y', 'eye(n) * x', y, x, model=model)

        result = eq.calc()
        passed = float(result[0, 0]) == 10.0 and float(result[1, 0]) == 20.0
        return self._report('equation_constants', passed, f"result={result.T}")

    # ==================== Model Tests ====================

    def test_model_add_var_add_eq(self):
        """Test the model.add_var/add_eq workflow."""
        Variable._clean_vars()
        Equation._clean_vars()

        model = Model(t_length=5, iterations=10)
        x = model.add_var('x', 10.0, 'input', scalar=True)
        y = model.add_var('y', None, 'output', scalar=True)
        model.add_eq('eq_y', '2 * x', y, x)

        model.run_all()

        passed = float(y.value) == 20.0 and len(y.hist) > 1
        return self._report('model_add_var_eq', passed,
                          f"y={float(y.value)}, hist_len={len(y.hist)}")

    def test_model_add_all(self):
        """Test the add_all=True workflow."""
        Variable._clean_vars()
        Equation._clean_vars()

        x = Variable('x', 10.0, 'input', scalar=True)
        y = Variable('y', None, 'output', scalar=True)
        Equation('eq_y', '3 * x', y, x)

        model = Model(t_length=3, add_all=True)
        model.run_all()

        passed = float(y.value) == 30.0
        return self._report('model_add_all', passed, f"y={float(y.value)}")

    def test_model_lag_registration(self):
        """Test that LagVariables are auto-registered via add_eq."""
        Variable._clean_vars()
        Equation._clean_vars()

        model = Model(t_length=3, iterations=5)
        x = model.add_var('x', 10.0, 'state', scalar=True)
        x_lag = x.lag('x_lag')
        model.add_eq('eq_x', 'x_lag * 2', x, x_lag)

        lag_registered = 'x_lag' in model.variables
        model.run_all()
        # After 3 periods: 10 -> 20 -> 40 -> 80
        final_correct = abs(float(x.value) - 80.0) < 1e-10

        passed = lag_registered and final_correct
        return self._report('model_lag_registration', passed,
                          f"registered={lag_registered}, x={float(x.value)}")

    def test_gauss_seidel_convergence(self):
        """Test that circular dependencies converge via Gauss-Seidel."""
        Variable._clean_vars()
        Equation._clean_vars()

        model = Model(t_length=3, iterations=50)
        K = model.add_var('K', 100.0, 'capital', scalar=True)
        C = model.add_var('C', 50.0, 'consumption', scalar=True)
        W = model.add_var('W', 30.0, 'wages', scalar=True)
        X = model.add_var('X', 80.0, 'output', scalar=True)

        # C = 0.8*W + 10, W = 0.5*X, X = C + 0.1*K
        model.add_eq('eq_C', '0.8 * W + 10', C, W)
        model.add_eq('eq_W', '0.5 * X', W, X)
        model.add_eq('eq_X', 'C + 0.1 * K', X, C, K)

        model.run_all()

        # Check convergence: last two iterations should be close
        iterations = C.iterations
        if len(iterations) >= 2:
            diff = abs(float(iterations[-1]) - float(iterations[-2]))
            converged = diff < 1e-6
        else:
            converged = False

        passed = converged and float(C.value) > 0
        return self._report('gauss_seidel_convergence', passed,
                          f"C={float(C.value):.4f}, converged={converged}")

    def test_scalar_performance(self):
        """Test that scalar models run efficiently (no Rational explosion)."""
        Variable._clean_vars()
        Equation._clean_vars()

        model = Model(t_length=200, iterations=5)
        x = model.add_var('x', 1.0, 'state', scalar=True)
        x_lag = x.lag('x_lag')
        model.add_eq('growth', 'x_lag * 1.02', x, x_lag)

        t0 = time.time()
        model.run_all()
        elapsed = time.time() - t0

        # Should complete in under 10 seconds (was infinite before fix)
        passed = elapsed < 10.0 and float(x.value) > 1.0
        return self._report('scalar_performance', passed,
                          f"200 periods in {elapsed:.2f}s, x={float(x.value):.2f}")

    def test_time_series_methods(self):
        """Test total_history, sector_history, growth_rate."""
        Variable._clean_vars()
        Equation._clean_vars()

        model = Model(t_length=3, iterations=5)
        x = model.add_var('x', 10.0, 'state', scalar=True)
        x_lag = x.lag('x_lag')
        model.add_eq('double', 'x_lag * 2', x, x_lag)
        model.run_all()

        th = x.total_history()
        gr = x.growth_rate()

        hist_ok = len(th) == 4  # initial + 3 periods
        growth_ok = len(gr) == 3 and all(abs(g - 1.0) < 1e-10 for g in gr)

        passed = hist_ok and growth_ok
        return self._report('time_series_methods', passed,
                          f"hist={th}, growth={gr}")

    # ==================== Runner ====================

    def run_all(self, verbose=True):
        self.results = {}

        tests = [
            # Variables
            ("Scalar Variable", self.test_scalar_variable),
            ("Matrix Variable", self.test_matrix_variable),
            ("Empty Variable", self.test_empty_variable),
            ("Variable History", self.test_variable_history),
            ("Variable Registry", self.test_variable_registry),
            # LagVariables
            ("Lag Variable", self.test_lag_variable),
            ("Lag Update Blocked", self.test_lag_update_blocked),
            # Equations
            ("Scalar Equation", self.test_scalar_equation),
            ("Matrix Equation", self.test_matrix_equation),
            ("Equation Constants", self.test_equation_with_constants),
            # Model
            ("Model add_var/add_eq", self.test_model_add_var_add_eq),
            ("Model add_all", self.test_model_add_all),
            ("Model Lag Registration", self.test_model_lag_registration),
            ("Gauss-Seidel Convergence", self.test_gauss_seidel_convergence),
            ("Scalar Performance", self.test_scalar_performance),
            ("Time Series Methods", self.test_time_series_methods),
        ]

        all_passed = True

        if verbose:
            print("\n" + "=" * 60)
            print("MATMOD CORE TESTS")
            print("=" * 60)

        for test_name, test_func in tests:
            try:
                passed = test_func()
            except Exception as e:
                passed = False
                self._report(test_name, False, f"Exception: {e}")

            all_passed = all_passed and passed

            if verbose:
                status = "PASS" if passed else "FAIL"
                result = self.results.get(test_name, {})
                msg = result.get('message', '')
                print(f"\n{status}: {test_name}")
                if msg:
                    print(f"       {msg}")

        if verbose:
            print("\n" + "=" * 60)
            overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
            print(f"OVERALL: {overall}")
            print("=" * 60)

        return all_passed


if __name__ == '__main__':
    tests = CoreTests()
    tests.run_all(verbose=True)
