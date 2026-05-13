"""
Test suite for matmod analysis and calibration methods.

Tests the analysis framework including:
- Dependency analysis (dependency_graph, classify_variables, dependency_matrix)
- Model analysis (check_determination, find_circular_dependencies, jacobian, stability)
- Calibration (topological_order, find_fixed_point, calibration_report)

Usage:
    python tests/test_analysis.py
"""

import numpy as np
import pandas as pd
import sympy as sp

from matmod import Variable, LagVariable, Equation, Model


class AnalysisTests:
    """Test suite for Model analysis and calibration methods."""

    def __init__(self):
        self.results = {}

    def _report(self, test_name, passed, message=""):
        """Record test result."""
        self.results[test_name] = {'passed': passed, 'message': message}
        return passed

    def setup_simple_model(self):
        """Create a simple 2-variable scalar model for testing."""
        Variable._clean_vars()
        Equation._clean_vars()

        # Simple model: y = 2 * x, z = 3 * y
        # This creates a chain: x (exog) -> y -> z
        x = Variable('x', 10, 'Input', scalar=True)
        y = Variable('y', [], 'Intermediate', scalar=True)
        z = Variable('z', [], 'Output', scalar=True)

        eq1 = Equation('eq_y', '2 * x', y, x)
        eq2 = Equation('eq_z', '3 * y', z, y)

        model = Model(t_length=5, add_all=True)
        return model

    def setup_circular_model(self):
        """Create a scalar model with circular dependencies."""
        Variable._clean_vars()
        Equation._clean_vars()

        # Circular model: C depends on W, W depends on X, X depends on C
        # Plus an exogenous K
        K = Variable('K', 100, 'Capital', scalar=True)
        C = Variable('C', 50, 'Consumption', scalar=True)
        W = Variable('W', 30, 'Wages', scalar=True)
        X = Variable('X', 80, 'Output', scalar=True)

        # C = 0.8 * W + 10
        eq_C = Equation('eq_C', '0.8 * W + 10', C, W)
        # W = 0.5 * X
        eq_W = Equation('eq_W', '0.5 * X', W, X)
        # X = C + 0.1 * K
        eq_X = Equation('eq_X', 'C + 0.1 * K', X, C, K)

        model = Model(t_length=5, add_all=True)
        return model

    def setup_matrix_model(self):
        """Create a model with matrix variables (2-sector)."""
        Variable._clean_vars()
        Equation._clean_vars()

        # 2-sector model
        a = Variable('a', [[0.1, 0.2], [0.15, 0.1]], 'IO coefficients', scalar=False)
        y = Variable('y', [[100], [80]], 'Final demand', scalar=False)
        x = Variable('x', [], 'Output', scalar=False)

        # x = (I - a)^-1 * y
        eq_x = Equation('eq_x', 'inv(eye(2) - a) * y', x, a, y)

        model = Model(t_length=5, add_all=True)
        return model

    # ==================== Phase 1: Foundation Tests ====================

    def test_equation_properties(self):
        """Test Equation.input_names and output_name properties."""
        model = self.setup_simple_model()

        eq = model.equations['eq_y']

        # Test input_names
        input_names = eq.input_names
        has_input_names = 'x' in input_names

        # Test output_name
        output_name = eq.output_name
        correct_output = output_name == 'y'

        passed = has_input_names and correct_output
        msg = f"input_names={input_names}, output_name={output_name}"
        return self._report('equation_properties', passed, msg)

    def test_dependency_graph(self):
        """Test Model.dependency_graph() method."""
        model = self.setup_simple_model()

        graph = model.dependency_graph()

        # Should have entries for y and z
        has_y = 'y' in graph and 'x' in graph['y']
        has_z = 'z' in graph and 'y' in graph['z']

        passed = has_y and has_z
        msg = f"graph={graph}"
        return self._report('dependency_graph', passed, msg)

    def test_classify_variables(self):
        """Test Model.classify_variables() method."""
        model = self.setup_simple_model()

        classification = model.classify_variables()

        # x should be exogenous (used but no equation)
        # y, z should be endogenous (have equations)
        x_exog = 'x' in classification['exogenous']
        y_endog = 'y' in classification['endogenous']
        z_endog = 'z' in classification['endogenous']

        passed = x_exog and y_endog and z_endog
        msg = f"classification={classification}"
        return self._report('classify_variables', passed, msg)

    def test_dependency_matrix(self):
        """Test Model.dependency_matrix() method."""
        model = self.setup_simple_model()

        dep_matrix = model.dependency_matrix()

        # Should be a DataFrame with equations as rows, variables as columns
        is_df = isinstance(dep_matrix, pd.DataFrame)
        correct_shape = dep_matrix.shape[0] == 2  # 2 equations

        # eq_y should have x=1 (input), y=-1 (output)
        eq_y_correct = dep_matrix.loc['eq_y', 'x'] == 1 and dep_matrix.loc['eq_y', 'y'] == -1

        passed = is_df and correct_shape and eq_y_correct
        msg = f"shape={dep_matrix.shape}, eq_y_x={dep_matrix.loc['eq_y', 'x']}, eq_y_y={dep_matrix.loc['eq_y', 'y']}"
        return self._report('dependency_matrix', passed, msg)

    # ==================== Phase 2: Analysis Tests ====================

    def test_check_determination(self):
        """Test Model.check_determination() method."""
        model = self.setup_simple_model()

        result = model.check_determination()

        # Simple model should be exactly determined (2 equations, 2 endogenous)
        is_determined = result['determined']
        correct_status = result['status'] == 'exactly_determined'
        correct_counts = result['n_equations'] == 2 and result['n_endogenous'] == 2

        passed = is_determined and correct_status and correct_counts
        msg = f"result={result}"
        return self._report('check_determination', passed, msg)

    def test_find_circular_dependencies_none(self):
        """Test find_circular_dependencies with no cycles."""
        model = self.setup_simple_model()

        cycles = model.find_circular_dependencies()

        # Simple chain model should have no cycles
        passed = len(cycles) == 0
        msg = f"cycles={cycles}"
        return self._report('find_circular_dependencies_none', passed, msg)

    def test_find_circular_dependencies_cycle(self):
        """Test find_circular_dependencies with a cycle."""
        model = self.setup_circular_model()

        cycles = model.find_circular_dependencies()

        # Should find one cycle: C -> W -> X -> C
        has_cycle = len(cycles) >= 1

        # Check that the cycle contains the expected variables
        cycle_vars = set()
        for cycle in cycles:
            cycle_vars.update(cycle)

        expected_vars = {'C', 'W', 'X'}
        contains_expected = expected_vars.issubset(cycle_vars)

        passed = has_cycle and contains_expected
        msg = f"cycles={cycles}"
        return self._report('find_circular_dependencies_cycle', passed, msg)

    def test_jacobian_symbolic(self):
        """Test Model.jacobian_symbolic() method."""
        model = self.setup_simple_model()

        J = model.jacobian_symbolic()

        # Should be a SymPy matrix
        is_matrix = isinstance(J, (sp.Matrix, sp.ImmutableDenseMatrix))

        # Should be 2x2 (2 equations, 2 endogenous variables)
        correct_shape = J.shape == (2, 2)

        passed = is_matrix and correct_shape
        msg = f"J.shape={J.shape}, type={type(J)}"
        return self._report('jacobian_symbolic', passed, msg)

    def test_jacobian_numeric(self):
        """Test Model.jacobian_numeric() method."""
        model = self.setup_simple_model()

        J = model.jacobian_numeric()

        # Should be a numpy array
        is_array = isinstance(J, np.ndarray)

        # Should be 2x2
        correct_shape = J.shape == (2, 2)

        passed = is_array and correct_shape
        msg = f"J.shape={J.shape}, J={J}"
        return self._report('jacobian_numeric', passed, msg)

    def test_stability_analysis(self):
        """Test Model.stability_analysis() method."""
        model = self.setup_simple_model()

        result = model.stability_analysis()

        # Should have required keys
        has_keys = all(key in result for key in ['eigenvalues', 'stable', 'max_real'])

        # Eigenvalues should be an array
        is_array = isinstance(result['eigenvalues'], np.ndarray)

        passed = has_keys and is_array
        msg = f"result={result}"
        return self._report('stability_analysis', passed, msg)

    # ==================== Phase 3: Calibration Tests ====================

    def test_topological_order_exists(self):
        """Test topological_order when order exists."""
        model = self.setup_simple_model()

        order = model.topological_order()

        # Should return an order (list)
        is_list = isinstance(order, list)

        # y should come before z (since z depends on y)
        if order:
            y_idx = order.index('y')
            z_idx = order.index('z')
            correct_order = y_idx < z_idx
        else:
            correct_order = False

        passed = is_list and correct_order
        msg = f"order={order}"
        return self._report('topological_order_exists', passed, msg)

    def test_topological_order_cycle(self):
        """Test topological_order when cycles exist."""
        model = self.setup_circular_model()

        order = model.topological_order()

        # Should return None when cycles exist
        passed = order is None
        msg = f"order={order} (expected None)"
        return self._report('topological_order_cycle', passed, msg)

    def test_find_steady_state(self):
        """Test Model.find_steady_state() method."""
        model = self.setup_simple_model()

        result = model.find_steady_state(max_iter=100, tol=1e-8)

        # Should have required keys
        has_keys = all(key in result for key in ['converged', 'iterations', 'max_change', 'values'])

        # Simple model should converge quickly
        converged = result['converged']

        # Check values are reasonable
        values = result['values']
        has_values = 'y' in values and 'z' in values

        passed = has_keys and converged and has_values
        msg = f"converged={converged}, iterations={result['iterations']}, values={values}"
        return self._report('find_steady_state', passed, msg)

    def test_calibration_report(self):
        """Test Model.calibration_report() method."""
        model = self.setup_simple_model()

        # Capture print output (the method prints a report)
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            df = model.calibration_report()

        output = f.getvalue()

        # Should return a DataFrame
        is_df = isinstance(df, pd.DataFrame)

        # Should have required columns
        has_columns = all(col in df.columns for col in ['variable', 'type', 'value', 'in_cycle'])

        # Should print something
        has_output = 'CALIBRATION REPORT' in output

        passed = is_df and has_columns and has_output
        msg = f"DataFrame shape={df.shape}, columns={list(df.columns)}"
        return self._report('calibration_report', passed, msg)

    # ==================== Matrix Model Tests ====================

    def test_matrix_model_analysis(self):
        """Test analysis methods work with matrix variables."""
        model = self.setup_matrix_model()

        # Test classification
        classification = model.classify_variables()
        x_endog = 'x' in classification['endogenous']
        a_exog = 'a' in classification['exogenous']
        y_exog = 'y' in classification['exogenous']

        # Test determination check
        determination = model.check_determination()
        is_determined = determination['determined']

        passed = x_endog and a_exog and y_exog and is_determined
        msg = f"classification={classification}, determination={determination['status']}"
        return self._report('matrix_model_analysis', passed, msg)

    # ==================== Main Test Runner ====================

    def run_all(self, verbose=True):
        """Run all tests and return overall pass/fail."""
        self.results = {}

        tests = [
            # Phase 1: Foundation
            ("Equation Properties", self.test_equation_properties),
            ("Dependency Graph", self.test_dependency_graph),
            ("Classify Variables", self.test_classify_variables),
            ("Dependency Matrix", self.test_dependency_matrix),

            # Phase 2: Analysis
            ("Check Determination", self.test_check_determination),
            ("Find Circular (None)", self.test_find_circular_dependencies_none),
            ("Find Circular (Cycle)", self.test_find_circular_dependencies_cycle),
            ("Jacobian Symbolic", self.test_jacobian_symbolic),
            ("Jacobian Numeric", self.test_jacobian_numeric),
            ("Stability Analysis", self.test_stability_analysis),

            # Phase 3: Calibration
            ("Topological Order (Exists)", self.test_topological_order_exists),
            ("Topological Order (Cycle)", self.test_topological_order_cycle),
            ("Find Steady State", self.test_find_steady_state),
            ("Calibration Report", self.test_calibration_report),

            # Matrix model
            ("Matrix Model Analysis", self.test_matrix_model_analysis),
        ]

        all_passed = True

        if verbose:
            print("\n" + "=" * 60)
            print("MODEL ANALYSIS & CALIBRATION TESTS")
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
    tests = AnalysisTests()
    tests.run_all(verbose=True)
