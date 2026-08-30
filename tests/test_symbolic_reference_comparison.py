import unittest

from benchmarks.compare_symbolic_reference import compare_symbolic_payloads


class SymbolicReferenceComparisonTests(unittest.TestCase):
    def test_symbolic_comparator_accepts_algebraically_equivalent_different_strings(self):
        reference = _reference_payload("x**2 + 2*x + 1")
        python = _python_payload("(x + 1)**2")

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)

        self.assertTrue(all(result.passed for result in results))
        self.assertIn("symbolic_residual", {result.check for result in results})

    def test_symbolic_comparator_parses_mathematica_input_form_function_syntax(self):
        reference = _reference_payload("Plus[Power[x, 2], Times[2, x], 1]", expression_format="mathematica_input_form")
        python = _python_payload("(x + 1)**2")

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)

        self.assertTrue(all(result.passed for result in results))

    def test_symbolic_comparator_parses_mathematica_input_form_infix_syntax(self):
        reference = _reference_payload("x^2 + 2*x + 1", expression_format="mathematica_input_form")
        python = _python_payload("(x + 1)**2")

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)

        self.assertTrue(all(result.passed for result in results))

    def test_symbolic_comparator_catches_numeric_difference_when_symbolic_check_skipped(self):
        reference = _reference_payload("x**2 + 2*x + 1")
        python = _python_payload("(x + 1)**2 + 0.01")

        results = compare_symbolic_payloads(
            reference,
            python,
            atol=1e-12,
            rtol=1e-12,
            check_symbolic_residual=False,
        )
        failed = [result for result in results if not result.passed]

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].check, "numeric_substitution[0]")
        self.assertGreater(failed[0].max_abs_error, 0.009)

    def test_symbolic_comparator_catches_symbol_role_mismatch(self):
        reference = _reference_payload("x + y")
        python = _python_payload("x + y")
        python["expressions"][0]["symbol_roles"]["y"] = "wrong_role"

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)
        failed = [result for result in results if not result.passed]

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].check, "symbol_roles")

    def test_symbolic_comparator_catches_vector_shape_mismatch(self):
        reference = _reference_payload(["x", "y"])
        python = _python_payload(["x", "y", "z"])
        python["expressions"][0]["symbol_roles"]["z"] = "extra"

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)
        failed = [result for result in results if not result.passed]

        self.assertTrue(any(result.check == "shape" for result in failed))

    def test_symbolic_comparator_rejects_placeholder_status(self):
        reference = _reference_payload("x + 1")
        reference["status"] = "mathematica_symbolic_reference_pending"

        with self.assertRaisesRegex(ValueError, "pending"):
            compare_symbolic_payloads(reference, _python_payload("x + 1"), atol=1e-12, rtol=1e-12)

    def test_symbolic_comparator_accepts_matching_singular_diagnostics(self):
        reference = _singular_reference_payload("n2w - nw")
        python = _singular_python_payload("-(nw - n2w)")

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)

        self.assertTrue(all(result.passed for result in results))
        self.assertIn("singular_diagnostics.denominator_factor", {result.check for result in results})

    def test_symbolic_comparator_catches_singular_classification_mismatch(self):
        reference = _singular_reference_payload("n2w - nw")
        python = _singular_python_payload("n2w - nw")
        python["expressions"][0]["singular_diagnostics"]["classification"] = "regular"

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)
        failed = [result for result in results if not result.passed]

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].check, "singular_diagnostics.classification")

    def test_symbolic_comparator_rejects_missing_singular_diagnostics(self):
        reference = _singular_reference_payload("n2w - nw")
        reference["expressions"][0].pop("singular_diagnostics")

        with self.assertRaisesRegex(ValueError, "singular_diagnostics"):
            compare_symbolic_payloads(reference, _singular_python_payload("n2w - nw"), atol=1e-12, rtol=1e-12)


def _reference_payload(expression, *, expression_format="sympy"):
    return {
        "source": "Mathematica SHAARP symbolic export",
        "status": "mathematica_symbolic_reference_exported",
        "expressions": [_record(expression, expression_format=expression_format)],
    }


def _python_payload(expression, *, expression_format="sympy"):
    return {
        "source": "Python SymPy symbolic output",
        "status": "python_symbolic_output",
        "expressions": [_record(expression, expression_format=expression_format)],
    }


def _record(expression, *, expression_format="sympy"):
    return {
        "expression_id": "expr_001",
        "source_stage": "P2eo",
        "output_component": "x",
        "expression_format": expression_format,
        "expression": expression,
        "symbol_roles": {
            "x": "omega_field_x",
            "y": "omega_field_y",
        },
        "numeric_substitutions": [
            {
                "x": {"real": 0.25, "imag": 0.1},
                "y": {"real": -0.4, "imag": 0.2},
            }
        ],
    }


def _singular_reference_payload(denominator_factor):
    return {
        "source": "Mathematica SHAARP symbolic export",
        "status": "mathematica_symbolic_reference_exported",
        "expressions": [_singular_record(denominator_factor)],
    }


def _singular_python_payload(denominator_factor):
    return {
        "source": "Python SymPy symbolic output",
        "status": "python_symbolic_output",
        "expressions": [_singular_record(denominator_factor)],
    }


def _singular_record(denominator_factor):
    return {
        "expression_id": "phase_match_001",
        "source_stage": "solveInhom_ee",
        "output_component": "diagnostic",
        "comparison_mode": "singular_diagnostics",
        "symbol_roles": {
            "nw": "omega_refractive_index",
            "n2w": "two_omega_refractive_index",
        },
        "singular_diagnostics": {
            "classification": "phase_matched_singular",
            "denominator_factor": denominator_factor,
            "diagnostic_expression_format": "sympy",
            "condition_marker": "zero_denominator",
            "condition_number": 1.0e16,
        },
    }


if __name__ == "__main__":
    unittest.main()
