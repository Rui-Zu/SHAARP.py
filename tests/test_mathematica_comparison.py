import unittest

from benchmarks.compare_mathematica_reference import compare_payloads


class MathematicaComparisonTests(unittest.TestCase):
    def test_compare_payloads_checks_numeric_values_with_tolerance(self):
        mathematica = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "single_interface_intensity": [1.0, 2.0, 3.0],
                    },
                }
            ]
        }
        python = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "single_interface_intensity": [1.0 + 1e-10, 2.0, 3.0],
                    },
                }
            ]
        }
        results = compare_payloads(
            mathematica,
            python,
            keys=["single_interface_intensity"],
            atol=1e-8,
            rtol=1e-8,
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)
        self.assertGreater(results[0].max_abs_error, 0.0)

    def test_compare_payloads_fails_value_mismatch(self):
        mathematica = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "single_interface_intensity": [1.0, 2.0, 3.0],
                    },
                }
            ]
        }
        python = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "single_interface_intensity": [1.0, 2.5, 3.0],
                    },
                }
            ]
        }
        results = compare_payloads(
            mathematica,
            python,
            keys=["single_interface_intensity"],
            atol=1e-8,
            rtol=1e-8,
        )
        self.assertFalse(results[0].passed)
        self.assertAlmostEqual(results[0].max_abs_error, 0.5)

    def test_compare_payloads_accepts_complex_real_imag_objects(self):
        mathematica = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": [{"real": 1.0, "imag": -0.5}],
                    },
                }
            ]
        }
        python = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": [{"real": 1.0, "imag": -0.5 + 1e-10}],
                    },
                }
            ]
        }
        results = compare_payloads(mathematica, python, keys=["field"], atol=1e-8, rtol=1e-8)
        self.assertTrue(results[0].passed)

    def test_strict_compare_requires_mathematica_source_metadata(self):
        mathematica = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": [1.0],
                    },
                }
            ]
        }
        python = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": [1.0],
                    },
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "source metadata"):
            compare_payloads(mathematica, python, keys=["field"], atol=1e-8, rtol=1e-8, strict_reference=True)

    def test_strict_compare_rejects_python_regression_as_reference(self):
        mathematica = {
            "source": "Python benchmark generator",
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": [1.0],
                    },
                }
            ],
        }
        python = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": [1.0],
                    },
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "Mathematica or Wolfram"):
            compare_payloads(mathematica, python, keys=["field"], atol=1e-8, rtol=1e-8, strict_reference=True)

    def test_strict_compare_rejects_unfilled_reference_placeholders(self):
        mathematica = {
            "source": "Mathematica SHAARP",
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": None,
                    },
                }
            ],
        }
        python = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": [1.0],
                    },
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "Unfilled Mathematica output placeholder"):
            compare_payloads(mathematica, python, keys=["field"], atol=1e-8, rtol=1e-8, strict_reference=True)

    def test_strict_compare_accepts_filled_mathematica_reference(self):
        mathematica = {
            "source": "Mathematica SHAARP",
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": [{"real": 1.0, "imag": 0.2}],
                    },
                }
            ],
        }
        python = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": [{"real": 1.0, "imag": 0.2 + 1e-10}],
                    },
                }
            ]
        }

        results = compare_payloads(mathematica, python, keys=["field"], atol=1e-8, rtol=1e-8, strict_reference=True)
        self.assertTrue(results[0].passed)

    def test_compare_payloads_accepts_suite_template_shape_with_suite_id(self):
        mathematica = {
            "source": "Mathematica SHAARP",
            "suites": [
                {
                    "suite_id": "solver_suite",
                    "items": [
                        {
                            "id": "case_001",
                            "mathematica_outputs": {
                                "field": [{"real": 1.0, "imag": -0.25}],
                            },
                        }
                    ],
                }
            ],
        }
        python = {
            "cases": [
                {
                    "id": "case_001",
                    "outputs": {
                        "field": [{"real": 1.0, "imag": -0.25 + 1e-10}],
                    },
                }
            ]
        }

        results = compare_payloads(
            mathematica,
            python,
            keys=["field"],
            atol=1e-8,
            rtol=1e-8,
            strict_reference=True,
            suite_id="solver_suite",
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)

    def test_suite_template_shape_requires_suite_id_when_ambiguous(self):
        mathematica = {
            "source": "Mathematica SHAARP",
            "suites": [
                {"suite_id": "one", "items": [{"id": "case_001", "mathematica_outputs": {"field": [1.0]}}]},
                {"suite_id": "two", "items": [{"id": "case_001", "mathematica_outputs": {"field": [1.0]}}]},
            ],
        }
        python = {"cases": [{"id": "case_001", "outputs": {"field": [1.0]}}]}

        with self.assertRaisesRegex(ValueError, "pass --suite-id"):
            compare_payloads(mathematica, python, keys=["field"], atol=1e-8, rtol=1e-8, strict_reference=True)

    def test_strict_suite_template_rejects_unfilled_mathematica_outputs(self):
        mathematica = {
            "source": "Mathematica SHAARP",
            "suites": [
                {
                    "suite_id": "solver_suite",
                    "items": [
                        {
                            "id": "case_001",
                            "mathematica_outputs": {
                                "field": None,
                            },
                        }
                    ],
                }
            ],
        }
        python = {"cases": [{"id": "case_001", "outputs": {"field": [1.0]}}]}

        with self.assertRaisesRegex(ValueError, "Unfilled Mathematica output placeholder"):
            compare_payloads(
                mathematica,
                python,
                keys=["field"],
                atol=1e-8,
                rtol=1e-8,
                strict_reference=True,
                suite_id="solver_suite",
            )


if __name__ == "__main__":
    unittest.main()
