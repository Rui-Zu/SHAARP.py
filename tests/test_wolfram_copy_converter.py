import unittest

from benchmarks.convert_wolfram_copied_output import extract_column, parse_wolfram_real_list, upsert_case_output


class WolframCopyConverterTests(unittest.TestCase):
    def test_parse_wolfram_real_list(self):
        parsed = parse_wolfram_real_list("{{0., 1.2*^-3}, {10.`20., -4.5}}")
        self.assertEqual(parsed, [[0.0, 0.0012], [10.0, -4.5]])

    def test_extract_column(self):
        self.assertEqual(extract_column([[0.0, 1.0], [10.0, 2.5]], 1), [1.0, 2.5])

    def test_upsert_case_output(self):
        payload = {"cases": [{"id": "case_001", "outputs": {"a": [1.0]}}]}
        upsert_case_output(payload, "case_001", "b", [2.0])
        upsert_case_output(payload, "case_002", "a", [3.0])
        self.assertEqual(payload["cases"][0]["outputs"]["b"], [2.0])
        self.assertEqual(payload["cases"][1]["id"], "case_002")

    def test_complex_lists_are_rejected_by_real_converter(self):
        with self.assertRaises(ValueError):
            parse_wolfram_real_list("{1 + 2 I}")


if __name__ == "__main__":
    unittest.main()
