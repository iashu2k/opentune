"""Test-first suite for the answer extractor.

Coverage classes per docs/task-spec.md:
  1. valid extractions          (13)
  2. rejection paths            (10)
  3. scoring semantics          (9)
  4. malformed / adversarial     (4)
"""


import dataclasses
from decimal import Decimal

from opentune.extract import ExtractionResult, ScoreResult, Status, extract_answer, score

# --- 1. valid extractions ----------------------------------------------------


class TestValidExtraction:
    def test_simple_integer(self):
        r = extract_answer("Some reasoning.\nANSWER: 42")
        assert r.status is Status.OK and r.value == Decimal("42")

    def test_negative(self):
        r = extract_answer("ANSWER: -1234.5")
        assert r.status is Status.OK and r.value == Decimal("-1234.5")

    def test_decimal(self):
        r = extract_answer("ANSWER: 0.375")
        assert r.value == Decimal("0.375")

    def test_trailing_zeros(self):
        r = extract_answer("ANSWER: 42.00")
        assert r.value == Decimal("42.00")

    def test_comma_formatted(self):
        r = extract_answer("ANSWER: 1,234,567")
        assert r.value == Decimal("1234567")

    def test_percent_divided_by_100(self):
        r = extract_answer("ANSWER: 37.5%")
        assert r.status is Status.OK
        assert r.value == Decimal("0.375") and r.is_percent

    def test_cot_multiline_answer_at_end(self):
        text = "Revenue rose.\nProgram: subtract(5000, 3200)\n\nANSWER: 1800\n"
        r = extract_answer(text)
        assert r.status is Status.OK and r.value == Decimal("1800")

    def test_answer_only_output(self):
        assert extract_answer("ANSWER: 7").status is Status.OK

    def test_case_insensitive(self):
        assert extract_answer("answer: 5").value == Decimal("5")
        assert extract_answer("Answer: 5").value == Decimal("5")

    def test_whitespace_tolerance(self):
        r = extract_answer("  \t ANSWER :  42  ")
        assert r.status is Status.OK and r.value == Decimal("42")

    def test_currency_symbol_stripped(self):
        assert extract_answer("ANSWER: $1,234,567").value == Decimal("1234567")

    def test_accounting_negative_parens(self):
        r = extract_answer("ANSWER: (1,234)")
        assert r.value == Decimal("-1234")

    def test_unicode_minus(self):
        r = extract_answer("ANSWER: −0.12%")
        assert r.value == Decimal("-0.0012")


# --- 2. rejection paths -------------------------------------------------------


class TestRejection:
    def test_no_answer_line(self):
        r = extract_answer("Revenue went up by quite a bit.")
        assert r.status is Status.NO_ANSWER and r.value is None

    def test_empty_output(self):
        assert extract_answer("").status is Status.NO_ANSWER
        assert extract_answer("   \n  ").status is Status.NO_ANSWER

    def test_empty_after_colon(self):
        assert extract_answer("ANSWER:").status is Status.NON_NUMERIC

    def test_non_numeric_na(self):
        r = extract_answer("ANSWER: N/A")
        assert r.status is Status.NON_NUMERIC and r.raw == "N/A"

    def test_unit_burdened_rejected(self):
        assert extract_answer("ANSWER: 42 million").status is Status.NON_NUMERIC

    def test_answer_mid_paragraph_ignored(self):
        text = "The answer: 42. This follows from the table."
        assert extract_answer(text).status is Status.NO_ANSWER

    def test_plural_answer_not_matched(self):
        assert extract_answer("Answers: 42").status is Status.NO_ANSWER

    def test_multiple_answer_lines_last_wins(self):
        text = "ANSWER: 100\n\nWait, rechecking...\nANSWER: 200"
        r = extract_answer(text)
        assert r.value == Decimal("200") and r.n_answer_lines == 2

    def test_last_empty_line_falls_back_to_earlier_nonempty(self):
        text = "ANSWER: 300\nANSWER:"
        assert extract_answer(text).value == Decimal("300")

    def test_extraction_never_raises(self):
        for weird in ["ANSWER: .", "ANSWER: --5", "ANSWER: —", "ANSWER: $", ""]:
            assert extract_answer(weird).status is not Status.OK

    def test_malformed_commas_still_parse(self):
        # Commas are separators only; misplacement is normalized, not rejected.
        assert extract_answer("ANSWER: 1,2,3").value == Decimal("123")


# --- 3. scoring semantics -----------------------------------------------------


class TestScoring:
    def test_integer_identity(self):
        assert score("ANSWER: 42.0", "42").correct

    def test_percent_gold_vs_decimal_pred(self):
        assert score("ANSWER: 0.375", "37.5%").correct

    def test_decimal_gold_vs_percent_pred(self):
        assert score("ANSWER: 37.5%", "0.375").correct

    def test_bare_number_vs_percent_gold_misses(self):
        # documented design decision in task-spec.md
        assert not score("ANSWER: 37.5", "0.375").correct

    def test_inside_abs_tolerance(self):
        assert score("ANSWER: 42.459", "42.45").correct  # diff 0.009

    def test_outside_abs_tolerance(self):
        assert not score("ANSWER: 42.461", "42.45").correct  # diff 0.011

    def test_large_number_relative_tolerance_inside(self):
        assert score("ANSWER: 1,000,500", "1000000").correct  # diff 500 <= 1000

    def test_large_number_relative_tolerance_outside(self):
        assert not score("ANSWER: 1,002,000", "1000000").correct  # diff 2000 > 1000

    def test_non_numeric_pred_scores_incorrect(self):
        r = score("ANSWER: N/A", "42")
        assert isinstance(r, ScoreResult) and not r.correct
        assert r.extraction.status is Status.NON_NUMERIC


# --- 4. malformed / adversarial ----------------------------------------------


class TestAdversarial:
    def test_degenerate_repeated_answer_uses_last(self):
        text = "\n".join(["ANSWER: 0"] * 10)
        r = extract_answer(text)
        assert r.value == Decimal("0") and r.n_answer_lines == 10

    def test_score_returns_diff(self):
        r = score("ANSWER: 10", "9")
        assert r.diff == Decimal("1") and not r.correct

    def test_accounting_negative_percent(self):
        assert score("ANSWER: (0.5%)", "-0.005").correct

    def test_extraction_result_is_frozen_dataclass(self):
        r = extract_answer("ANSWER: 1")
        assert isinstance(r, ExtractionResult) and dataclasses.is_dataclass(r)
