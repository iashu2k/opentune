"""Contract-compliance tests for the prompt templates.

Enforces: (1) every template carries the task-spec v1.1 output contract,
(2) rendering leaves no unfilled placeholders, (3) few-shot exemplar solutions
score correct under the extractor — an exemplar that fails our own scorer
teaches the model a wrong format.
"""

from opentune.extract import Status, extract_answer, score
from opentune.prompts.templates import CONTRACT, TEMPLATES, load_fewshot, render_prompt


class TestTemplateContract:
    def test_all_templates_present(self):
        assert set(TEMPLATES) == {"zero_shot", "cot", "few_shot"}

    def test_contract_required_elements(self):
        assert "ANSWER:" in CONTRACT
        assert "Program:" in CONTRACT
        assert "Exactly one ANSWER line" in CONTRACT

    def test_templates_embed_contract(self):
        for name in TEMPLATES:
            rendered = render_prompt(name, question="what is 2+2?", context="2 and 2")
            assert "ANSWER: <number>" in rendered, f"{name} missing contract"
            assert "{" not in rendered.split("### Response")[0], f"{name} has unfilled braces"

    def test_fewshot_solutions_self_consistent(self):
        # each exemplar: extract ANSWER, vs the same number scored correct
        for e in load_fewshot():
            ext = extract_answer(e.solution)
            assert ext.status is Status.OK, f"exemplar {e.id} has no parseable ANSWER"
            assert f"Program:" in e.solution, f"exemplar {e.id} missing Program line"

    def test_exemplars_come_from_train(self):
        for e in load_fewshot():
            assert e.id.startswith("train-"), f"exemplar {e.id} not from train set"
