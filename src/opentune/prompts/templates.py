"""Prompt templates for OpenTune baselines.

Output contract (docs/task-spec.md v1.1) is identical across all arms and is
what Phase 2's GRPO format reward will parse:

    <reasoning>
    Program: op(arg, arg), op(arg, arg)
    ANSWER: <number>

Few-shot exemplars MUST come from the decontaminated TRAIN set and be
hand-verified (compute the program; confirm it equals the gold).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_FEWSHOT_PATH = Path(__file__).parent / "fewshot.json"

CONTRACT = """Rules:
1. Reason step by step, referencing specific figures from the context.
2. Then write one line: Program: op(arg, arg), op(arg, arg), ...
   using FinQA-style operators (add, subtract, multiply, divide, greater,
   exp, power). Refer to earlier results as #0, #1, ...
3. Final line must be exactly: ANSWER: <number>
   The number may use %, $, commas, or parentheses for negatives.
   No units, no words, nothing after the number.
4. Exactly one ANSWER line, and it must be the last line you write."""

ZERO_SHOT_TEMPLATE = """You are a financial QA assistant. Answer the question from the report excerpt.

### Context
{context}

### Question
{question}

{contract}

### Response"""

COT_TEMPLATE = """You are a financial QA assistant with expert table-reading skills. Work through the question methodically before answering.

### Context
{context}

### Question
{question}

{contract}

### Response
Let me work through this step by step."""

FEW_SHOT_TEMPLATE = """You are a financial QA assistant. Study the examples, then answer the new question following the same format exactly.

{exemplars}

### New Context
{context}

### New Question
{question}

{contract}

### Response"""


@dataclass(frozen=True)
class Exemplar:
  id: str  # train-set id of origin (e.g. "train-1182")
  context: str
  question: str
  solution: str  # reasoning + Program + ANSWER, hand-verified


def load_fewshot() -> list[Exemplar]:
  with open(_FEWSHOT_PATH) as f:
    return [Exemplar(**e) for e in json.load(f)]


def _format_exemplars(exemplars: list[Exemplar]) -> str:
  blocks = []
  for e in exemplars:
    blocks.append(
        f"### Context\n{e.context}\n\n### Question\n{e.question}\n\n### Response\n{e.solution}"
    )
  return "\n\n---\n\n".join(blocks)


TEMPLATES = {
    "zero_shot": ZERO_SHOT_TEMPLATE,
    "cot": COT_TEMPLATE,
    "few_shot": FEW_SHOT_TEMPLATE,
}


def render_prompt(name: str, question: str, context: str) -> str:
  """Render a baseline prompt. Every eval harness call goes through here so
  prompts are reproducible from git history alone."""
  if name == "few_shot":
    return FEW_SHOT_TEMPLATE.format(
        exemplars=_format_exemplars(load_fewshot()),
        context=context,
        question=question,
        contract=CONTRACT,
    )
  return TEMPLATES[name].format(context=context, question=question, contract=CONTRACT)
