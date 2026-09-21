"""Recompute contract-failure rate (NO_ANSWER + NON_NUMERIC) from a Phase 0
prediction file. Used to fill the Phase 1 prereg S2 floor.

Usage:
    uv run python scripts/compute_contract_failure_rate.py results/raw/Qwen3-8B__cot__finqa_test.jsonl --expect-n 1127 --expect-acc 0.6557
"""
import json
import collections
import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--expect-n", type=int, default=None)
    p.add_argument("--expect-acc", type=float, default=None)
    p.add_argument("--acc-tol", type=float, default=0.005)
    args = p.parse_args()

    counts = collections.Counter()
    n = 0
    correct = 0
    degenerate = 0

    with open(args.path) as f:
        for line in f:
            r = json.loads(line)
            counts[r["status"]] += 1
            correct += r["correct"]
            n += 1
            if r["output"].count("ANSWER:") > 2:
                degenerate += 1

    if args.expect_n is not None and n != args.expect_n:
        sys.exit(f"FAIL: expected {args.expect_n} rows, got {n}")

    acc = correct / n
    if args.expect_acc is not None and abs(acc - args.expect_acc) > args.acc_tol:
        sys.exit(f"FAIL: accuracy {acc:.4f} not within {args.acc_tol} of expected {args.expect_acc}")

    fail = counts.get("no_answer", 0) + counts.get("non_numeric", 0)

    print(f"file: {args.path}")
    print(f"n: {n}")
    print(f"status counts: {dict(counts)}")
    print(f"accuracy: {100*acc:.2f}%")
    print(f"contract-failure rate: {100*fail/n:.2f}% ({fail}/{n})")
    print(f"degenerate-repetition rate: {100*degenerate/n:.2f}% ({degenerate}/{n})")


if __name__ == "__main__":
    main()
