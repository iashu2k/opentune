"""Build the time-split SEC eval from EDGAR XBRL company facts.

Anti-contamination construction: every fact used has filed >= 2026-01-01,
i.e. reported in filings published after Qwen3-8B's release (2025-04-29) and
after plausible frontier-model cutoffs. Golds are COMPUTED from XBRL values —
never LLM-generated. Contexts are condensed fact tables (limitation noted in
dataset card): this set tests reasoning over post-cutoff SEC data, not raw
filing-HTML reading.

Run: uv run python scripts/build_sec_eval.py
Writes: data/eval/sec_2026_eval.parquet (commit), data/eval/sec_2026_report.json
"""

import json
import random
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Ashutosh Research iashu2k@gmail.com"}
FILED_AFTER = "2026-01-01"
OUT_DIR = Path("data/eval")
N_COMPANIES = 60
N_TARGET = 180
SEED = 7
PAUSE = 0.2  # EDGAR rate limit: stay well under 10 req/s

CONCEPTS = {
    "Revenues": "total revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "NetIncomeLoss": "net income",
    "OperatingIncomeLoss": "operating income",
    "Assets": "total assets",
    "CashAndCashEquivalentsAtCarryingValue": "cash and cash equivalents",
}


def get(url: str) -> dict:
  time.sleep(PAUSE)
  r = requests.get(url, headers=UA, timeout=30)
  r.raise_for_status()
  return r.json()


def quarterly_usd_facts(facts: dict, concept: str) -> list[dict]:
  node = facts.get("facts", {}).get("us-gaap", {}).get(concept, {})
  out = []
  for e in node.get("units", {}).get("USD", []):
    if e.get("frame", "").startswith("CY202") and e.get("start") and \
       e.get("form") in ("10-Q", "10-K") and e.get("filed", "") >= FILED_AFTER:
      out.append(e)
  return out


def build_candidates(cik: int, name: str) -> list[dict]:
  try:
    facts = get(
      f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
  except Exception:
    return []
  cands = []

  # QoQ percent change within 2026 filings
  for concept, label in CONCEPTS.items():
    fs = quarterly_usd_facts(facts, concept)
    fs = sorted({(f["end"], f["val"]): f for f in fs}.values(),
                key=lambda f: f["end"])
    for a, b in zip(fs, fs[1:]):
      if a["val"] and b["val"] and abs(a["val"]) > 0:
        gold = (b["val"] - a["val"]) / abs(a["val"])
        if 0.0005 < abs(gold) < 10:
          cands.append({
              "company": name,
              "question": f"what was the percentage change in {label} "
              f"from the period ended {a['end']} to the period ended {b['end']}?",
              "concepts": [(label, a["end"], a["val"]), (label, b["end"], b["val"])],
              "program": f"subtract({b['val']}, {a['val']}), divide(#0, {a['val']})",
              "gold": round(gold, 5),
              "filed": b["filed"],
          })

  # Portion: cash / assets, same period
  cash = {f["end"]: f for f in quarterly_usd_facts(
    facts, "CashAndCashEquivalentsAtCarryingValue")}
  assets = {f["end"]: f for f in quarterly_usd_facts(facts, "Assets")}
  for end in sorted(set(cash) & set(assets)):
    c, a = cash[end], assets[end]
    if a["val"] and 0.001 < c["val"] / a["val"] < 0.95:
      cands.append({
          "company": name,
          "question": f"what portion of total assets was held as cash and cash "
          f"equivalents as of {end}?",
          "concepts": [("cash and cash equivalents", end, c["val"]), ("total assets", end, a["val"])],
          "program": f"divide({c['val']}, {a['val']})",
          "gold": round(c["val"] / a["val"], 5),
          "filed": c["filed"],
      })
  return cands


def render_context(c: dict) -> str:
  head = f"condensed financial facts from {c['company']} 2026 sec filing (xbrl-derived)."
  rows = ["item | period ended | value (usd)"]
  rows += [f"{lab} | {end} | {val}" for lab, end, val in c["concepts"]]
  return head + "\n" + "\n".join(rows)


def main() -> None:
  tickers = get("https://www.sec.gov/files/company_tickers.json")
  universe = [(int(v["cik_str"]), v["title"]) for v in tickers.values()]
  rng = random.Random(SEED)
  rng.shuffle(universe)

  cands: list[dict] = []
  for cik, name in universe[:N_COMPANIES]:
    got = build_candidates(cik, name)
    cands.extend(got)
    print(f"{name[:30]:32s} +{len(got)}")

  rng.shuffle(cands)
  cands = cands[:N_TARGET] if len(cands) > N_TARGET else cands

  records = [
      {
          "id": f"sec2026-{i}",
          "split": "sec_2026",
          "question": c["question"],
          "context": render_context(c),
          "gold": str(c["gold"]),
          "gold_source": "xbrl_computed",
          "program": c["program"],
          "company": c["company"],
          "filed": c["filed"],
      }
      for i, c in enumerate(cands)
  ]
  df = pd.DataFrame(records)
  df.to_parquet(OUT_DIR / "sec_2026_eval.parquet", index=False)

  report = {
      "built": date.today().isoformat(),
      "filed_after": FILED_AFTER,
      "n": len(df),
      "companies": int(df["company"].nunique()),
      "filed_range": [str(df["filed"].min()), str(df["filed"].max())],
  }
  (OUT_DIR / "sec_2026_report.json").write_text(json.dumps(report, indent=2))
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
