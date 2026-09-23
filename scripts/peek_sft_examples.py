
import json

with open("data/processed/sft_train_v1.jsonl") as f:
    lines = [json.loads(l) for l in f]

print(f"Total examples: {len(lines)}\n")
for ex in lines[:2]:
    print("=" * 80)
    print(f"id: {ex['id']}")
    print("--- USER MESSAGE (rendered prompt) ---")
    print(ex["messages"][0]["content"][:1500])
    print("...\n" if len(ex["messages"][0]["content"]) > 1500 else "")
    print("--- ASSISTANT MESSAGE (SFT target) ---")
    print(ex["messages"][1]["content"])
    print()
