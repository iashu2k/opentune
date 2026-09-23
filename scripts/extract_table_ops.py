
import pandas as pd

df = pd.read_parquet("data/processed/finqa_train_decontaminated.parquet")

ops = ["table_average", "table_max", "table_min", "table_sum"]
for op in ops:
    matches = df[df["program"].str.contains(op, na=False)]
    print(f"--- {op}: {len(matches)} rows ---")
    for _, row in matches.head(3).iterrows():
        print(f"id={row['id']}")
        print(f"question={row['question']}")
        print(f"program={row['program']}")
        print(f"gold={row['gold']}")
        print()
