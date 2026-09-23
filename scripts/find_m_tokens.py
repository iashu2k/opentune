
import pandas as pd

df = pd.read_parquet("data/processed/finqa_train_decontaminated.parquet")

matches = df[df["program"].str.contains(r"m\d", regex=True, na=False)]
print(f"Rows with m<digit> tokens: {len(matches)}")
for _, row in matches.head(5).iterrows():
    print(f"id={row['id']}")
    print(f"question={row['question']}")
    print(f"program={row['program']}")
    print(f"gold={row['gold']}")
    print()
