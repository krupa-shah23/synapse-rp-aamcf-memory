"""Show exactly what the DistilBERT gate is given: the first rows of the units
table (group, key, text, label) for each dataset and label type, plus one
tokenised example. Run from packages/aamcf-data:
    uv run python scripts/inspect/peek_training_rows.py
"""

import pandas as pd
from transformers import AutoTokenizer

from aamcf_data.gate.dataset import LME, LOCOMO, load_units, make_windows

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
for name, cfg in [("LoCoMo", LOCOMO), ("LongMemEval_S", LME)]:
    gold = load_units(cfg, "label_supervised")
    proxy = load_units(cfg, "label_overlap")
    print(f"\n===== {name}: {len(gold)} rows | columns the model reads: {list(gold.columns)} =====")
    show = gold.copy()
    show["text"] = show["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 70)
    show = show.rename(columns={"label": "gold_label"})
    show["proxy_label"] = proxy["label"]
    print(show.head(8).to_string())
    print("gold positives:", int(gold["label"].sum()), f"({gold['label'].mean():.1%})", "| proxy positives:", int(proxy["label"].sum()), f"({proxy['label'].mean():.1%})")
    pos = gold[gold["label"] == 1].iloc[0]
    enc = make_windows(tok, [pos["text"]])
    first = tok.convert_ids_to_tokens(enc["input_ids"][0])
    print(f"first gold-positive unit -> {len(enc['input_ids'])} window(s); first window has {len(first)} tokens:")
    print(" ", first[:25], "...")
