from aamcf_data.locomo.labels import build_label_table

turns_df, qa_df, evidence_index = build_label_table("data/locomo/locomo10.json")

print(f"Total turns: {len(turns_df)}  (expected ≈ 5,882)")
print(f"label_supervised positive rate: {turns_df['label_supervised'].mean():.2%}  (expected ≈ 24%)")
print(f"label_overlap positive rate: {turns_df['label_overlap'].mean():.2%}")
print(f"combined label positive rate: {turns_df['label'].mean():.2%}")

turns_df.to_parquet("data/locomo/locomo_labels.parquet", index=False)
qa_df.to_parquet("data/locomo/locomo_qa.parquet", index=False)
print("Saved to data/locomo/locomo_labels.parquet and locomo_qa.parquet")