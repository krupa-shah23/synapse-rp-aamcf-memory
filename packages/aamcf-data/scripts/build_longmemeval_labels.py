from aamcf_data.longmemeval.labels import build_label_table

sessions_df, questions_df, evidence_index = build_label_table(
    "data/longmemeval/longmemeval_s_cleaned.json"
)

print(f"Total (question_id, session_id) rows: {len(sessions_df)}")
print(f"label_supervised positive rate: {sessions_df['label_supervised'].mean():.2%}  (expected ≈ 5%)")
print(f"label_overlap positive rate: {sessions_df['label_overlap'].mean():.2%}")
print(f"combined label positive rate: {sessions_df['label'].mean():.2%}")

sessions_df.to_parquet("data/longmemeval/longmemeval_labels.parquet", index=False)
questions_df.to_parquet("data/longmemeval/longmemeval_questions.parquet", index=False)
print("Saved to data/longmemeval/longmemeval_labels.parquet and longmemeval_questions.parquet")

