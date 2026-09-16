from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="xiaowu0162/longmemeval-cleaned",
    filename="longmemeval_s_cleaned.json",
    repo_type="dataset",
    local_dir="data/longmemeval",
)