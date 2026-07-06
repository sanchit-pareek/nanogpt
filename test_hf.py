from huggingface_hub import hf_hub_download

print("Downloading...")

path = hf_hub_download(
    repo_id="HuggingFaceFW/fineweb",
    repo_type="dataset",
    filename="sample/10BT/000_00000.parquet",
)

print(path)