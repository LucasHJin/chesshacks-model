# upload_to_hf.py
from huggingface_hub import HfApi, login

login()

api = HfApi()

# Upload files
api.upload_file(
    path_or_fileobj="models/model_best_simple.pt",
    path_in_repo="model_best.pt",
    repo_id="LucasHJ/chess-engine",
    repo_type="model",
)

api.upload_file(
    path_or_fileobj="data/processed/move_vocab.json",
    path_in_repo="move_vocab.json",
    repo_id="LucasHJ/chess-engine",
    repo_type="model",
)

print("Uploaded to HuggingFace!")