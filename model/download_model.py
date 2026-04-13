# download_model.py
from huggingface_hub import snapshot_download
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="./model",
    local_dir_use_symlinks=False
)

