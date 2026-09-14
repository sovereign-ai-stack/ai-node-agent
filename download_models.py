import os, sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from huggingface_hub import snapshot_download

models = [
    ('Qwen/Qwen2.5-1.5B-Instruct-AWQ', 'D:/models/Qwen2.5-1.5B-Instruct-AWQ'),
    ('Qwen/Qwen2.5-3B-Instruct-AWQ', 'D:/models/Qwen2.5-3B-Instruct-AWQ'),
]

for repo_id, target_dir in models:
    print(f'\n========================================================', flush=True)
    print(f'Downloading {repo_id} to {target_dir}...', flush=True)
    print('========================================================', flush=True)
    os.makedirs(target_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f'Successfully downloaded {repo_id} to {target_dir}!\n', flush=True)

print('All models downloaded successfully!', flush=True)
