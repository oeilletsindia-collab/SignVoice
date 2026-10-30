"""Download explicitly selected actual assets at a pinned HF revision.
No guessed file names. Never auto-execute downloaded code.
"""
import argparse,json
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download

def download(repo,files,output='data/downloads',revision=None,max_gb=8):
    info=HfApi().dataset_info(repo,revision=revision,files_metadata=True)
    assets={s.rfilename:s for s in info.siblings}
    if not files or any(f not in assets for f in files): raise ValueError('Select exact filenames from inspect_datasets.py inventory')
    size=sum(assets[f].size or 0 for f in files)
    if size>max_gb*1024**3: raise ValueError(f'{size/1024**3:.2f} GB exceeds budget; choose sample assets or increase --max-gb explicitly')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    paths=[hf_hub_download(repo,f,repo_type='dataset',revision=info.sha,local_dir=output) for f in files]
    (output/'download_provenance.json').write_text(json.dumps(dict(repo=repo,revision=info.sha,files=files),indent=2))
    return paths

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repo',default='Exploration-Lab/iSign'); p.add_argument('--files',nargs='+',required=True)
    p.add_argument('--revision'); p.add_argument('--output',default='data/downloads'); p.add_argument('--max-gb',type=float,default=8)
    a=p.parse_args(); print(download(a.repo,a.files,a.output,a.revision,a.max_gb))
