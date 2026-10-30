"""Query actual HF metadata, license, revision and asset inventory; save evidence."""
import argparse,json,datetime
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download

REPOS=['Exploration-Lab/iSign','pedroodb/glosl-isl']
def inspect(output='reports/dataset_inventory.json'):
    results=[]; api=HfApi()
    for repo in REPOS:
        try:
            info=api.dataset_info(repo,files_metadata=True)
            files=[dict(path=s.rfilename,size=s.size) for s in info.siblings]
            card=info.card_data.to_dict() if info.card_data else {}
            result=dict(repo=repo,revision=info.sha,gated=info.gated,license=card.get('license'),card=card,files=files)
            print(json.dumps({k:result[k] for k in ('repo','revision','gated','license','files')},indent=2))
            try:
                path=hf_hub_download(repo,'README.md',repo_type='dataset',revision=info.sha)
                result['readme']=Path(path).read_text()
            except Exception as e: result['readme_error']=f'{type(e).__name__}: {e}'
        except Exception as e:
            result=dict(repo=repo,error=f'{type(e).__name__}: {e}'); print(result)
        results.append(result)
    out=dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),datasets=results)
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(out,indent=2))
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',default='reports/dataset_inventory.json')
    inspect(p.parse_args().output)
