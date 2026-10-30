import argparse,json
from pathlib import Path
import torch
import sacrebleu
from jiwer import wer
from torch.utils.data import DataLoader
from .dataset import read_manifest,PoseDataset,collate,check_splits,manifest_digest
from .checkpoint import load_checkpoint
from .tokenizer import Tokenizer

def metrics(references,predictions):
    if not references or len(references)!=len(predictions): raise ValueError('Need aligned nonempty predictions')
    refs=[' '.join(Tokenizer.words(s)) for s in references]
    hyps=[' '.join(Tokenizer.words(s)) for s in predictions]
    bleu=sacrebleu.metrics.BLEU(effective_order=True)
    score=bleu.corpus_score(hyps,[refs])
    return dict(bleu=score.score,bleu_signature=str(bleu.get_signature()),wer=wer(refs,hyps),
                exact_match=sum(a==b for a,b in zip(refs,hyps))/len(refs),samples=len(refs))

@torch.no_grad()
def predict(model,tokenizer,loader,device):
    refs=[]; preds=[]
    model.eval()
    for x,lengths,_,texts in loader:
        ids=model.generate(x.to(device),lengths)
        refs.extend(texts); preds.extend(tokenizer.decode(row) for row in ids.cpu().tolist())
    return refs,preds

def evaluate(checkpoint,data_dir,output):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,tok,meta=load_checkpoint(checkpoint,device)
    splits={n:read_manifest(Path(data_dir)/f'{n}.jsonl') for n in ('train','val','test')}
    check_splits(splits)
    if any(r['preprocessing']!=meta['preprocessing'] for rows in splits.values() for r in rows):
        raise ValueError('Evaluation feature configuration differs from checkpoint')
    import hashlib
    digest=manifest_digest(splits['train'])
    if digest!=meta.get('train_digest'): raise ValueError('Training manifest differs from checkpoint; verify evaluation provenance')
    refs,preds=predict(model,tok,DataLoader(PoseDataset(splits['test'],tok),batch_size=16,collate_fn=collate),device)
    report=metrics(refs,preds); report['checkpoint_sha256']=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(); report['predictions']=[dict(reference=r,prediction=p) for r,p in zip(refs,preds)]
    # A transparent non-neural baseline available even when sentence classes do not repeat.
    from collections import Counter
    common=Counter(r['text'] for r in splits['train']).most_common(1)[0][0]
    report['most_frequent_sentence_baseline']=metrics(refs,[common]*len(refs))
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='models/best.pt')
    p.add_argument('--data',default='data/prepared'); p.add_argument('--output',default='reports/evaluation.json')
    a=p.parse_args(); evaluate(a.checkpoint,a.data,a.output)
