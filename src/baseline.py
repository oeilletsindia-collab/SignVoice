"""Train/evaluate closed-set BiLSTM classification only if repeated classes exist.
The held-out test uses new source groups, allows repeated class text, and reports
coverage explicitly. This experiment is separate from the text-disjoint main split.
"""
import argparse,json,random
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader
from .dataset import read_manifest,PoseDataset,collate
from .tokenizer import Tokenizer
from .model import SentenceClassifier
from .train import seed_everything

def run(data_dir,output,epochs=10,seed=42):
    seed_everything(seed); torch.set_num_threads(4)
    rows=[r for n in ('train','val','test') for r in read_manifest(Path(data_dir)/f'{n}.jsonl')]
    # Case/punctuation normalization for classes; original annotations remain intact.
    for r in rows: r['label']=' '.join(Tokenizer.words(r['text']))
    groups=sorted(set(r['group'] for r in rows)); random.Random(seed).shuffle(groups)
    n=max(1,len(groups)//5); test_groups=set(groups[:n]); val_groups=set(groups[n:2*n])
    train=[r for r in rows if r['group'] not in test_groups|val_groups]
    repeated=Counter(r['label'] for r in train); labels=sorted(k for k,v in repeated.items() if v>=2)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if len(labels)<2:
        report=dict(status='ineligible',reason='Fewer than two repeated training sentence classes. Use the most-frequent-sentence baseline in evaluation.json.')
        (out/'baseline.json').write_text(json.dumps(report,indent=2)); print(report); return report
    index={s:i for i,s in enumerate(labels)}; tok=Tokenizer.fit(labels)
    partitions={'train':train,'val':[r for r in rows if r['group'] in val_groups],
                'test':[r for r in rows if r['group'] in test_groups]}
    eligible={k:[r for r in v if r['label'] in index] for k,v in partitions.items()}
    if any(not r for r in eligible.values()):
        report=dict(status='ineligible',reason='No covered held-out examples for repeated sentence classes')
        (out/'baseline.json').write_text(json.dumps(report,indent=2)); print(report); return report
    import hashlib
    fingerprints={}
    for name,subset in eligible.items():
        values={hashlib.sha256(Path(r['pose']).read_bytes()).hexdigest() for r in subset}
        if any(values & old for old in fingerprints.values()):
            report=dict(status='ineligible',reason='Duplicate pose content crosses baseline splits')
            (out/'baseline.json').write_text(json.dumps(report,indent=2)); print(report); return report
        fingerprints[name]=values
    # Use label text in the data collator to select classification labels.
    for r in rows: r['text']=r['label']
    loaders={k:DataLoader(PoseDataset(v,tok),batch_size=16,collate_fn=collate,shuffle=k=='train') for k,v in eligible.items()}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=SentenceClassifier(len(labels)).to(device); optim=torch.optim.AdamW(model.parameters(),lr=3e-4)
    best=float('inf')
    for epoch in range(epochs):
        model.train()
        for x,lens,_,texts in loaders['train']:
            y=torch.tensor([index[s] for s in texts],device=device)
            loss=torch.nn.functional.cross_entropy(model(x.to(device),lens),y)
            optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optim.step()
        model.eval(); val=0.; count=0
        with torch.no_grad():
            for x,lens,_,texts in loaders['val']:
                y=torch.tensor([index[s] for s in texts],device=device)
                val+=torch.nn.functional.cross_entropy(model(x.to(device),lens),y,reduction='sum').item(); count+=len(texts)
        if val/count<best:
            best=val/count; torch.save(dict(model=model.state_dict(),labels=labels,task='closed-set sentence classification'),out/'baseline.pt')
    model.load_state_dict(torch.load(out/'baseline.pt',map_location=device,weights_only=True)['model']); model.eval()
    refs=[]; preds=[]
    with torch.no_grad():
        for x,lens,_,texts in loaders['test']:
            refs.extend(index[s] for s in texts); preds.extend(model(x.to(device),lens).argmax(-1).cpu().tolist())
    from sklearn.metrics import accuracy_score,confusion_matrix
    report=dict(status='evaluated',task='closed-set sentence classification; separate source-group split',
                accuracy=accuracy_score(refs,preds),classes=labels,test_samples=len(refs),
                test_coverage=len(refs)/len(partitions['test']),confusion_matrix=confusion_matrix(refs,preds,labels=list(range(len(labels)))).tolist())
    (out/'baseline.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report)); return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/prepared'); p.add_argument('--output',default='reports/baseline')
    p.add_argument('--epochs',type=int,default=10); a=p.parse_args(); run(a.data,a.output,a.epochs)
