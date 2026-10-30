import argparse,json,random,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .dataset import read_manifest,check_splits,PoseDataset,collate,manifest_digest
from .tokenizer import Tokenizer
from .model import SignTranslator
from .checkpoint import save_checkpoint,load_checkpoint

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def epoch_loss(model,loader,device,optimizer=None):
    model.train(optimizer is not None); total=0.; count=0
    criterion=nn.CrossEntropyLoss(ignore_index=0,reduction='sum')
    with torch.set_grad_enabled(optimizer is not None):
        for x,lengths,y,_ in loader:
            x=x.to(device); y=y.to(device)
            logits=model(x,lengths,y[:,:-1]); target=y[:,1:]
            loss=criterion(logits.reshape(-1,logits.size(-1)),target.reshape(-1))
            tokens=target.ne(0).sum()
            if optimizer:
                optimizer.zero_grad(); (loss/tokens).backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total+=loss.item(); count+=tokens.item()
    if count==0: raise ValueError('No nonpadding target tokens')
    return total/count

def train(data_dir,output,epochs=20,batch_size=16,hidden=128,lr=3e-4,patience=5,seed=42,resume=None):
    seed_everything(seed); torch.set_num_threads(min(4,torch.get_num_threads()))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    splits={n:read_manifest(Path(data_dir)/f'{n}.jsonl') for n in ('train','val','test')}
    check_splits(splits)
    if sum(map(len,splits.values()))<100: raise ValueError('Initial real-data milestone requires >=100 aligned examples')
    config=splits['train'][0]['preprocessing']; digest=manifest_digest(splits['train'])
    tok=Tokenizer.fit(r['text'] for r in splits['train'])
    model=SignTranslator(len(tok.vocabulary),hidden=hidden).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2)
    start=0; best=float('inf'); stale=0; history=[]
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if resume:
        restored,rtok,meta=load_checkpoint(resume,device)
        if meta['train_digest']!=digest or meta['preprocessing']!=config: raise ValueError('Resume dataset/config mismatch')
        if rtok.vocabulary!=tok.vocabulary or restored.config!=model.config: raise ValueError('Resume model/vocabulary mismatch')
        model.load_state_dict(restored.state_dict()); optimizer.load_state_dict(meta['optimizer'])
        scheduler.load_state_dict(meta['scheduler']); start=meta['epoch']+1
        best=meta['best_val_loss']; stale=meta['stale']; history=meta['history']
        torch.set_rng_state(meta['torch_rng'].cpu())
        if device=='cuda' and meta.get('cuda_rng'): torch.cuda.set_rng_state_all([s.cpu() for s in meta['cuda_rng']])
        if not (out/'best.pt').exists(): raise ValueError('Resume in the original output directory containing best.pt')
    val_loader=DataLoader(PoseDataset(splits['val'],tok),batch_size=batch_size,collate_fn=collate)
    for epoch in range(start,epochs):
        generator=torch.Generator().manual_seed(seed+epoch)
        loader=DataLoader(PoseDataset(splits['train'],tok),batch_size=batch_size,shuffle=True,generator=generator,collate_fn=collate)
        loss=epoch_loss(model,loader,device,optimizer); val=epoch_loss(model,val_loader,device)
        scheduler.step(val); history.append(dict(epoch=epoch+1,train_loss=loss,val_loss=val))
        improved=val<best
        if improved: best=val; stale=0
        else: stale+=1
        metadata=dict(trained=True,synthetic=False,epoch=epoch,history=history,best_val_loss=best,
                      stale=stale,train_digest=digest,seed=seed,dataset='ISLTranslate raw-video extraction',
                      train_samples=len(splits['train']),val_samples=len(splits['val']),
                      optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [])
        save_checkpoint(out/'last.pt',model,tok,config,**metadata)
        if improved: save_checkpoint(out/'best.pt',model,tok,config,**metadata)
        (out/'history.json').write_text(json.dumps(history,indent=2))
        (out/'vocabulary.json').write_text(json.dumps(tok.vocabulary,indent=2))
        (out/'preprocessing.json').write_text(json.dumps(config,indent=2))
        (out/'model_config.json').write_text(json.dumps(model.config,indent=2))
        print(json.dumps(history[-1]),flush=True)
        if stale>=patience: break
    if not history: raise ValueError('No training epochs executed; increase --epochs beyond the resumed epoch')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key in ('train_loss','val_loss'): plt.plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    plt.xlabel('Epoch'); plt.ylabel('Token cross entropy'); plt.legend(); plt.tight_layout()
    plt.savefig(out/'learning_curves.png'); plt.close()
    return out/'best.pt'

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/prepared'); p.add_argument('--output',default='models')
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--hidden',type=int,default=128); p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--patience',type=int,default=5); p.add_argument('--seed',type=int,default=42); p.add_argument('--resume')
    a=p.parse_args(); train(a.data,a.output,a.epochs,a.batch_size,a.hidden,a.lr,a.patience,a.seed,a.resume)
