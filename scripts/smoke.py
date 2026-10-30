"""Synthetic engineering check only: never a translation performance result."""
import json
from pathlib import Path
import numpy as np
import torch
from src.preprocessing import load_config,normalize_pose
from src.model import SignTranslator
from src.tokenizer import Tokenizer
from src.checkpoint import save_checkpoint,load_checkpoint

def run():
    torch.set_num_threads(2); torch.manual_seed(42)
    config=load_config(); raw=np.zeros((12,94,4),np.float32)
    raw[:,:,3]=1.; raw[:,0,0]=.3; raw[:,1,0]=.7
    x=torch.from_numpy(normalize_pose(raw,config))[None]
    tok=Tokenizer.fit(['engineering smoke test']); y=torch.tensor([tok.encode('engineering smoke test')])
    model=SignTranslator(len(tok.vocabulary),hidden=16,embedding=16,dropout=0)
    opt=torch.optim.AdamW(model.parameters(),lr=.02); losses=[]
    for _ in range(20):
        logits=model(x,torch.tensor([len(raw)]),y[:,:-1])
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y[:,1:].flatten())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(loss.item())
    assert losses[-1]<losses[0]
    save_checkpoint('models/smoke.pt',model,tok,config,trained=False,synthetic=True)
    restored,_,_=load_checkpoint('models/smoke.pt',require_trained=False)
    generated=restored.generate(x,torch.tensor([len(raw)]))
    report=dict(kind='synthetic engineering smoke check; not real ISL training',input_shape=list(x.shape),
                forward_shape=list(logits.shape),initial_loss=losses[0],final_loss=losses[-1],
                checkpoint='models/smoke.pt',generated_shape=list(generated.shape),real_aligned_samples_loaded=0)
    Path('reports/smoke.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': run()
