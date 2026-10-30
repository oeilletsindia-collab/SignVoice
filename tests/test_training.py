"""Temporary synthetic fixtures exercise training mechanics, never dataset claims."""
import json
import numpy as np
import torch
from src.preprocessing import load_config
from src.train import train
from src.checkpoint import load_checkpoint
from src.evaluate import evaluate

def test_train_resume_evaluate(tmp_path):
    data=tmp_path/'data'; data.mkdir(); c=load_config(); rng=np.random.default_rng(7)
    for split,count in [('train',80),('val',10),('test',10)]:
        rows=[]
        for i in range(count):
            uid=f'{split}-{i}'; path=data/f'{uid}.npy'; np.save(path,rng.normal(size=(3+i%3,376)).astype('float32'))
            rows.append(dict(uid=uid,text=f'{split} phrase {i}',pose=path.name,group=uid,schema=c['schema'],preprocessing=c))
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    output=tmp_path/'models'
    checkpoint=train(data,output,epochs=1,batch_size=32,hidden=8)
    _,_,meta=load_checkpoint(checkpoint)
    assert meta['epoch']==0 and len(meta['history'])==1
    train(data,output,epochs=2,batch_size=32,hidden=8,resume=output/'last.pt')
    _,_,last=load_checkpoint(output/'last.pt')
    assert last['epoch']==1 and len(last['history'])==2
    report=evaluate(checkpoint,data,tmp_path/'evaluation.json')
    assert report['samples']==10 and len(report['predictions'])==10

def test_baseline_reports_ineligibility(tmp_path):
    from src.baseline import run
    c=load_config()
    for split in ('train','val','test'):
        rows=[]
        for i in range(3):
            uid=f'{split}{i}'; pose=tmp_path/f'{uid}.npy'; np.save(pose,np.ones((3,376),np.float32))
            rows.append(dict(uid=uid,text=f'unique {split} {i}',pose=str(pose),group=uid,schema=c['schema'],preprocessing=c))
        (tmp_path/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=run(tmp_path,tmp_path/'baseline',epochs=1)
    assert report['status']=='ineligible' and 'accuracy' not in report
