"""Strict canonical manifests; never substitute zeros for a missing training asset."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .preprocessing import SCHEMA, validate_config

def read_manifest(path):
    path=Path(path)
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows: raise ValueError(f'Empty manifest: {path}')
    seen=set()
    for row in rows:
        for key in ('uid','text','pose','group','schema','preprocessing'):
            if key not in row: raise ValueError(f'Missing {key} in manifest')
        if not isinstance(row['text'],str) or not row['text'].strip(): raise ValueError('Empty annotation')
        if row['schema'] != SCHEMA: raise ValueError('Unverified feature schema')
        validate_config(row['preprocessing'])
        if row['uid'] in seen: raise ValueError('Duplicate uid')
        seen.add(row['uid'])
        pose=Path(row['pose'])
        if not pose.is_absolute(): pose=path.parent/pose
        row['pose']=str(pose.resolve())
    return rows

def check_splits(splits):
    """Reject overlap in IDs, source groups, normalized text, and feature hashes."""
    from .tokenizer import Tokenizer
    from hashlib import sha256
    seen=[set() for _ in range(4)]
    config=None
    for name,rows in splits.items():
        current=[set() for _ in range(4)]
        for r in rows:
            if config is None: config=r['preprocessing']
            if config != r['preprocessing']: raise ValueError('Inconsistent preprocessing across dataset')
            fingerprint=sha256(Path(r['pose']).read_bytes()).hexdigest()
            for s,v in zip(current,(r['uid'],r['group'],' '.join(Tokenizer.words(r['text'])),fingerprint)):
                s.add(v)
        for old,new in zip(seen,current):
            if old & new: raise ValueError(f'Cross-split leakage detected in {name}')
            old.update(new)

class PoseDataset(Dataset):
    def __init__(self,rows,tokenizer): self.rows=rows; self.tokenizer=tokenizer
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]
        x=np.load(r['pose'],allow_pickle=False)
        if x.ndim!=2 or x.shape[1]!=376 or len(x)<2 or not np.isfinite(x).all():
            raise ValueError(f"Invalid normalized pose: {r['uid']}")
        if len(x)>r['preprocessing']['max_frames']: raise ValueError('Pose exceeds frame limit')
        return torch.from_numpy(x.astype(np.float32)),torch.tensor(self.tokenizer.encode(r['text'])),r['text']

def collate(batch):
    x,y,text=zip(*batch)
    return pad_sequence(x,batch_first=True),torch.tensor([len(a) for a in x]),pad_sequence(y,batch_first=True),text

def manifest_digest(rows):
    """Content identity independent of the machine's dataset mount location."""
    import hashlib
    records=[]
    for r in rows:
        record={k:r[k] for k in ('uid','text','group','schema','preprocessing')}
        record['pose_sha256']=hashlib.sha256(Path(r['pose']).read_bytes()).hexdigest()
        records.append(record)
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
