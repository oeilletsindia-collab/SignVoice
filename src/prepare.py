"""Create aligned manifests from verified ISLTranslate uid,text CSV and raw videos.
No assumptions about precomputed archive formats are made.
"""
import argparse,csv,json,random,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from .pose_extraction import extract_video
from .preprocessing import load_config
from .tokenizer import Tokenizer

def prepare(csv_path,video_dir,output,limit=1000,seed=42):
    out=Path(output); (out/'poses').mkdir(parents=True,exist_ok=True)
    config=load_config(); candidates=[]; videos={}
    for path in sorted(Path(video_dir).rglob('*')):
        if path.suffix.lower() in {'.mp4','.avi','.mov'}:
            if path.stem in videos: raise ValueError(f'Ambiguous duplicate video stem: {path.stem}')
            videos[path.stem]=path
    with open(csv_path,encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'uid','text'} <= set(reader.fieldnames or []):
            raise ValueError(f'Expected verified uid,text columns; observed {reader.fieldnames}. Inspect annotations first.')
        seen=set()
        for row in reader:
            uid=row['uid'].strip(); text=row['text'].strip()
            if uid in seen: raise ValueError(f'Duplicate annotation uid: {uid}')
            seen.add(uid)
            if uid in videos and text:
                # Verified ISLTranslate UID format: source-video-id followed by segment index.
                group,sep,index=uid.rpartition('-')
                if not sep or not index.isdigit(): raise ValueError(f'Cannot derive source group from uid {uid}')
                candidates.append(dict(uid=uid,text=text,group=group,video=videos[uid]))
    random.Random(seed).shuffle(candidates)
    aligned=[]; rejected=[]
    for r in candidates:
        if len(aligned)>=limit: break
        try:
            x=extract_video(r['video'],config)
            # Hash filenames rather than trusting annotation path components.
            name=hashlib.sha256(r['uid'].encode()).hexdigest()+'.npy'
            np.save(out/'poses'/name,x)
            aligned.append(dict(uid=r['uid'],text=r['text'],group=r['group'],pose='poses/'+name,
                                schema=config['schema'],preprocessing=config))
        except (ValueError,RuntimeError) as e: rejected.append(dict(uid=r['uid'],error=str(e)))
    if len(aligned)<100:
        (out/'rejections.json').write_text(json.dumps(rejected,indent=2))
        raise ValueError(f'Only {len(aligned)} valid aligned examples; need at least 100. No usable split claimed.')
    # Union source groups AND repeated sentences before splitting to avoid leakage.
    parent=list(range(len(aligned)))
    def root(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    lookup={}
    for i,r in enumerate(aligned):
        for key in [('source',r['group']),('text',' '.join(Tokenizer.words(r['text'])))]:
            if key in lookup: parent[root(i)]=root(lookup[key])
            lookup[key]=i
    groups=defaultdict(list)
    for i,r in enumerate(aligned): groups[root(i)].append(r)
    blocks=list(groups.values()); random.Random(seed).shuffle(blocks)
    if len(blocks)<3: raise ValueError('Need at least three independent source/text components to split safely')
    splits={'train':[],'val':[],'test':[]}
    # Assign largest components first to approximately 80/10/10 without splitting components.
    for block in sorted(blocks,key=len,reverse=True):
        name=min(splits,key=lambda n:len(splits[n])/({'train':.8,'val':.1,'test':.1}[n]*len(aligned)))
        splits[name].extend(block)
    if any(not rows for rows in splits.values()): raise ValueError('Unable to create three nonempty disjoint splits')
    for name,rows in splits.items():
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    audit=dict(aligned=len(aligned),matched_candidates=len(candidates),counts={n:len(r) for n,r in splits.items()},
               rejected=rejected,seed=seed,split_policy='source-video and normalized-text component disjoint; not signer-disjoint',
               target_truncation_count=sum(len(Tokenizer.words(r['text']))>62 for r in aligned),
               annotation_sha256=hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),preprocessing=config)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)); print(json.dumps(audit['counts']))
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--videos',required=True)
    p.add_argument('--output',default='data/prepared'); p.add_argument('--limit',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    prepare(a.csv,a.videos,a.output,a.limit,a.seed)
