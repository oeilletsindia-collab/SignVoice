import csv
import numpy as np
import pytest
from src.prepare import prepare
from src.dataset import read_manifest,check_splits

def test_preparation_exact_alignment_and_disjoint_groups(tmp_path,monkeypatch):
    import src.prepare as preparation
    rng=np.random.default_rng(2)
    # Temporary synthetic fixtures verify matching and audit behavior only.
    monkeypatch.setattr(preparation,'extract_video',lambda *_:rng.normal(size=(3,376)).astype(np.float32))
    videos=tmp_path/'videos'; videos.mkdir(); annotation=tmp_path/'annotations.csv'
    with annotation.open('w') as f:
        writer=csv.writer(f); writer.writerow(['uid','text'])
        for i in range(110):
            uid=f'source{i//2}-{i%2}'
            writer.writerow([uid,f'sentence {i}'])
            (videos/f'{uid}.mp4').write_bytes(b'fixture, not a real video')
        writer.writerow(['unmatched-1','must not be silently paired'])
    audit=prepare(annotation,videos,tmp_path/'prepared',limit=100)
    assert audit['aligned']==100 and audit['matched_candidates']==110
    splits={n:read_manifest(tmp_path/'prepared'/f'{n}.jsonl') for n in ('train','val','test')}
    check_splits(splits)
    assert all(splits.values())
    assert all(r['uid']!='unmatched-1' for rows in splits.values() for r in rows)

def test_incompatible_annotation_schema_rejected(tmp_path):
    csv=tmp_path/'wrong.csv'; csv.write_text('label,video\nhello,something.mp4\n')
    with pytest.raises(ValueError,match='uid,text'): prepare(csv,tmp_path,tmp_path/'out')
