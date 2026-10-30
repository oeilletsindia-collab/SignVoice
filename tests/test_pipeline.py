import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.preprocessing import load_config,normalize_pose
from src.tokenizer import Tokenizer
from src.model import SignTranslator
from src.dataset import read_manifest,PoseDataset,collate,check_splits
from src.checkpoint import save_checkpoint,load_checkpoint
from src.pose_extraction import video_path,extract_video

torch.set_num_threads(2)
@pytest.fixture
def raw():
    x=np.zeros((9,94,4),np.float32); x[:,:,3]=1
    x[:,0,0]=.25; x[:,1,0]=.75; x[:,6:48,1]=.6
    return x

def test_normalization_missing_and_translation_invariance(raw):
    c=load_config(); raw[2:4,8]=0
    a=normalize_pose(raw,c)
    shifted=raw.copy(); shifted[...,:3]+=.2
    b=normalize_pose(shifted,c)
    np.testing.assert_allclose(a,b,atol=1e-6)
    assert a.shape==(9,376) and np.isfinite(a).all()
    assert a.reshape(9,94,4)[2,8,3]>0
    raw[:,9]=0
    assert (normalize_pose(raw,c).reshape(9,94,4)[:,9]==0).all()

def test_bad_pose_and_bounded_length(raw):
    c=load_config(); c['max_frames']=4
    assert normalize_pose(raw,c).shape==(4,376)
    with pytest.raises(ValueError): normalize_pose(np.zeros((9,94,4)),c)
    with pytest.raises(ValueError): normalize_pose(np.zeros((9,94,3)),c)

def test_tokenizer_and_autoregression():
    tok=Tokenizer.fit(['hello world.'])
    assert tok.decode(tok.encode('Hello world.'))=='hello world.'
    assert tok.encode('unseen')[1]==3
    m=SignTranslator(len(tok.vocabulary),hidden=16,embedding=8,dropout=0)
    x=torch.randn(2,8,376); lengths=torch.tensor([8,3]); y=torch.tensor([[1,4,5],[1,4,0]])
    logits=m(x,lengths,y)
    assert logits.shape==(2,3,len(tok.vocabulary)); logits.sum().backward()
    m.eval()
    # Padding must never affect encoded states or attention.
    altered=x.clone(); altered[1,3:]=1000
    with torch.no_grad(): torch.testing.assert_close(m(x,lengths,y)[1],m(altered,lengths,y)[1])
    assert m.generate(x,lengths,max_tokens=5).shape[1]<=5

def test_checkpoint_roundtrip_and_smoke_rejection(tmp_path):
    tok=Tokenizer.fit(['hello']); model=SignTranslator(len(tok.vocabulary),hidden=8)
    file=tmp_path/'smoke.pt'; save_checkpoint(file,model,tok,load_config(),trained=False,synthetic=True)
    with pytest.raises(ValueError,match='smoke'): load_checkpoint(file)
    loaded,rtok,_=load_checkpoint(file,require_trained=False)
    for a,b in zip(model.parameters(),loaded.parameters()): torch.testing.assert_close(a,b)
    assert rtok.vocabulary==tok.vocabulary
    with pytest.raises(FileNotFoundError): load_checkpoint(tmp_path/'absent.pt')

def test_manifest_alignment_dimensions_and_collation(tmp_path,raw):
    c=load_config(); rows=[]
    for i,t in enumerate([5,9]):
        p=tmp_path/f'{i}.npy'; np.save(p,normalize_pose(raw[:t],c))
        rows.append(dict(uid=str(i),text='hello',pose=p.name,group=f'g{i}',schema=c['schema'],preprocessing=c))
    manifest=tmp_path/'train.jsonl'; manifest.write_text('\n'.join(json.dumps(r) for r in rows))
    rows=read_manifest(manifest); dataset=PoseDataset(rows,Tokenizer.fit(['hello']))
    x,lens,y,texts=collate([dataset[0],dataset[1]])
    assert x.shape==(2,9,376); assert lens.tolist()==[5,9]; assert len(texts)==2
    with pytest.raises(ValueError,match='leakage'): check_splits({'train':rows[:1],'test':rows[:1]})
    np.save(tmp_path/'0.npy',np.zeros((3,42)))
    with pytest.raises(ValueError,match='Invalid'): dataset[0]

def test_video_and_recording_input_validation(tmp_path):
    path=tmp_path/'recording.mp4'; path.write_bytes(b'invalid video')
    assert video_path(str(path))==video_path({'video':str(path)})
    with pytest.raises(ValueError,match='decode'): extract_video(path)
    with pytest.raises(ValueError): video_path(None)

def test_actual_mediapipe_rejects_no_signer(tmp_path):
    import cv2
    p=tmp_path/'blank.mp4'; writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),12,(64,64))
    for _ in range(6): writer.write(np.zeros((64,64,3),np.uint8))
    writer.release()
    with pytest.raises(ValueError,match='No hands'): extract_video(p)

def test_speech_contract(tmp_path,monkeypatch):
    import src.speech as speech
    class FakeTTS:
        def __init__(self,**kwargs): assert kwargs['lang']=='en' and kwargs['text']=='hello'
        def save(self,path): Path(path).write_bytes(b'mocked audio, not playable')
    monkeypatch.setattr(speech,'gTTS',FakeTTS)
    result=speech.synthesize('hello',tmp_path)
    assert Path(result).is_file()
    with pytest.raises(ValueError): speech.synthesize('',tmp_path)

def test_metrics():
    from src.evaluate import metrics
    m=metrics(['hello world'],['hello world'])
    assert m['wer']==0 and m['exact_match']==1 and m['bleu']==pytest.approx(100)
    assert metrics(['hello world'],[''])['wer']==1

def test_app_build_and_missing_model_error(tmp_path):
    from app import build_app
    from src.inference import Translator
    demo=build_app(tmp_path/'missing.pt')
    config=demo.get_config_file()
    sources=[c['props'].get('sources') for c in config['components'] if c['type']=='video']
    assert ['upload'] in sources and ['webcam'] in sources
    assert all(c['props'].get('format')=='mp4' for c in config['components'] if c['type']=='video')
    with pytest.raises(FileNotFoundError): Translator(tmp_path/'missing.pt').translate('anything.mp4')

def test_video_extraction_positive_layout_and_shared_input(tmp_path,monkeypatch):
    import cv2,mediapipe as mp
    from types import SimpleNamespace as S
    # A detector fixture verifies the real video-decoding/sampling/normalization
    # path; this is NOT evidence that a synthetic image contains a signer.
    points=[S(x=.4,y=.5,z=.01,visibility=1.) for _ in range(468)]
    points[11]=S(x=.25,y=.5,z=0.,visibility=1.)
    points[12]=S(x=.75,y=.5,z=0.,visibility=1.)
    class Detector:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def process(self,image):
            return S(pose_landmarks=S(landmark=points),left_hand_landmarks=S(landmark=points[:21]),
                     right_hand_landmarks=None,face_landmarks=S(landmark=points))
    monkeypatch.setattr(mp.solutions.holistic,'Holistic',Detector)
    file=tmp_path/'recording.mp4'
    writer=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*'mp4v'),24,(64,64))
    for _ in range(24): writer.write(np.zeros((64,64,3),np.uint8))
    writer.release()
    upload=extract_video(str(file)); webcam=extract_video({'video':str(file)})
    np.testing.assert_array_equal(upload,webcam)
    assert upload.shape==(12,376)
    assert (upload.reshape(12,94,4)[:,27:48]==0).all() # absent right hand remains missing

def test_inference_runs_model_and_preserves_text_on_speech_failure(raw,monkeypatch):
    import src.inference as inference
    config=load_config(); features=normalize_pose(raw,config)
    tok=Tokenizer.fit(['test output']); model=SignTranslator(len(tok.vocabulary),hidden=8,dropout=0)
    # Force termination after a known vocabulary token via a fixture generation
    # function; verify Translator uses the returned model tokens, never filename.
    called=[]
    def generate(x,lengths):
        called.append(tuple(x.shape)); return torch.tensor([[tok.index['test'],2]])
    monkeypatch.setattr(model,'generate',generate)
    translator=inference.Translator('unused')
    monkeypatch.setattr(translator,'load',lambda:(model,tok,{'preprocessing':config}))
    monkeypatch.setattr(inference,'extract_video',lambda video,c:features)
    def unavailable(text): raise ConnectionError('offline')
    monkeypatch.setattr(inference,'synthesize',unavailable)
    text,audio,status=translator.translate('filename_is_not_prediction.mp4')
    assert text=='test' and audio is None and 'unavailable' in status
    assert called==[(1,9,376)]
