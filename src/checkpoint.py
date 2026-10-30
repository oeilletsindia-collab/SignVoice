from pathlib import Path
import torch
from .model import SignTranslator
from .tokenizer import Tokenizer
from .preprocessing import validate_config

def save_checkpoint(path,model,tokenizer,preprocessing,**extra):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(format_version=1,model_config=model.config,model_state=model.state_dict(),
                 vocabulary=tokenizer.vocabulary,preprocessing=preprocessing,
                 tokenizer_config={'type':'word','max_tokens':64,'lowercase':True},**extra)
    temp=path.with_suffix('.tmp'); torch.save(payload,temp); temp.replace(path)

def load_checkpoint(path,device='cpu',require_trained=True):
    if not Path(path).is_file(): raise FileNotFoundError('No trained checkpoint found. Train in Colab and copy best.pt to models/best.pt.')
    data=torch.load(path,map_location=device,weights_only=True)
    if data.get('format_version')!=1: raise ValueError('Unsupported checkpoint format')
    validate_config(data['preprocessing'])
    if require_trained and (not data.get('trained') or data.get('synthetic',False)):
        raise ValueError('This checkpoint is a smoke-test artifact, not a real-data trained translation model')
    tok=Tokenizer(data['vocabulary'])
    if data['model_config']['vocab_size']!=len(tok.vocabulary): raise ValueError('Vocabulary/model mismatch')
    model=SignTranslator(**data['model_config']).to(device)
    model.load_state_dict(data['model_state']); model.eval()
    return model,tok,data
