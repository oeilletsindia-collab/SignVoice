from pathlib import Path
import tempfile
from gtts import gTTS

def synthesize(text,output_dir='outputs/audio'):
    if not text.strip(): raise ValueError('No translated text to synthesize')
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.mp3',dir=out,delete=False) as f: path=Path(f.name)
    try:
        gTTS(text=text,lang='en',timeout=(5,20)).save(str(path))
        if path.stat().st_size==0: raise RuntimeError('Speech service returned an empty file')
    except Exception:
        path.unlink(missing_ok=True); raise
    return str(path)
