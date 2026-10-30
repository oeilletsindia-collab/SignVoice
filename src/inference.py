from pathlib import Path
import threading
import torch
from .checkpoint import load_checkpoint
from .pose_extraction import extract_video
from .speech import synthesize

class Translator:
    def __init__(self,checkpoint='models/best.pt'):
        self.checkpoint=Path(checkpoint); self._cache=None; self._stamp=None; self._lock=threading.Lock()
    def load(self):
        if not self.checkpoint.is_file():
            raise FileNotFoundError('No trained model installed. Run the Colab training notebook, then place best.pt in models/.')
        stamp=(self.checkpoint.stat().st_mtime_ns,self.checkpoint.stat().st_size)
        if self._cache is None or self._stamp!=stamp:
            self._cache=load_checkpoint(self.checkpoint); self._stamp=stamp
        return self._cache
    def translate(self,video):
        with self._lock:
            model,tok,meta=self.load()
            features=extract_video(video,meta['preprocessing'])
            with torch.inference_mode():
                ids=model.generate(torch.from_numpy(features)[None],torch.tensor([len(features)]))[0]
            text=tok.decode(ids.tolist())
        if not text or text=='<unk>': raise ValueError('Model produced no usable text. Try a clearer sequence; inspect validation predictions.')
        try: audio=synthesize(text)
        except Exception as e:
            return text,None,f'Translation complete. Speech service unavailable ({type(e).__name__}); check Internet access.'
        return text,audio,'Translation complete. Review the result: this research model may make errors.'
