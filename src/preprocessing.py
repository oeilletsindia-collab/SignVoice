"""Versioned, shared training/inference landmark representation."""
import json
from pathlib import Path
import numpy as np

SCHEMA = 'signvoice-holistic94-v1'
POSE = [11, 12, 13, 14, 23, 24]
FACE = [61,146,91,181,84,17,314,405,321,375,291,409,270,95,88,178,87,14,78,308,
        33,133,160,158,153,144,145,246,362,263,387,385,380,373,374,249,
        107,66,105,63,70,336,296,334,293,300]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config/preprocessing.json'

def load_config(path=DEFAULT_CONFIG):
    c = json.loads(Path(path).read_text())
    validate_config(c)
    return c

def validate_config(c):
    if c.get('schema') != SCHEMA or c.get('feature_dim') != 376:
        raise ValueError('Unsupported pose schema; matching dimensions alone do not establish compatibility.')
    if c.get('normalization') != 'shoulder-center-width-v1':
        raise ValueError('Unsupported normalization')
    for key in ('sample_fps', 'max_frames', 'max_seconds', 'max_bytes'):
        if c[key] <= 0: raise ValueError(f'{key} must be positive')

def normalize_pose(raw, config):
    validate_config(config)
    x = np.asarray(raw, dtype=np.float32).copy()
    if x.ndim != 3 or x.shape[1:] != (94, 4) or len(x) == 0:
        raise ValueError('Expected a nonempty [time,94,4] raw landmark sequence')
    if not np.isfinite(x).all(): raise ValueError('Nonfinite landmarks')
    valid = x[..., 3] >= config['visibility_threshold']
    # Only interpolate bounded internal gaps. Missing hands remain missing elsewhere.
    for k in range(94):
        good = np.flatnonzero(valid[:, k])
        for a, b in zip(good[:-1], good[1:]):
            if 1 < b-a <= config['max_gap'] + 1:
                for j in range(a+1, b):
                    x[j,k] = x[a,k] + (x[b,k]-x[a,k]) * ((j-a)/(b-a))
                    valid[j,k] = True
    shoulders = valid[:,0] & valid[:,1]
    if not shoulders.any(): raise ValueError('No reliable shoulders detected; keep the upper body visible')
    centers = (x[:,0,:3] + x[:,1,:3])/2
    scales = np.linalg.norm(x[:,0,:2]-x[:,1,:2], axis=1)
    reliable = shoulders & (scales > 1e-3)
    if not reliable.any(): raise ValueError('Degenerate shoulder scale')
    # Nearest temporal interpolation of the coordinate frame, not absent body parts.
    times = np.arange(len(x))
    for d in range(3): centers[:,d] = np.interp(times, times[reliable], centers[reliable,d])
    scales = np.interp(times, times[reliable], scales[reliable])
    x[...,:3] = (x[...,:3]-centers[:,None,:])/scales[:,None,None]
    x[...,:3] = np.clip(x[...,:3], -10, 10)
    x[~valid] = 0
    if len(x) > config['max_frames']:
        x = x[np.linspace(0,len(x)-1,config['max_frames']).astype(int)]
    return x.reshape(len(x),376).astype(np.float32)
