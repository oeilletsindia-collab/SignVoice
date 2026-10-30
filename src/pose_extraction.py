"""Raw video -> precisely defined MediaPipe 94-landmark sequence."""
from pathlib import Path
import importlib.metadata
import cv2
import numpy as np
from .preprocessing import POSE, FACE, normalize_pose, load_config

def video_path(value):
    # Gradio 5 Video returns a filepath for both upload and recorded webcam.
    if isinstance(value, dict): value = value.get('video') or value.get('path')
    if not isinstance(value, (str, Path)): raise ValueError('Upload or record a video first')
    return Path(value)

def extract_video(value, config=None):
    config = config or load_config()
    path = video_path(value)
    if not path.is_file(): raise ValueError('Video file does not exist')
    if path.suffix.lower() not in {'.mp4','.avi','.mov'}: raise ValueError('Use MP4, AVI, or MOV video')
    if path.stat().st_size > config['max_bytes']: raise ValueError('Video exceeds the 200 MB limit')
    if importlib.metadata.version('mediapipe') != config['mediapipe_version']:
        raise ValueError('MediaPipe version differs from the checkpoint preprocessing metadata')
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError('Unable to decode this video')
    frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0: raise ValueError('Video has no valid frame rate')
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if count/fps > config['max_seconds'] + 0.5: raise ValueError('Record at most 60 seconds')
        step = fps/config['sample_fps']
        next_frame = 0.
        with mp.solutions.holistic.Holistic(model_complexity=1, smooth_landmarks=True,
                refine_face_landmarks=False, min_detection_confidence=config['min_detection_confidence'],
                min_tracking_confidence=0.5) as holistic:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok: break
                if index/fps > config['max_seconds']: raise ValueError('Video exceeds duration limit')
                if index + 1e-6 >= next_frame:
                    # No horizontal flipping: signer handedness is preserved in both modes.
                    height,width = frame.shape[:2]
                    if width > 960: frame = cv2.resize(frame,(960,round(height*960/width)))
                    result = holistic.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    x = np.zeros((94,4),dtype=np.float32)
                    if result.pose_landmarks:
                        for i,k in enumerate(POSE):
                            l = result.pose_landmarks.landmark[k]; x[i] = [l.x,l.y,l.z,l.visibility]
                    for offset,hand in [(6,result.left_hand_landmarks),(27,result.right_hand_landmarks)]:
                        if hand:
                            for i,l in enumerate(hand.landmark): x[offset+i] = [l.x,l.y,l.z,1.]
                    if result.face_landmarks:
                        for i,k in enumerate(FACE):
                            l = result.face_landmarks.landmark[k]; x[48+i] = [l.x,l.y,l.z,1.]
                    frames.append(x)
                    next_frame += max(1.,step)
                index += 1
    finally: cap.release()
    if len(frames) < 2: raise ValueError('Video is empty or too short')
    raw = np.stack(frames)
    if not (raw[:,6:48,3] > 0).any(): raise ValueError('No hands detected; use a well-lit, front-facing signing video')
    return normalize_pose(raw,config)
