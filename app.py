"""SignVoice: shared upload/webcam inference, no demo predictions."""
import json,os
from pathlib import Path
import gradio as gr
from src.inference import Translator

ROOT=Path(__file__).resolve().parent
CSS='''
.gradio-container {max-width:1180px!important;font-family:Inter,system-ui,sans-serif;}
#hero {padding:24px;border-radius:16px;background:linear-gradient(120deg,#123568,#176ba0);color:white;margin-bottom:16px;}
#hero h1,#hero p {color:white!important;} footer {visibility:hidden;}
'''

def model_information(checkpoint):
    if not Path(checkpoint).is_file():
        return '**Training status: no trained checkpoint installed.**\n\nUpload and webcam controls are available; translation requires real-data training in Colab. No evaluation results are available yet.'
    try:
        from src.checkpoint import load_checkpoint
        _,_,m=load_checkpoint(checkpoint)
        text=f"**Dataset:** {m.get('dataset','Unspecified')}\n\n**Architecture:** MediaPipe 94 landmarks → projection → BiLSTM → masked attention GRU text decoder.\n\n**Training samples:** {m.get('train_samples','unknown')} · **Selected epoch:** {m.get('epoch',-1)+1}\n\n**Validation token loss:** {m.get('best_val_loss','unavailable')}\n\n"
        report=ROOT/'reports/evaluation.json'
        if report.is_file():
            r=json.loads(report.read_text())
            import hashlib
            if r.get('checkpoint_sha256') != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
                return text+'Evaluation report belongs to a different checkpoint. Run evaluation again.'
            text+='**Evaluation file (verify it belongs to this checkpoint):**\n\n'+json.dumps({k:r[k] for k in ('bleu','wer','exact_match','samples')},indent=2)
        return text
    except Exception as e: return f'**Checkpoint unavailable:** {e}'

def build_app(checkpoint=None):
    translator=Translator(checkpoint or os.environ.get('SIGNVOICE_CHECKPOINT',str(ROOT/'models/best.pt')))
    def translate(video,progress=gr.Progress()):
        if video is None: raise gr.Error('Upload or record a signing video first.')
        progress(.1,desc='Extracting landmarks and translating…')
        try: text,audio,status=translator.translate(video)
        except (ValueError,FileNotFoundError,RuntimeError) as e: raise gr.Error(str(e)) from e
        progress(1,desc='Complete')
        return text,audio,audio,status
    with gr.Blocks(title='SignVoice',theme=gr.themes.Soft(primary_hue='blue',neutral_hue='slate'),css=CSS) as demo:
        gr.Markdown('# SignVoice\nContinuous Indian Sign Language to Text and Speech Translation',elem_id='hero')
        gr.Markdown('Record a connected signing sequence with your face, shoulders, and both hands visible. Research prototype; verify every translation.')
        for title,source,label in [('Upload Video','upload','Upload an MP4, AVI, or MOV'),('Webcam Recognition','webcam','Record a signing sequence')]:
            with gr.Tab(title):
                if source=='webcam': gr.Markdown('Allow camera access, record, then stop and select **Translate recording**. This translates a completed recording; live streaming is not implemented.')
                with gr.Row():
                    video=gr.Video(sources=[source],format='mp4',label=label,webcam_options=gr.WebcamOptions(mirror=False),include_audio=False,max_length=60)
                    with gr.Column():
                        text=gr.Textbox(label='English translation',lines=4,interactive=False)
                        audio=gr.Audio(label='English speech',type='filepath',interactive=False)
                        download=gr.DownloadButton(label='Download speech',interactive=True)
                        status=gr.Textbox(label='Status',value='Ready for a video. A trained checkpoint is required.',interactive=False)
                with gr.Row():
                    go=gr.Button('Translate' if source=='upload' else 'Translate recording',variant='primary')
                    clear=gr.Button('Clear' if source=='upload' else 'Record another sequence')
                go.click(translate,inputs=video,outputs=[text,audio,download,status],concurrency_limit=1)
                clear.click(lambda:(None,'',None,None,'Cleared. Ready for another video.'),outputs=[video,text,audio,download,status],queue=False)
        with gr.Tab('Model Information'):
            gr.Markdown(model_information(translator.checkpoint))
            gr.Markdown('**Scope:** English word-sequence prediction within the training vocabulary; unseen words map to `<unk>`. No claim of unrestricted translation, signer independence, or production accuracy. Best results require similar camera framing and signing domain. gTTS needs Internet access. No fake confidence score is displayed.')
        with gr.Tab('About SignVoice'):
            gr.Markdown('''## Making connected signing more accessible
An MCA research project studying temporal Indian Sign Language video translation into English text and speech.

**Pipeline:** video → OpenCV sampling → MediaPipe body, hands and face → shoulder normalization → BiLSTM encoder → attention decoder → English text → gTTS audio.

**Technologies:** Python, PyTorch, NumPy, Pandas, MediaPipe, OpenCV, Gradio, scikit-learn and Google Colab.

Videos are processed on the machine hosting this app. Generated text is sent to Google's TTS service when speech is requested automatically after translation. Use consenting participants. Files may remain in Gradio's temporary cache and `outputs/audio`; delete these after your session.''')
    return demo

if __name__=='__main__':
    build_app().queue(max_size=8).launch(server_name='0.0.0.0',server_port=int(os.environ.get('PORT','7860')),share=False)
