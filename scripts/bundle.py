"""Produce a source notebook bundle without caches, data or private files."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def bundle(output=None):
    output=Path(output) if output else ROOT.parent/'SignVoice-project.zip'
    excluded={'.git','.venv','__pycache__','.pytest_cache','data','outputs','.agents','.codex'}
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(ROOT.rglob('*')):
            relative=p.relative_to(ROOT)
            if p.is_file() and not p.is_symlink() and not (set(relative.parts)&excluded):
                if p.suffix in {'.pt','.pth','.mp3','.mp4','.avi','.mov'}: continue
                archive.write(p,Path('SignVoice')/relative)
    print(output)
if __name__=='__main__': bundle()
