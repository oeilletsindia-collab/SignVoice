"""Extract regular archive members only, with traversal and size protections."""
import argparse,tarfile,zipfile,shutil
from pathlib import Path

def extract(archive,output,max_gb=12):
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=True); used=0
    def destination(name,size):
        nonlocal used
        path=(root/name).resolve()
        if not path.is_relative_to(root): raise ValueError('Unsafe archive path')
        used+=size
        if used>max_gb*1024**3: raise ValueError('Archive exceeds expanded-size budget')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise ValueError(f'Refusing to overwrite {path}')
        return path
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if item.is_dir(): continue
                if (item.external_attr >> 16)&0o170000 == 0o120000: raise ValueError('Archive symlinks are unsupported')
                path=destination(item.filename,item.file_size)
                with z.open(item) as src,path.open('wb') as dst: shutil.copyfileobj(src,dst)
    else:
        with tarfile.open(archive,'r:*') as tar:
            for item in tar:
                if item.isdir(): continue
                if not item.isfile(): raise ValueError('Only regular archive files are allowed')
                path=destination(item.name,item.size)
                with tar.extractfile(item) as src,path.open('wb') as dst: shutil.copyfileobj(src,dst)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('archive'); p.add_argument('output'); p.add_argument('--max-gb',type=float,default=12)
    a=p.parse_args(); extract(a.archive,a.output,a.max_gb)
