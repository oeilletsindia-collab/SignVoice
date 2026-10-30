import io,tarfile
import pytest
from scripts.extract_archive import extract

def test_archive_traversal_rejected(tmp_path):
    p=tmp_path/'bad.tar'
    with tarfile.open(p,'w') as t:
        entry=tarfile.TarInfo('../escape'); entry.size=1; t.addfile(entry,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='Unsafe'): extract(p,tmp_path/'out')
    assert not (tmp_path/'escape').exists()
