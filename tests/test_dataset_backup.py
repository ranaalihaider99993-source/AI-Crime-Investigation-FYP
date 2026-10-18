import io
import json
import tarfile
from pathlib import Path
import pytest
from scripts.dataset_backup import safe_extract, create_backup, restore_backup
from scripts import dataset_backup, review_paths
from tests.test_review_relocation import saved_review


@pytest.mark.parametrize('name,kind', [('../escape','file'),('/absolute','file'),('link','link'),('C:/escape','file'),('a\\b','file')])
def test_unsafe_archive_rejected(tmp_path,name,kind):
    archive=tmp_path/'test.tar'
    with tarfile.open(archive,'w') as out:
        member=tarfile.TarInfo(name)
        if kind=='link':
            member.type=tarfile.SYMTYPE;member.linkname='/outside';out.addfile(member)
        else:
            member.size=1;out.addfile(member,io.BytesIO(b'x'))
    with pytest.raises(ValueError):safe_extract(archive,tmp_path/'out',100)
    assert not (tmp_path/'out').exists()


def test_budget_and_collision(tmp_path):
    archive=tmp_path/'test.tar'
    with tarfile.open(archive,'w') as out:
        member=tarfile.TarInfo('a');member.size=2;out.addfile(member,io.BytesIO(b'ab'))
    with pytest.raises(ValueError):safe_extract(archive,tmp_path/'out',1)
    safe_extract(archive,tmp_path/'out',2)
    assert (tmp_path/'out/a').read_bytes()==b'ab'
    with pytest.raises(ValueError):safe_extract(archive,tmp_path/'out',2)


def test_backup_restore_and_tamper(saved_review,tmp_path,monkeypatch):
    directory,root,_,_=saved_review
    original=review_paths.recovery_evidence
    monkeypatch.setattr(dataset_backup,'recovery_evidence',lambda d:dict(original(d),recorded_versions=['1']))
    quota=dict(source='Google Drive API storageQuota',available_bytes=10**12)
    before={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    folder=create_backup(directory,root,quota)
    restored=restore_backup(directory)
    assert restored and restored!=root
    assert all(Path(p).read_bytes()==data for p,data in before.items())
    assert json.loads((folder/'provenance.json').read_text())['status']=='COMPLETE'
    with (folder/'dcsass-v1.tar').open('ab') as out:out.write(b'tamper')
    with pytest.raises(ValueError,match='SHA-256'):restore_backup(directory)


def test_missing_backup_does_not_download(tmp_path):
    assert restore_backup(tmp_path/'review') is None
