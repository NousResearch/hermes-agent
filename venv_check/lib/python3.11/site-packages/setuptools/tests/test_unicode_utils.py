from setuptools import unicode_utils


def test_filesys_decode_fs_encoding_is_None(monkeypatch):
    """
    Test filesys_decode does not raise TypeError when
    getfilesystemencoding returns None.
    """
    monkeypatch.setattr('sys.getfilesystemencoding', lambda: None)
    unicode_utils.filesys_decode(b'test')
