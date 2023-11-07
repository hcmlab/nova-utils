"""Utilities for file download and caching. Partially transferred from https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/utils/data_utils.py#L148-L277"""
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib
import zipfile
import requests
import mimetypes
from tempfile import NamedTemporaryFile
from pathlib import Path
from urllib import request, error
from zipfile import ZipFile

def retreive_from_url(url, fp=None):

    with requests.get(url, stream=True, headers={'Accept-Encoding': None}) as r:

        if fp is None:
            suffix = mimetypes.guess_extension(r.headers.get('content-type'))
            tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            fp = Path(tmp_file.name)
        # save the output to a file
        with open(fp, 'wb')as output:
            shutil.copyfileobj(r.raw, output)

        return fp, r.headers
def _resolve_hasher(algorithm, file_hash=None):
    """Returns hash algorithm as hashlib function."""
    if algorithm == "sha256":
        return hashlib.sha256()

    if algorithm == "auto" and file_hash is not None and len(file_hash) == 64:
        return hashlib.sha256()

    # This is used only for legacy purposes.
    return hashlib.md5()


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    Example:

    ```python
    _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    Args:
        fpath: path to the file being validated
        algorithm: hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
            The default `'auto'` detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        The file hash
    """
    if isinstance(algorithm, str):
        hasher = _resolve_hasher(algorithm)
    else:
        hasher = algorithm

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def _extract_archive(file_path, path=".", archive_format="auto"):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Args:
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    file_path = file_path
    path = path

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False

def retreive_and_unzip(url, extract_to='./tmp', tmp_dir='./tmp'):
    fn = url.split('/')[-1]
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_file = tmp_dir / fn

    if tmp_file.is_file():
        print(f'File {tmp_file} already exists. Skipping download.')
    else:
        retreive_from_url(url, tmp_file)

    with ZipFile(tmp_file) as zObject:
        zObject.extractall(path=Path(extract_to))


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    Args:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        Whether the file is valid
    """
    hasher = _resolve_hasher(algorithm, file_hash)

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def get_file(
        fname,
        origin,
        untar=False,
        md5_hash=None,
        file_hash=None,
        cache_subdir="downloads",
        hash_algorithm="auto",
        extract=False,
        archive_format="auto",
        cache_dir=None,
        tmp_dir=None
):
    """Downloads a file from a URL if it not already in the cache.

    By default, the file at the url `origin` is downloaded to the
    cache_dir '~/.hcai_downloads', placed in the cache_subdir `downloads`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.hcai_downloads/downloads/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    Example:

    ```python
    #TODO
    ```

    Args:
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        untar: Deprecated in favor of `extract` argument.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of `file_hash` argument.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are `'md5'`, `'sha256'`, and `'auto'`.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are `'auto'`, `'tar'`, `'zip'`, and `None`.
            `'tar'` includes tar, tar.gz, and tar.bz files.
            The default `'auto'` corresponds to `['tar', 'zip']`.
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the default directory `~/.nova_utils/`.
        tmp_dir: Location to download temporary files, when no tmp_dir is specificed
            cache dir is used.

    Returns:
        Path to the downloaded file
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".nova_utils"
    else:
        cache_dir = Path(cache_dir)

    if tmp_dir is None:
        tmp_dir = Path(cache_dir)
    else:
        tmp_dir = Path(tmp_dir)

    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = "md5"

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if not os.access(cache_dir, os.W_OK):
        cache_dir = os.path.join("/tmp", ".hcai_models")

    data_dir = cache_dir / cache_subdir
    data_dir.mkdir(parents=True, exist_ok=True)

    #TODO if a zipfile is loaded we only compare the hash of the zip file. If the original zip file is deleted it
    # will be downloaded again every time

    if untar:
        untar_fpath = data_dir / fname
        fpath = untar_fpath.parent / (untar_fpath.name  + '.tar.gz')
    else:
        fpath = data_dir / fname

    download = False
    if fpath.exists():
        # File found; verify integrity if a hash was provided.
        if file_hash:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print(
                    "A local file was found, but it seems to be "
                    "incomplete or outdated because the "
                    + hash_algorithm
                    + " file hash does not match the original value of "
                    + file_hash
                    + " so we will re-download the data."
                )
                download = True
    else:
        download = True

    if download:
        print("Downloading data from", origin)

        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                tmp_path = tmp_dir / fname
                request.urlretrieve(origin, tmp_path)
                shutil.move(tmp_path, fpath)
            except urllib.error.HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, str(e)))
            except urllib.error.URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if untar:
        if not untar_fpath.exists():
            _extract_archive(fpath, untar_fpath, archive_format="tar")
            #fpath.unlink()
        return untar_fpath

    if extract:
        _extract_archive(fpath, data_dir, archive_format)

    return fpath


if __name__ == '__main__':
    #retreive_from_url("https://github.com/saveli/syncpy/archive/refs/heads/master.zip",'blub.zip')
    cache_dir = '/Users/dominikschiller/Work/github/nova-server/cache'
    tmp_dir = '/Users/dominikschiller/Work/github/nova-server/tmp'
    url = 'https://hcm-lab.de/cloud/s/FNiyD76YAs33D3D/download?path=%2Fblazeface&files=blazeface.pth'

    # Download file and get filepath
    local_fp = get_file(fname='blazeface.pth', origin=url)

    # Untar local file
    local_fp = get_file(fname='test', origin=url, untar=True)

    # Unzip local file
    local_fp = get_file(fname='blazeface_test.pth.zip', origin=url, extract=True)

