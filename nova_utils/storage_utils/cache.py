from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import requests
import shutil
from tqdm.auto import tqdm
from pathlib import Path

def retreive_from_url(url, fp):

    with requests.get(url, stream=True, headers={'Accept-Encoding': None}) as r:

        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"),0)

        # implement progress bar via tqdm
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="")as raw:

            # save the output to a file
            with open(fp, 'wb')as output:
                shutil.copyfileobj(raw, output)


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

if __name__ == '__main__':
    retreive_from_url("https://github.com/saveli/syncpy/archive/refs/heads/master.zip",'blub.zip')