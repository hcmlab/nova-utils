import os
def get_tmp_dir():
    return os.environ.get('TMP_DIR', './tmp')