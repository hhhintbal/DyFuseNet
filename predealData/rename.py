import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from natsort import natsorted
from tqdm import tqdm  # 关键修改：明确导入tqdm
from pathlib import Path
class SmartRenamer:
    def __init__(self, folderpath, startindex=0, dryrun=False):
        self.folder = os.path.abspath(folderpath)
        self.start = startindex
        self.dryrun = dryrun
        self.logger = self.setup_logger()
        self.files = self.loadfiles()

    def setup_logger(self):
        logger = logging.getLogger('SmartRenamer')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def loadfiles(self):
        files = sorted(Path(self.folder).iterdir(), key=lambda x: x.name.lower())
        return [f for f in files if f.is_file()]

    def rename_files(self):
        for idx, file in tqdm(enumerate(self.files, start=self.start),
                              total=len(self.files),
                              desc="Processing files"):
            new_name = f"{idx}.tif"
            new_path = file.parent / new_name
            if not self.dryrun:
                file.rename(new_path)
            self.logger.info(f"Renamed: {file.name} -> {new_name}")

    def run(self):
        self.rename_files()

# 使用示例
if __name__ == "__main__":
    renamer = SmartRenamer("/root/autodl-tmp/py_testASMF/predealData/dealed_data/RGB/10/", startindex=0)
    renamer.run()
