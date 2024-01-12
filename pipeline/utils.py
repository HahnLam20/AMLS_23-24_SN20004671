from shutil import rmtree
from os import path, mkdir

def clean_folder(folder: str):
    if path.exists(folder):
        rmtree(folder, ignore_errors=True)
    mkdir(folder)
