from shutil import rmtree
from os import path, mkdir

#define function to clean folders before each task is run
def clean_folder(folder):
    if path.exists(folder):
        rmtree(folder, ignore_errors=True)
    mkdir(folder)
