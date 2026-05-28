"""
@file        clean_pychaces.py
@author      Lucía Parreño Legorburo
@brief       Deletes all pycaches from the folders
"""

import os
import shutil

def delete_all_pycaches(root:str = ".", show_msg:bool = False):
    """
    Goes through all the folders to eliminate pychache files

    Args:
        root (str, optional): root to find pycaches. Defaults to ".".
        show_msg (bool, optional): show where the pycaches eliminated were from. Defaults to False.
    """
    for dirpath, dirnames, _ in os.walk(root):
        for dirname in dirnames:
            if dirname == "__pycache__":
                full_path = os.path.join(dirpath, dirname)
                if show_msg:
                    print(f"Eliminando: {full_path}")
                shutil.rmtree(full_path, ignore_errors=True)

