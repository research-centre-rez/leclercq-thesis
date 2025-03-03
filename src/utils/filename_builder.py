import sys
import os
def create_out_filename(filename:str, prefixes:list[str], suffixes:list[str]):
    '''
    Add suffixes to filename, used when saving a file into storage.
    Args:
        filename (str): Base filename, can contain the file extension
        prefixes (list[str]): Prefixes to be added before the filename
        suffixes (list[str]): Suffixes to be added after the filename
    Returns:
        new_filename (str): A new filename
    '''

    # Removes the file extension (works even if there is none present)
    # Removes relative path
    filename = os.path.basename(filename).split('.')[0]

    suff = '_'.join(suffixes)
    pref = '_'.join(prefixes)

    parts = [part for part in (pref, filename, suff) if part]
    new_f = '_'.join(parts)

    return new_f
