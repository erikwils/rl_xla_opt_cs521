import os
import tempfile
import shutil
from typing import Optional
import atexit
import sys

# Add parent directory to path to import jax_hlo
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)
from jax_hlo.hlo_parser import parse_hlo_from_filepath


def convert_hlo_to_txt(hlo_file_path : str) -> str:
    """
    Convert an HLO file to a temporary txt file for compatibility with jax_hlo module.

    Returns filepath to the temporary txt file.
    """
    # Create temporary file with .txt extension in a persistent temp directory
    base_name = os.path.basename(hlo_file_path).split('.')[0]
    with tempfile.NamedTemporaryFile(suffix='.txt', prefix=f"{base_name}_", delete=False) as temp_file:
        temp_path = temp_file.name

    shutil.copy(hlo_file_path, temp_path) # copy content from hlo file to txt


    return temp_path


