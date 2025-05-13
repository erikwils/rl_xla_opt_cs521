import os
import sys
import json
from pprint import pprint

# Add parent directory to path to import modules from jax_hlo
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from jax_hlo.hlo_parser import parse_hlo_from_filepath
from jax_hlo.hlo_feature_extraction import extract_all_features


def main():
    # Path to example HLO file:
    file_str = "conv_relu_hlo.txt"
    hlo_file = os.path.join(project_dir, "jax_hlo", "hlo_data", file_str)

    if not os.path.exists(hlo_file):
        print(f"[ERROR] File Not Found: {hlo_file}")
        return 1

    # parse hlo file
    try:
        hlo_module = parse_hlo_from_filepath(hlo_file)

        # get entry computation
        print(hlo_module)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    sys.exit(main())
