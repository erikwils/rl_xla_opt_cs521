import os
import subprocess
import tempfile
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import time
from utils import convert_hlo_to_txt
from jax_hlo.hlo_feature_extraction import extract_all_features
from jax_hlo.hlo_parser import parse_hlo_from_filepath
import tempfile
import re

class XLAInterface:
    def __init__(self, xla_dir: str, verbose: bool = False):
        """
        Initialize XLA-Python interface for applying compiler passes.

        Args:
            xla_dir: Path to XLA directory
            verbose: Whether to print verbose information
        """

        self.verbose = verbose

        self.xla_dir = xla_dir
        self.hlo_opt_path = os.path.join(xla_dir, "bazel-bin/xla/hlo/tools/hlo-opt")

        if not os.path.exists(self.hlo_opt_path):
            raise ValueError(f"hlo_opt not found at {self.hlo_opt_path}")

        # Create a dedicated directory for optimized outputs in the reinforcement_learning folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.optimized_dir = os.path.join(script_dir, "optimized_hlo")
        os.makedirs(self.optimized_dir, exist_ok=True)

        if self.verbose:
            print(f"Optimized HLO files will be saved to: {self.optimized_dir}")

        self._available_passes = None

    def get_available_passes(self) -> List[str]:
        """
        Get list of available XLA optimization passes.

        Returns:
            List of pass names
        """
        if self._available_passes is not None:
            return self._available_passes

        cmd = [self.hlo_opt_path, "--list-passes"]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.xla_dir,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse the output to extract pass names
            # First, clean up the output and split by lines
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]

            # The output is comma-separated on a single line
            self._available_passes = []
            for line in lines:
                # Split each line by commas and add to the list of passes
                self._available_passes.extend([p.strip() for p in line.split(',') if p.strip()])

            if self.verbose:
                print(f"\nFound {len(self._available_passes)} available passes")

            return self._available_passes

        except subprocess.CalledProcessError as e:
            print(f"Error listing passes: {e.stderr}")
            return []

    def apply_pass(self, hlo_file: str, pass_name: str) -> Tuple[bool, Optional[str]]:
        """
        Apply an XLA optimization pass to an HLO file.

        Args:
            hlo_file: Path to HLO file to optimize
            pass_name: Name of the pass to apply

        Returns:
            Tuple of (success, output_file_path)
            If success is False, output_file_path will be None
        """
        # Create a descriptive filename based on the input file and pass name
        input_filename = os.path.basename(hlo_file)
        base_name = os.path.splitext(input_filename)[0]

        # get first part of base name (before any underscore)
        # this will prevent the name from growing with each pass
        if '_' in base_name:
            base_name = base_name.split('_')[0]
        
        # use end of timestamp for differentiation:
        timestamp = int(time.time() * 1000)
        output_filename = f"{base_name}_{timestamp}.hlo"
        output_file = os.path.join(self.optimized_dir, output_filename)

        # Build the command to run the pass:
        # Based on the error message, hlo-opt doesn't like the -o flag
        # Instead we'll use shell redirection (>) to save the output
        cmd = f"{self.hlo_opt_path} --passes={pass_name} {hlo_file} > {output_file}"

        if self.verbose:
            print(f"Running command: {cmd}")

        try:
            # Use shell=True to enable output redirection
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.xla_dir,
                capture_output=True,
                text=True,
                check=True
            )

            if self.verbose:
                print(f"Pass {pass_name} successfully applied")
                print(f"Output saved to: {output_file}")

            # Make sure the output file was actually created and has content
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return True, output_file
            else:
                print("Output file was not created or is empty")
                return False, None

        except subprocess.CalledProcessError as e:
            print(f"Error applying pass {pass_name}: {e.stderr}")
            # clean up output file if failure
            if os.path.exists(output_file):
                os.unlink(output_file)
            return False, None

    def extract_features(self, hlo_file: str) -> Dict[str, Any]:
        """
        Uses jax_hlo module to extract features from an hlo file.

        Args:
            hlo_file: Path to the HLO file

        Returns:
            numpy array of features for the HLO module
        """
        # pass the hlo txt file into the jax_hlo pipeline to extract the features

        try:
            hlo_module = parse_hlo_from_filepath(hlo_file)
            features_dict = extract_all_features(hlo_module) # type: ignore
            return features_dict
        except ValueError as e:
            error_msg = str(e)

            # check for common parsing errors:
            if "First line is not a valid HloModule header" in error_msg or "unexpected token" in error_msg.lower():
                # try preprocessing the file
                if self.verbose:
                    print(f"Preprocessing HLO file to remove annotations: {os.path.basename(hlo_file)}")
                
                preprocessed_file = self.preprocess_hlo_file(hlo_file)
                try:
                    hlo_module = parse_hlo_from_filepath(preprocessed_file)
                    features_dict = extract_all_features(hlo_module) # type: ignore

                    # cleanup
                    os.remove(preprocessed_file)

                    return features_dict
                
                except Exception as inner_e:
                    # cleanup even on failure:
                    if os.path.exists(preprocessed_file):
                        os.remove(preprocessed_file)
                    
                    if self.verbose:
                        print(f"Preprocessing didn't help: {str(inner_e)}")
                    
                    raise inner_e
            else:
                raise e
    
    def preprocess_hlo_file(self, input_file_path):
        """
        Preprocess an HLO file to make it compatible with the parser.
        Creates a temporary preprocessed version.

        # TODO: NOTE -- This is an extremely temporary solution!
        """

        # create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.hlo', delete=False) as temp_file:
            temp_path = temp_file.name
        
        with open(input_file_path, 'r') as input_file, open(temp_path, 'w') as output_file:
            # process the HLO file line by line
            for line in input_file:
                # Remove index annotations like /*index=5*/
                line = re.sub(r'/\*index=[0-9]+\*/', '', line)

                # remove origin annotations like origin = {{"constant_one"}}
                line = re.sub(r'origin=\{\{.*?\}\}', '', line)

                # write the processed line
                output_file.write(line)
        
        return temp_path




if __name__ == "__main__":

    print()
    xla_dir = "/Users/rayaanfaruqi/Documents/CS521/Final_Project/xla"
    xla_interface = XLAInterface(xla_dir=xla_dir, verbose=True)

    # # list available passes
    # available_passes = xla_interface.get_available_passes()
    # print("\nAvailable XLA passes:")
    # for i, pass_name in enumerate(available_passes):
    #     print(f"{i+1}. {pass_name}")

    file_str = "/Users/rayaanfaruqi/Documents/CS521/Final_Project/rl_xla_opt_cs521/reinforcement_learning/optimized_hlo/conv_relu_hlo_algsimp_opt.hlo"
    features = xla_interface.extract_features(file_str)
    print(features)

    # # test on a sample HLO file if available
    # import glob
    # sample_files = glob.glob("../jax_hlo/hlo_data/*.txt")

    # if sample_files:
    #     print(f"\nTesting with sample file: {sample_files[0]}")

