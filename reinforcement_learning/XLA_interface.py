import os
import subprocess
import tempfile
import numpy as np
from typing import List, Dict, Optional, Tuple
import time

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
        
        # Add timestamp to prevent filename collisions
        timestamp = int(time.time() * 1000)
        output_filename = f"{base_name}_{pass_name}_{timestamp}.hlo"
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

    def extract_features(self, hlo_file: str) -> np.ndarray:
        ...
    

if __name__ == "__main__":
    
    print()
    xla_dir = "/Users/rayaanfaruqi/Documents/CS521/Final_Project/xla"
    xla_interface = XLAInterface(xla_dir=xla_dir, verbose=True)

    # list available passes
    available_passes = xla_interface.get_available_passes()
    print("\nAvailable XLA passes:")
    for i, pass_name in enumerate(available_passes):
        print(f"{i+1}. {pass_name}")

    # # test on a sample HLO file if available
    # import glob
    # sample_files = glob.glob("../jax_hlo/hlo_data/*.txt")

    # if sample_files:
    #     print(f"\nTesting with sample file: {sample_files[0]}")

