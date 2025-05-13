#!/usr/bin/env python3
"""
Test script for XLA interface
This script tests applying optimization passes to HLO files from the jax_hlo/hlo_data directory
"""

import os
import sys
from XLA_interface import XLAInterface

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    xla_dir = os.path.join(os.path.dirname(project_dir), "xla")
    
    # Path to example HLO file
    hlo_file = os.path.join(project_dir, "jax_hlo", "hlo_data", "test_algsimp.txt")
    
    if not os.path.exists(hlo_file):
        print(f"Error: HLO file not found at {hlo_file}")
        return 1
    
    print(f"Testing with HLO file: {hlo_file}")
    print(f"Using XLA directory: {xla_dir}")
    
    # Initialize XLA interface
    xla_interface = XLAInterface(xla_dir=xla_dir, verbose=True)
    
    # Get available passes
    passes = xla_interface.get_available_passes()
    # print(f"\nFound {len(passes)} available optimization passes")
    
    # Select specific passes to test
    test_passes = ["algsimp"]
    passes_to_run = [p for p in test_passes if p in passes]
    
    if not passes_to_run:
        print("None of the specified test passes are available")
        return 1
    
    # Apply passes one by one
    for pass_name in passes_to_run:
        print(f"\n\nApplying pass: {pass_name}")
        success, output_file = xla_interface.apply_pass(hlo_file, pass_name)
        
        if success:
            # print(f"Success! Output written to: {output_file}")
            
            # Display the first 10 lines of the output
            print("\nFirst 10 lines of output:")
            try:
                with open(output_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 10:
                            break
                        print(f"  {line.strip()}")
            except Exception as e:
                print(f"Error reading output file: {e}")
        else:
            print(f"Failed to apply {pass_name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 