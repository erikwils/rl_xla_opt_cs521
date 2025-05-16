import os
import sys
import glob
import re
from XLA_interface import XLAInterface

def format_hlo_files():
    # Configure paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    xla_dir = "/Users/rayaanfaruqi/Documents/CS521/Final_Project/xla"
    hlo_data_dir = os.path.join(project_dir, "jax_hlo/hlo_data")
    output_dir = os.path.join(script_dir, "optimized_hlo")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize XLA interface
    interface = XLAInterface(xla_dir=xla_dir)
    
    # Get list of available passes
    passes = interface.get_available_passes()
    
    # For reformatting purposes, we'll use the 'nop-pass' or another simple pass
    # that doesn't modify the structure significantly
    # Let's find a pass that's likely to be a no-op or minimal change
    reformatting_pass = None
    for candidate in ["zero_sized_hlo_elimination", "nop", "hlo_dce"]:
        if candidate in passes:
            reformatting_pass = candidate
            break
    
    if not reformatting_pass:
        # If we can't find a preferred pass, use the first one
        reformatting_pass = passes[0]
    
    print(f"Using pass '{reformatting_pass}' for reformatting")
    
    # Find all HLO files in the data directory - first get .hlo files
    hlo_files = glob.glob(os.path.join(hlo_data_dir, "*.hlo"))
    
    # Get list of base filenames (without extension) of the .hlo files
    hlo_base_names = set(os.path.splitext(os.path.basename(f))[0] for f in hlo_files)
    
    # Add .txt files only if their base name doesn't already exist as a .hlo file
    for txt_file in glob.glob(os.path.join(hlo_data_dir, "*.txt")):
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        if base_name not in hlo_base_names:
            hlo_files.append(txt_file)
            hlo_base_names.add(base_name)
    
    if not hlo_files:
        print(f"No HLO files found in {hlo_data_dir}")
        return
    
    # Check for files already in the output directory
    existing_files = set(os.path.basename(f) for f in glob.glob(os.path.join(output_dir, "*.hlo")))
    print(f"Found {len(existing_files)} existing files in output directory")
    
    # Debug information
    print(f"Found {len(hlo_files)} unique HLO files to process")
    print(f"Base filenames: {sorted(list(hlo_base_names))}")
    
    # First, clear the output directory to avoid duplicates
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Cleared existing files in {output_dir}")
    
    # Process each file
    success_count = 0
    processed_files = []
    
    for file_path in hlo_files:
        # Get base name and ensure output has .hlo extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_name = f"{base_name}.hlo"
        
        print(f"\nProcessing {os.path.basename(file_path)}...")
        
        try:
            # Apply the pass to reformat the file
            success, output_path = interface.apply_pass(file_path, reformatting_pass)
            
            if success and output_path is not None:
                # Read the content of the reformatted file
                with open(output_path, 'r') as f:
                    content = f.read()
                
                # Save with consistent .hlo extension to the output directory
                output_file = os.path.join(output_dir, output_file_name)
                with open(output_file, 'w') as f:
                    f.write(content)
                
                # Keep track of successfully processed files
                processed_files.append(output_file_name)
                
                print(f"✅ Successfully reformatted to {output_file_name}")
                
                # Verify we can extract features from the new file
                try:
                    features = interface.extract_features(output_file)
                    
                    # Calculate cost to verify the file is usable
                    excluded_keys = ['source_file', 'graph_nodes', 'graph_edge_links', 'total_bytes',
                                   'input_bytes', 'output_bytes', 'elemwise_ratio', 'mixed_precision']
                    
                    total_ops = 0
                    for op_name, value in features.items():
                        if op_name not in excluded_keys and isinstance(value, (int, float)):
                            total_ops += value
                    
                    memory_cost = features.get('total_bytes', 0) / 1000.0
                    graph_nodes = features.get('graph_nodes', [])
                    graph_complexity = len(graph_nodes) * 10
                    
                    total_cost = total_ops * 10.0 + memory_cost * 1.0 + graph_complexity * 0.1
                    
                    print(f"  - Initial cost: {total_cost:.4f}")
                    success_count += 1
                    
                except Exception as e:
                    print(f"  - Warning: File was reformatted but feature extraction failed: {str(e)}")
            else:
                print(f"❌ Failed to reformat {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
    
    # Clean up temporary files - look for files with timestamp patterns
    # Example: black_1747430851455.hlo
    print("\nCleaning up temporary files...")
    temp_file_pattern = re.compile(r'^[a-zA-Z0-9_]+_\d+\.hlo$')
    keep_files = set(processed_files)
    removed_count = 0
    
    for file_name in os.listdir(output_dir):
        if os.path.isfile(os.path.join(output_dir, file_name)) and file_name.endswith('.hlo'):
            # If it's not in our list of properly named files and it looks like a temp file
            if file_name not in keep_files and temp_file_pattern.match(file_name):
                os.remove(os.path.join(output_dir, file_name))
                removed_count += 1
    
    print(f"Removed {removed_count} temporary files")
    
    # Count final files
    final_files = glob.glob(os.path.join(output_dir, "*.hlo"))
    print(f"\nReformatting complete: {success_count}/{len(hlo_files)} files successfully processed")
    print(f"{len(final_files)} files now in output directory")
    print(f"Reformatted files are available in: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    format_hlo_files()
