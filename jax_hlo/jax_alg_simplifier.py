import os
import random
import re

output_dir = "hlo_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# --- Constants and helper functions for file generation ---

def generate_constants_section(available_constants_for_file):
    constants_map = {
        "constant_zero": "  constant_zero = f32[] constant(0)",
        "constant_one": "  constant_one = f32[] constant(1)",
        "constant_two": "  constant_two = f32[] constant(2)",
        "constant_neg_one": "  constant_neg_one = f32[] constant(-1)",
        "constant_half": "  constant_half = f32[] constant(0.5)",
        "constant_pi": "  constant_pi = f32[] constant(3.14159)",
        "constant_e": "  constant_e = f32[] constant(2.71828)"
    }
    return "\n".join([constants_map[c] for c in available_constants_for_file if c in constants_map])


def generate_arrays_section(shape, available_constants_for_file):
    arrays = []
    if "constant_zero" in available_constants_for_file:
        arrays.append(f"  zero_array = f32{shape} broadcast(constant_zero), dimensions={{}}")
    if "constant_one" in available_constants_for_file:
        arrays.append(f"  one_array = f32{shape} broadcast(constant_one), dimensions={{}}")
    if "constant_two" in available_constants_for_file:
        arrays.append(f"  two_array = f32{shape} broadcast(constant_two), dimensions={{}}")
    if "constant_neg_one" in available_constants_for_file:
        arrays.append(f"  neg_one_array = f32{shape} broadcast(constant_neg_one), dimensions={{}}")
    if "constant_half" in available_constants_for_file:
        arrays.append(f"  half_array = f32{shape} broadcast(constant_half), dimensions={{}}")
    if "constant_pi" in available_constants_for_file:
        arrays.append(f"  pi_array = f32{shape} broadcast(constant_pi), dimensions={{}}")
    if "constant_e" in available_constants_for_file:
        arrays.append(f"  e_array = f32{shape} broadcast(constant_e), dimensions={{}}")
    return "\n".join(arrays)


simplification_patterns = [
    {"name": "add_zero", "template": "  add_zero_{idx} = f32{shape} add({array}, zero_array)",
     "needs_constants": ["constant_zero"], "needs_arrays": ["zero_array"]},
    {"name": "mul_one", "template": "  mul_one_{idx} = f32{shape} multiply({array}, one_array)",
     "needs_constants": ["constant_one"], "needs_arrays": ["one_array"]},
    {"name": "sub_zero", "template": "  sub_zero_{idx} = f32{shape} subtract({array}, zero_array)",
     "needs_constants": ["constant_zero"], "needs_arrays": ["zero_array"]},
    {"name": "mul_zero", "template": "  mul_zero_{idx} = f32{shape} multiply({array}, zero_array)",
     "needs_constants": ["constant_zero"], "needs_arrays": ["zero_array"]},
    {"name": "power_one", "template": "  power_one_{idx} = f32{shape} power({array}, one_array)",
     "needs_constants": ["constant_one"], "needs_arrays": ["one_array"]},
    {"name": "sub_self", "template": "  sub_self_{idx} = f32{shape} subtract({array}, {array})", "needs_constants": [],
     "needs_arrays": []},
    {"name": "add_neg", "template": "  add_neg_{idx} = f32{shape} add({array}, neg_one_array)",
     "needs_constants": ["constant_neg_one"], "needs_arrays": ["neg_one_array"]},
    {"name": "div_one", "template": "  div_one_{idx} = f32{shape} divide({array}, one_array)",
     "needs_constants": ["constant_one"], "needs_arrays": ["one_array"]},
    {"name": "mul_two", "template": "  mul_two_{idx} = f32{shape} multiply({array}, two_array)",
     "needs_constants": ["constant_two"], "needs_arrays": ["two_array"]},
    {"name": "power_zero", "template": "  power_zero_{idx} = f32{shape} power({array}, zero_array)",
     "needs_constants": ["constant_zero"], "needs_arrays": ["zero_array"]},
    {"name": "complex_add_mul",
     "template": "  complex_add_mul_{idx} = f32{shape} add(multiply({array}, one_array), zero_array)",
     "needs_constants": ["constant_one", "constant_zero"], "needs_arrays": ["one_array", "zero_array"]},
    {"name": "complex_mul_add",
     "template": "  complex_mul_add_{idx} = f32{shape} multiply(add({array}, zero_array), one_array)",
     "needs_constants": ["constant_zero", "constant_one"], "needs_arrays": ["zero_array", "one_array"]},
    {"name": "complex_sub_add",
     "template": "  complex_sub_add_{idx} = f32{shape} subtract(add({array}, {array2}), {array2})",
     "needs_constants": [], "needs_arrays": []},
    {"name": "complex_pow_mul",
     "template": "  complex_pow_mul_{idx} = f32{shape} power(multiply({array}, one_array), one_array)",
     "needs_constants": ["constant_one"], "needs_arrays": ["one_array"]},
    {"name": "double_neg",
     "template": "  double_neg_{idx} = f32{shape} multiply({array}, multiply(neg_one_array, neg_one_array))",
     "needs_constants": ["constant_neg_one"], "needs_arrays": ["neg_one_array"]},
    {"name": "square_sqrt", "template": "  square_sqrt_{idx} = f32{shape} power(sqrt({array}), two_array)",
     "needs_constants": ["constant_two"], "needs_arrays": ["two_array"]},
]
# --- End of helper functions and patterns ---

# Number of different HLO files to generate
num_files = 30
print(f"Generating {num_files} HLO files (no comments/headers) with synthetic algebraic patterns...")

all_constants_names = ["constant_zero", "constant_one", "constant_two", "constant_neg_one", "constant_half",
                       "constant_pi", "constant_e"]
all_arrays_names_map = {
    "constant_zero": "zero_array", "constant_one": "one_array", "constant_two": "two_array",
    "constant_neg_one": "neg_one_array", "constant_half": "half_array",
    "constant_pi": "pi_array", "constant_e": "e_array"
}

for file_idx in range(1, num_files + 1):
    shape = random.choice(["[4,4]", "[8,8]", "[16]", "[2,3,4]", "[3,5]"])
    module_name = f"test_algebraic_simplifier_{file_idx}"

    # --- Determine available constants and arrays for this file ---
    available_constants_in_file = set(["constant_zero", "constant_one", "constant_two"])
    num_additional_consts = random.randint(1, len(all_constants_names) - len(available_constants_in_file))
    additional_consts = random.sample([c for c in all_constants_names if c not in available_constants_in_file],
                                      k=num_additional_consts)
    available_constants_in_file.update(additional_consts)
    available_arrays_in_file = [all_arrays_names_map[c] for c in available_constants_in_file if
                                c in all_arrays_names_map]

    hlo_content_parts = []
    hlo_content_parts.append(f"HloModule {module_name}")
    hlo_content_parts.append(
        f"ENTRY main {{")

    # --- Build the HLO content body ---
    constants_hlo = generate_constants_section(list(available_constants_in_file))
    if constants_hlo:
        hlo_content_parts.append(constants_hlo)

    arrays_hlo = generate_arrays_section(shape, list(available_constants_in_file))
    if arrays_hlo:
        hlo_content_parts.append(arrays_hlo)

    # Selecting simplification patterns
    possible_patterns_for_file = []
    for p in simplification_patterns:
        can_form_pattern = True
        for needed_c in p.get("needs_constants", []):
            if needed_c not in available_constants_in_file: can_form_pattern = False; break
        if not can_form_pattern: continue
        for needed_a_type in p.get("needs_arrays", []):
            if needed_a_type not in available_arrays_in_file: can_form_pattern = False; break
        if not can_form_pattern: continue
        if "{array}" in p["template"] and not available_arrays_in_file: can_form_pattern = False
        if "{array2}" in p["template"] and len(available_arrays_in_file) < 2:
            if not (p["name"] == "sub_self" and len(available_arrays_in_file) >= 1): can_form_pattern = False
        if can_form_pattern: possible_patterns_for_file.append(p)

    if not possible_patterns_for_file:
        print(f"Warning: No possible patterns for file {file_idx}. Generating minimal HLO.")
        if "constant_zero" not in available_constants_in_file:
            hlo_content_parts.append("  constant_zero = f32[] constant(0)")
        hlo_content_parts.append("  ROOT result = f32[] constant_zero")
        hlo_content_parts.append("}")  # Close ENTRY
        final_hlo_content = "\n".join(hlo_content_parts)
        filename = os.path.join(output_dir, f"{module_name}.hlo")
        with open(filename, "w") as f:
            f.write(final_hlo_content)
        continue

    num_patterns_to_select = random.randint(min(4, len(possible_patterns_for_file)),
                                            min(8, len(possible_patterns_for_file)))
    selected_patterns = random.sample(possible_patterns_for_file, k=num_patterns_to_select)

    operations_names_for_root = []
    for i, pattern in enumerate(selected_patterns):
        op_idx = f"{file_idx}_{i}"
        array_for_pattern = random.choice(available_arrays_in_file) if available_arrays_in_file else "one_array"
        array2_for_pattern = random.choice(available_arrays_in_file) if len(
            available_arrays_in_file) > 1 else array_for_pattern

        operation = pattern["template"].format(idx=op_idx, shape=shape, array=array_for_pattern,
                                               array2=array2_for_pattern)
        hlo_content_parts.append(operation)  # Adding operation to file

        op_name_for_root = f"{pattern['name']}_{op_idx}"
        operations_names_for_root.append(op_name_for_root)

    # Add complex combinations
    if operations_names_for_root and random.random() > 0.3:
        num_combos = random.randint(1, min(3, len(operations_names_for_root)))
        for i in range(num_combos):
            combo_idx = f"{file_idx}_combo_{i}"
            op1 = random.choice(operations_names_for_root)
            op2 = random.choice(operations_names_for_root)
            combo_type = random.choice(["add", "multiply", "subtract"])
            combo = f"  complex_combo_{combo_idx} = f32{shape} {combo_type}({op1}, {op2})"
            hlo_content_parts.append(combo)  # Add operation to file
            operations_names_for_root.append(f"complex_combo_{combo_idx}")

    # Build ROOT result
    if operations_names_for_root:
        root_tuple_elements = ",\n    ".join(operations_names_for_root)
        tuple_types = ", ".join([f'f32{shape}'] * len(operations_names_for_root))
        hlo_content_parts.append(f"  ROOT result = ({tuple_types}) tuple(\n    {root_tuple_elements}\n  )")
    else:
        if "constant_zero" not in "\n".join(hlo_content_parts):
            hlo_content_parts.insert(2, "  constant_zero = f32[] constant(0)")
        hlo_content_parts.append("  ROOT result = f32[] constant_zero")

    hlo_content_parts.append("}")
    final_hlo_content = "\n".join(hlo_content_parts)

    # Write to file
    filename = os.path.join(output_dir, f"{module_name}.hlo")
    with open(filename, "w") as f:
        f.write(final_hlo_content)

print(f"Successfully generated {num_files} HLO files in the '{output_dir}' directory (no comments/headers).")