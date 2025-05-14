import os
import sys
from XLA_interface import XLAInterface


xla_dir = "/Users/rayaanfaruqi/Documents/CS521/Final_Project/xla"
file_path = "/Users/rayaanfaruqi/Documents/CS521/Final_Project/rl_xla_opt_cs521/jax_hlo/hlo_data/mha_hlo.hlo"
interface = XLAInterface(xla_dir=xla_dir)
passes = interface.get_available_passes()

interface.apply_pass(file_path, passes[2])
