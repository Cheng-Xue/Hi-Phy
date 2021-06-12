import os
import sys
import argparse
target_level_path = '../level_variations/generated_levels/fourth generation'

templates_to_run = []
levels = sorted(os.listdir(target_level_path))
for level in levels:
    capabilities = sorted(os.listdir(os.path.join(target_level_path, level)))
    for capability in capabilities:
        sys.stdout.write("{}_{}\n".format(level, capability))