import re
import os

def check_values(path, ds_name):
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        content = f.read()
    marker = f"Dataset (\"{ds_name}\")"
    start = 0
    while True:
        idx = content.find(marker, start)
        if idx == -1: break
        b_start = content.find("{", idx)
        stack = 1
        curr = b_start + 1
        while stack > 0 and curr < len(content):
            if content[curr] == "{": stack += 1
            elif content[curr] == "}": stack -= 1
            curr += 1
        body = content[b_start:curr]
        v_match = re.search(r"Values\s*\((\d+)\)\s*\{(.*?)\}", body, re.DOTALL)
        if v_match:
            declared = v_match.group(1)
            actual = len(v_match.group(2).split())
            print(f"{path} | {ds_name}: Declared={declared}, Actual={actual}")
        start = curr

check_values("SiC_electron/output/c14/step6_output/n4_c14_optical_generation.dat", "OpticalGeneration")
check_values("Silicon_electron/output/step6_output/n51_optical_generation.dat", "OpticalGeneration")
