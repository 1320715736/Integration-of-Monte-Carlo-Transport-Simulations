import re
import os

def analyze_dat(filepath):
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    info_match = re.search(r'Info\s*\{([^}]+)\}', content, re.DOTALL)
    info_stats = {}
    if info_match:
        info_text = info_match.group(1)
        for key in ['nb_vertices', 'nb_edges', 'nb_elements', 'nb_regions']:
            m = re.search(fr'{key}\s*=\s*(\d+)', info_text)
            info_stats[key] = int(m.group(1)) if m else None
        
        # Datasets/functions length
        ds_m = re.search(r'datasets\s*=\s*\[([^\]]+)\]', info_text, re.DOTALL)
        if ds_m:
            info_stats['datasets_len'] = len(re.findall(r'"[^"]+"', ds_m.group(1)))
        
        fn_m = re.search(r'functions\s*=\s*\[([^\]]+)\]', info_text, re.DOTALL)
        if fn_m:
            info_stats['functions_len'] = len(re.findall(r'"[^"]+"', fn_m.group(1)))

    datasets = []
    # Match Dataset blocks
    ds_blocks = re.finditer(r'Dataset\s*\(\s*"([^"]+)"\s*\)\s*\{([^}]+\})', content, re.DOTALL)
    for m in ds_blocks:
        name = m.group(1)
        body = m.group(2)
        
        func_m = re.search(r'function\s*=\s*"([^"]+)"', body)
        type_m = re.search(r'type\s*=\s*(\w+)', body)
        loc_m = re.search(r'location\s*=\s*(\w+)', body)
        val_m = re.search(r'validity\s*=\s*\[([^\]]+)\]', body)
        
        values_m = re.search(r'Values\s*\((\d+)\)\s*\{([^}]+)\}', body, re.DOTALL)
        val_count = None
        vals_list = []
        if values_m:
            val_count_declared = int(values_m.group(1))
            vals_list = values_m.group(2).split()
            val_count = len(vals_list)

        datasets.append({
            "name": name,
            "function": func_m.group(1) if func_m else None,
            "type": type_m.group(1) if type_m else None,
            "location": loc_m.group(1) if loc_m else None,
            "validity": val_m.group(1).strip() if val_m else None,
            "values_n": val_count_declared if values_m else 0,
            "values_actual": val_count,
            "nan_inf": any(x.lower() in ['nan', 'inf'] for x in vals_list)
        })

    # Braces balance
    open_b = content.count('{')
    close_b = content.count('}')
    open_p = content.count('(')
    close_p = content.count(')')

    return {
        "info": info_stats,
        "ds_count": len(datasets),
        "datasets": datasets,
        "braces": (open_b, close_b),
        "parens": (open_p, close_p),
        "closed_properly": content.strip().endswith('}')
    }

files = {
    "SiC_c14": "SiC_electron/output/c14/step6_output/n4_c14_optical_generation.dat",
    "Si_n51": "Silicon_electron/output/step6_output/n51_optical_generation.dat",
    "SiC_n4": "SiC_electron/n4_.dat"
}

results = {k: analyze_dat(v) for k, v in files.items()}

# Print Summary
for k, res in results.items():
    print(f"--- {k} ---")
    if "error" in res:
        print(res["error"])
        continue
    print(f"Info: {res['info']}")
    print(f"DS Count: {res['ds_count']}")
    print(f"Braces: {res['braces']}, Parens: {res['parens']}, Closed: {res['closed_properly']}")
    
    # Check for anomalies
    for ds in res['datasets']:
        if ds['values_n'] != ds['values_actual']:
            print(f"  [!] {ds['name']}: Declared {ds['values_n']} != Actual {ds['values_actual']}")
        if ds['nan_inf']:
            print(f"  [!] {ds['name']}: Contains NaN/Inf")

# Specific Comparison 1: SiC_c14 vs SiC_n4
print("\n--- Compare SiC_c14 vs SiC_n4 ---")
names_c14 = [d['name'] for d in results['SiC_c14']['datasets']]
names_n4 = [d['name'] for d in results['SiC_n4']['datasets']]
print(f"Added DS: {set(names_c14) - set(names_n4)}")
print(f"Removed DS: {set(names_n4) - set(names_c14)}")

# Specific Comparison 2: SiC_c14 vs Si_n51
print("\n--- Compare SiC_c14 vs Si_n51 Struct ---")
# Check the OpticalGeneration dataset in both
og_c14 = next((d for d in results['SiC_c14']['datasets'] if d['name'] == 'OpticalGeneration'), None)
og_si = next((d for d in results['Si_n51']['datasets'] if d['name'] == 'OpticalGeneration'), None)
if og_c14 and og_si:
    print(f"SiC_c14 OG: type={og_c14['type']}, loc={og_c14['location']}, val={og_c14['validity']}")
    print(f"Si_n51 OG:  type={og_si['type']}, loc={og_si['location']}, val={og_si['validity']}")

# Check validity consistency
def check_duplicate_validity(res):
    seen = {}
    for ds in res['datasets']:
        key = (ds['function'], ds['validity'])
        if key in seen:
            print(f"  [!] Duplicate Func/Validity: {ds['name']} and {seen[key]} both {key}")
        seen[key] = ds['name']

print("\nValidity Check SiC_c14:")
check_duplicate_validity(results['SiC_c14'])
