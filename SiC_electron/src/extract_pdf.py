import fitz
import re
import os

pdf_path = r"D:\VScode\project\Integration of Monte Carlo Transport Simulations\Integration_of_Monte_Carlo_Transport_Simulations_into_a_TCAD_Workflow_for_Electron_Detector_Developments.pdf"
if not os.path.exists(pdf_path):
    print(f"Error: File not found at {pdf_path}")
    exit(1)

doc = fitz.open(pdf_path)

keywords = [
    "optical generation", "generation rate", "normalization", "normalisation",
    "scaling", "scale factor", "incoming electrons", "primary particles",
    "histories", "integration time", "per second", "activity", "fluence",
    "flux", "transient", "steady-state", "DC", "eh pair", "electron-hole pair",
    "Sentaurus", "TCAD", "cm^-3 s^-1", "cm-3 s-1"
]

results = []

for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    blocks = page.get_text("blocks")
    for b in blocks:
        block_text = b[4]
        found_keywords = [kw for kw in keywords if kw.lower() in block_text.lower()]
        if found_keywords:
            results.append({
                "page": page_num + 1,
                "keywords": found_keywords,
                "text": block_text.strip()
            })

# Group blocks to provide more context if they are close
for res in results:
    print(f"--- Page {res['page']} (Keywords: {', '.join(res['keywords'])}) ---")
    print(res['text'])
    print("-" * 40)

doc.close()
