"""
Patch experiments.ipynb to auto-create figures directory before saving.
"""
import json

# Read the notebook
with open('experiments.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and patch cells with plt.savefig
patched = False
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this cell has plt.savefig but not os.makedirs
        source_text = ''.join(source) if isinstance(source, list) else source
        if 'plt.savefig' in source_text and 'os.makedirs' not in source_text:
            # Add the import and makedirs at the beginning
            if isinstance(source, list):
                # Insert after the first line or at the beginning
                new_lines = ["import os\n", "os.makedirs('figures', exist_ok=True)\n", "\n"]
                cell['source'] = new_lines + source
            else:
                cell['source'] = "import os\nos.makedirs('figures', exist_ok=True)\n\n" + source
            patched = True
            print(f"Patched a cell containing plt.savefig")

if patched:
    # Write the patched notebook
    with open('experiments.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook patched successfully!")
else:
    print("No cells needed patching.")
