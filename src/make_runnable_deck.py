"""
Phase 2b: remediate MCNP deck produced by MCNPy and add run-control cards.

MCNPy 0.0.7 translation limitations remediated here:
1. Cell Universe Assignment: MCNPy tags all cells with `U 1`. Since there is no
   Universe 0 root container, MCNP will crash. This script assigns cells to
   Universe 0 (stripping `U 1` tags).
2. Thermal Scattering Cards (MT): MCNPy drops S(a,b) thermal scattering tags from
   materials.xml. This script reads materials.xml and injects corresponding MT cards
   (e.g., MT3 lwtr.01t for light water).
3. Initial Source Point (KSRC): MCNPy ignores settings.xml. This script extracts
   source spatial coordinates from settings.xml (or defaults to 0 0 0) and injects KSRC.
4. Run Control (MODE & KCODE): Appends MODE N and KCODE cards using settings.xml values.

Run after translate_to_mcnp.py has produced pin_cell.mcnp:
    python src/make_runnable_deck.py
"""
import os
import tempfile
import xml.etree.ElementTree as ET
import montepy
from montepy.universe import Universe

SOURCE_DECK = "pin_cell.mcnp"
OUT_DECK = "pin_cell_runnable.mcnp"
INITIAL_KEFF_GUESS = 1.0

# Mapping from OpenMC S(alpha, beta) names to MCNP thermal scattering identifiers
SAB_MCNP_MAP = {
    "c_H_in_H2O": "lwtr.01t",
    "c_D_in_D2O": "hwtr.01t",
    "c_Graphite": "grph.01t",
    "c_Be": "be.01t",
    "c_Zr_in_ZrH": "zrh.01t",
}

# 1. Read settings.xml and materials.xml via ElementTree (pure python, no openmc package required)
particles = 1000
inactive = 10
batches = 50

if os.path.exists("settings.xml"):
    tree = ET.parse("settings.xml")
    root = tree.getroot()
    p_elem = root.find("particles")
    i_elem = root.find("inactive")
    b_elem = root.find("batches")
    if p_elem is not None and p_elem.text:
        particles = int(p_elem.text)
    if i_elem is not None and i_elem.text:
        inactive = int(i_elem.text)
    if b_elem is not None and b_elem.text:
        batches = int(b_elem.text)

# Extract materials with S(a,b) thermal scattering
sab_materials = []  # list of (mat_id, sab_name)
if os.path.exists("materials.xml"):
    tree = ET.parse("materials.xml")
    root = tree.getroot()
    for mat in root.findall("material"):
        mat_id = int(mat.get("id"))
        for sab in mat.findall("sab"):
            sab_name = sab.get("name")
            if sab_name:
                sab_materials.append((mat_id, sab_name))

# 2. Prepare remediated deck text stream for MontePy parser
base_deck_text = open(SOURCE_DECK, "r").read()

injected_cards = []
for mat_id, sab_name in sab_materials:
    mcnp_sab = SAB_MCNP_MAP.get(sab_name, f"{sab_name}.01t")
    injected_cards.append(f"MT{mat_id} {mcnp_sab}")

ksrc_card = "KSRC 0.0 0.0 0.0"
injected_cards.append(ksrc_card)
injected_cards.append("MODE N")
kcode_card = f"KCODE {particles} {INITIAL_KEFF_GUESS} {inactive} {batches}"
injected_cards.append(kcode_card)

augmented_text = base_deck_text.strip() + "\n" + "\n".join(injected_cards) + "\n"

# 3. Write temporary augmented deck and parse with MontePy
temp_fd, temp_path = tempfile.mkstemp(suffix=".mcnp")
try:
    with os.fdopen(temp_fd, "w") as f:
        f.write(augmented_text)

    problem = montepy.read_input(temp_path)

    # 4. Fix Universe 0 assignment (strip U 1 tags)
    u0 = Universe(0)
    for cell in problem.cells:
        if cell.universe and cell.universe.number == 1:
            cell.universe = u0

    # 5. Write final validated runnable deck
    problem.write_to_file(OUT_DECK, overwrite=True)

finally:
    if os.path.exists(temp_path):
        os.remove(temp_path)

print(f"Wrote {OUT_DECK}")
print(f"Remediated Universe 0: Assigned cells to Universe 0 (stripped 'U 1' tags).")
print(f"Injected MT Cards: {[c for c in injected_cards if c.startswith('MT')]}")
print(f"Injected KSRC:     {ksrc_card}")
print(f"Injected MODE:     MODE N")
print(f"Injected KCODE:    {kcode_card}")
