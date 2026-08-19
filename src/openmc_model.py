"""
Phase 1: OpenMC drafting.

Simple UO2 fuel pin in a water lattice cell, matching the dimensions used
in the project discussion:
    fuel radius      = 0.39 cm
    clad inner radius = 0.40 cm
    clad outer radius = 0.45 cm

Run directly to validate the model and export XML:
    python src/openmc_model.py
"""
import openmc

# ---- Explicit geometry variables (cm) ----------------------------------
FUEL_R = 0.39
CLAD_IR = 0.40
CLAD_OR = 0.45
PITCH = 1.26  # typical PWR-like pitch; adjust as needed

# ---- Materials -----------------------------------------------------------
fuel = openmc.Material(name="UO2 fuel")
fuel.add_element("U", 1, percent_type="ao", enrichment=3.0)
fuel.add_element("O", 2)
fuel.set_density("g/cm3", 10.4)

clad = openmc.Material(name="Zircaloy-4 clad")
clad.add_element("Zr", 0.982)
clad.add_element("Sn", 0.014)
clad.add_element("Fe", 0.002)
clad.add_element("Cr", 0.001)
clad.set_density("g/cm3", 6.55)

water = openmc.Material(name="Light water moderator")
water.add_element("H", 2)
water.add_element("O", 1)
water.set_density("g/cm3", 0.7)
water.add_s_alpha_beta("c_H_in_H2O")

materials = openmc.Materials([fuel, clad, water])
materials.export_to_xml()

# ---- Geometry --------------------------------------------------------
fuel_or = openmc.ZCylinder(r=FUEL_R)
clad_ir = openmc.ZCylinder(r=CLAD_IR)
clad_or = openmc.ZCylinder(r=CLAD_OR)

fuel_region = -fuel_or
gap_region = +fuel_or & -clad_ir
clad_region = +clad_ir & -clad_or
water_region = +clad_or  # ~ operator (complement) not needed here; bounded
                          # by the outer cell boundary below.

fuel_cell = openmc.Cell(name="fuel", fill=fuel, region=fuel_region)
gap_cell = openmc.Cell(name="gap", fill=None, region=gap_region)  # void
clad_cell = openmc.Cell(name="clad", fill=clad, region=clad_region)
water_cell = openmc.Cell(name="moderator", fill=water, region=water_region)

box = openmc.model.RectangularPrism(width=PITCH, height=PITCH,
                                     boundary_type="reflective")
water_cell.region &= -box

root_universe = openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, water_cell])
geometry = openmc.Geometry(root_universe)
geometry.export_to_xml()

# ---- Settings (minimal, enough to validate the model) -------------------
settings = openmc.Settings()
settings.batches = 50
settings.inactive = 10
settings.particles = 1000
settings.source = openmc.IndependentSource(
    space=openmc.stats.Point((0, 0, 0))
)
settings.export_to_xml()

if __name__ == "__main__":
    print("Model exported: materials.xml, geometry.xml, settings.xml")
    print("Run `openmc` in this directory to validate, or "
          "geometry.plot() interactively to check the layout.")
