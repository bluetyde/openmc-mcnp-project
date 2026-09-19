"""
Self-test for src/export_mcnp.py and src/validate_deck.py.

1. Builds three small OpenMC models (fixed-source shielding with tallies, photon box
   source, eigenvalue fuel sphere), exports each, and asserts the decks validate.
2. Breaks a valid deck one way at a time and asserts the validator catches each break
   with the expected message. A validator that passes everything proves nothing, so
   every check has to be seen failing.
3. Asserts unsupported features are refused instead of exported.
4. Exports rectangular lattices as LAT=1 / FILL cards and breaks them (surface order, FILL origin, FILL
   array order, a TRCL) to show the geometry check notices each mistake.
5. Does the same for hexagonal lattices (LAT=2), in both orientations, with face-order mistakes.
6. Exports tallies on cells inside lattice elements (CellInstanceFilter) as MCNP chains (c < L[i j k] < c0),
   rectangular and hexagonal, and breaks them (wrong element index, wrong cell, a missing bin).
7. Exports surface currents (SurfaceFilter x CellFromFilter, as OpenMC Studio writes them) as F1 + C + FS
   tallies, including a face cut by another part and a net current, refuses what MCNP can't match, and
   breaks the cards (FS sign, FS surface, direction, F1 surface, C card).
8. Exports several independent sources as one SDEF (ERG picks the source, the rest depends on it) for point +
   sphere, point + box and point + cylinder mixes, refuses two different volume shapes, and breaks the cards
   (positions swapped, strengths, energy order, radius order, particle).

Run from the project root, in the openmc-mcnp env:
    python tests/check_export_mcnp.py
Exit code 0 means every assertion held.
"""
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import openmc  # noqa: E402

from export_mcnp import export  # noqa: E402
from mcnp_cards import UnsupportedFeature  # noqa: E402
from remediate_deck import load_model  # noqa: E402
from validate_deck import validate_deck  # noqa: E402

SAMPLES = 5000
failures = []


def check(cond, what):
    print(("  PASS " if cond else "  FAIL ") + what)
    if not cond:
        failures.append(what)


def shielding_model(photon=False, eigen=False, absorption_in_fuel=False):
    openmc.reset_auto_ids()
    poly = openmc.Material(name="Polyethylene")
    poly.set_density("g/cm3", 0.93)
    poly.add_element("H", 0.143716, "wo")
    poly.add_element("C", 0.856284, "wo")
    poly.add_s_alpha_beta("c_H_in_CH2")
    lead = openmc.Material(name="Lead")
    lead.set_density("g/cm3", 11.35)
    lead.add_element("Pb", 1.0)
    fuel = openmc.Material(name="UO2")
    fuel.set_density("g/cm3", 10.4)
    fuel.add_element("U", 1.0, enrichment=5.0)
    fuel.add_element("O", 2.0)
    mats = [fuel, poly, lead] if (eigen or absorption_in_fuel) else [poly, lead]

    R = 60.0
    planes = [openmc.XPlane(-R, boundary_type="vacuum"), openmc.XPlane(R, boundary_type="vacuum"),
              openmc.YPlane(-R, boundary_type="vacuum"), openmc.YPlane(R, boundary_type="vacuum"),
              openmc.ZPlane(-R, boundary_type="vacuum"), openmc.ZPlane(R, boundary_type="vacuum")]
    world = +planes[0] & -planes[1] & +planes[2] & -planes[3] & +planes[4] & -planes[5]
    s_in, s_out = openmc.Sphere(r=10.0), openmc.Sphere(r=25.0)
    det_cyl, det_lo, det_hi = openmc.ZCylinder(x0=40.0, r=3.0), openmc.ZPlane(-8.0), openmc.ZPlane(8.0)
    detector = -det_cyl & +det_lo & -det_hi
    inner = openmc.Cell(name="Inner", fill=fuel if (eigen or absorption_in_fuel) else None, region=-s_in & world)
    shell = openmc.Cell(name="Shell", fill=poly, region=+s_in & -s_out & world)
    det = openmc.Cell(name="Detector", fill=lead, region=detector & world)
    outside = openmc.Cell(name="World", region=world & +s_out & ~detector)
    geometry = openmc.Geometry([inner, shell, det, outside])

    settings = openmc.Settings()
    settings.particles, settings.batches, settings.seed = 1000, 5, 1
    if eigen:
        settings.run_mode, settings.inactive = "eigenvalue", 2
        settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
    else:
        settings.run_mode = "fixed source"
        if photon:
            settings.photon_transport = True
            settings.source = openmc.IndependentSource(
                space=openmc.stats.Box((-2, -2, -2), (2, 2, 2)), particle="photon",
                energy=openmc.stats.Uniform(1e5, 2e6))
        else:
            settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)),
                                                       energy=openmc.stats.Discrete([14.1e6], [1.0]))
    t1 = openmc.Tally(name="Detector")
    t1.filters = [openmc.CellFilter([det]), openmc.EnergyFilter([0.0, 0.625, 1e5, 2e7])]
    t1.scores = ["flux", "(n,gamma)"]
    mesh = openmc.RegularMesh()
    mesh.dimension, mesh.lower_left, mesh.upper_right = [20, 1, 20], [-R, -1, -R], [R, 1, R]
    t2 = openmc.Tally(name="Map")
    t2.filters = [openmc.MeshFilter(mesh)]
    t2.scores = ["flux"]
    tallies = [t1, t2]
    if absorption_in_fuel:
        t3 = openmc.Tally(name="Fuel absorption")
        t3.filters = [openmc.CellFilter([inner])]
        t3.scores = ["absorption"]
        tallies.append(t3)
    return openmc.Model(geometry, openmc.Materials(mats), settings, openmc.Tallies(tallies))


def rotated_model():
    """A box rotated 30 degrees about z (general planes) and a cylinder tilted off-axis (GQ quadric)."""
    import numpy as np
    openmc.reset_auto_ids()
    lead = openmc.Material(name="Lead")
    lead.set_density("g/cm3", 11.35)
    lead.add_element("Pb", 1.0)
    water = openmc.Material(name="Water")
    water.set_density("g/cm3", 1.0)
    water.add_element("H", 2.0)
    water.add_element("O", 1.0)
    water.add_s_alpha_beta("c_H_in_H2O")
    R = 50.0
    planes = [openmc.XPlane(-R, boundary_type="vacuum"), openmc.XPlane(R, boundary_type="vacuum"),
              openmc.YPlane(-R, boundary_type="vacuum"), openmc.YPlane(R, boundary_type="vacuum"),
              openmc.ZPlane(-R, boundary_type="vacuum"), openmc.ZPlane(R, boundary_type="vacuum")]
    world = +planes[0] & -planes[1] & +planes[2] & -planes[3] & +planes[4] & -planes[5]
    a = np.radians(30)
    M = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    c, h = np.array([-15.0, 0.0, 0.0]), np.array([10.0, 4.0, 6.0])
    box = None
    for k in range(3):
        n = M[:, k]
        term = +openmc.Plane(a=n[0], b=n[1], c=n[2], d=float(n @ c - h[k])) & -openmc.Plane(a=n[0], b=n[1], c=n[2], d=float(n @ c + h[k]))
        box = term if box is None else box & term
    u = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    q, r, hh = np.array([20.0, 5.0, 0.0]), 5.0, 12.0
    Pm = np.eye(3) - np.outer(u, u)
    lin = -2 * Pm @ q
    quad = openmc.Quadric(a=Pm[0, 0], b=Pm[1, 1], c=Pm[2, 2], d=2 * Pm[0, 1], e=2 * Pm[1, 2], f=2 * Pm[0, 2],
                          g=lin[0], h=lin[1], j=lin[2], k=float(q @ Pm @ q - r * r))
    cyl = -quad & +openmc.Plane(a=u[0], b=u[1], c=u[2], d=float(u @ q - hh)) & -openmc.Plane(a=u[0], b=u[1], c=u[2], d=float(u @ q + hh))
    cells = [openmc.Cell(name="Box", fill=lead, region=box & world),
             openmc.Cell(name="Cylinder", fill=water, region=cyl & world & ~box),
             openmc.Cell(name="World", region=world & ~box & ~cyl)]
    settings = openmc.Settings(run_mode="fixed source", particles=1000, batches=5, seed=1)
    settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
    return openmc.Model(openmc.Geometry(cells), openmc.Materials([lead, water]), settings)


def _lattice_materials():
    steel = openmc.Material(name="Steel")
    steel.set_density("g/cm3", 7.9)
    steel.add_element("Fe", 1.0)
    water = openmc.Material(name="Water")
    water.set_density("g/cm3", 1.0)
    water.add_element("H", 2.0)
    water.add_element("O", 1.0)
    water.add_s_alpha_beta("c_H_in_H2O")
    return steel, water


def _world(R):
    planes = [openmc.XPlane(-R, boundary_type="vacuum"), openmc.XPlane(R, boundary_type="vacuum"),
              openmc.YPlane(-R, boundary_type="vacuum"), openmc.YPlane(R, boundary_type="vacuum"),
              openmc.ZPlane(-R, boundary_type="vacuum"), openmc.ZPlane(R, boundary_type="vacuum")]
    return +planes[0] & -planes[1] & +planes[2] & -planes[3] & +planes[4] & -planes[5]


def _box(lo, hi):
    return (+openmc.XPlane(lo[0]) & -openmc.XPlane(hi[0]) & +openmc.YPlane(lo[1]) & -openmc.YPlane(hi[1])
            & +openmc.ZPlane(lo[2]) & -openmc.ZPlane(hi[2]))


def lattice_3d_model():
    """A 3 x 2 x 2 RectLattice of steel rods in water: pitch differs per axis, lower_left off the origin, and
    element [2, 0, 1] (x index 2, y row 0 = bottom, z layer 1) holds a region-less water universe."""
    openmc.reset_auto_ids()
    steel, water = _lattice_materials()
    rod = openmc.ZCylinder(r=1.2)
    top, bottom = openmc.ZPlane(2.5), openmc.ZPlane(-2.5)
    pin = openmc.Universe(cells=[openmc.Cell(name="Rod", fill=steel, region=-rod & +bottom & -top),
                                 openmc.Cell(name="Rod water", fill=water, region=+rod | -bottom | +top)])
    solid = openmc.Universe(cells=[openmc.Cell(name="Empty site", fill=water)])  # no region: fills everything
    lat = openmc.RectLattice(name="Rods")
    lat.pitch, lat.lower_left = (4.0, 5.0, 6.0), (-7.0, -3.0, 2.0)
    grid = [[[pin] * 3 for _ in range(2)] for _ in range(2)]  # [z][y, top row first][x]
    grid[1][1][2] = solid  # z layer 1, bottom row (y index 0), x index 2
    lat.universes = grid
    world = _world(20.0)
    box = _box((-7.0, -3.0, 2.0), (5.0, 7.0, 14.0))
    cells = [openmc.Cell(name="Lattice", fill=lat, region=box & world),
             openmc.Cell(name="Outside", region=world & ~box)]
    settings = openmc.Settings(run_mode="fixed source", particles=1000, batches=5, seed=1)
    settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
    return openmc.Model(openmc.Geometry(cells), openmc.Materials([steel, water]), settings)


def lattice_2d_outer_model():
    """A 2D RectLattice (infinite in z) of 2 x 3 rods with an outer universe, inside a cell larger than the
    lattice, so the MCNP index ranges must extend past the array and use the outer universe there."""
    openmc.reset_auto_ids()
    steel, water = _lattice_materials()
    rod = openmc.ZCylinder(r=1.0)
    pin = openmc.Universe(cells=[openmc.Cell(name="Rod", fill=steel, region=-rod),
                                 openmc.Cell(name="Rod water", fill=water, region=+rod)])
    outer = openmc.Universe(cells=[openmc.Cell(name="Outer water", fill=water)])
    lat = openmc.RectLattice(name="Plane rods")
    lat.pitch, lat.lower_left = (3.0, 3.0), (-3.0, -4.5)
    lat.universes = [[pin, pin], [pin, outer], [pin, pin]]  # rows top first; middle row's right site is empty
    lat.outer = outer
    world = _world(15.0)
    box = _box((-7.5, -8.0, -5.0), (6.0, 9.5, 5.0))
    cells = [openmc.Cell(name="Lattice", fill=lat, region=box & world),
             openmc.Cell(name="Outside", region=world & ~box)]
    settings = openmc.Settings(run_mode="fixed source", particles=1000, batches=5, seed=1)
    settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
    return openmc.Model(openmc.Geometry(cells), openmc.Materials([steel, water]), settings)


def hex_lattice_model(orientation="y", two_levels=True):
    """A 3-ring HexLattice of rods with an asymmetric pattern (steel and water rods, one site different in
    each ring), two axial levels that differ, and an outer universe, inside a larger cylinder: every
    index-direction or ordering mistake in the MCNP cards changes which rod is where."""
    openmc.reset_auto_ids()
    steel, water = _lattice_materials()
    rod = openmc.ZCylinder(r=0.9)
    pin = openmc.Universe(cells=[openmc.Cell(name="Steel rod", fill=steel, region=-rod),
                                 openmc.Cell(name="Rod water", fill=water, region=+rod)])
    thin = openmc.ZCylinder(r=0.4)
    wire = openmc.Universe(cells=[openmc.Cell(name="Steel wire", fill=steel, region=-thin),
                                  openmc.Cell(name="Wire water", fill=water, region=+thin)])
    outer = openmc.Universe(cells=[openmc.Cell(name="Outer water", fill=water)])
    lat = openmc.HexLattice(name="Hex rods")
    lat.orientation = orientation
    level_a = [[pin] * 3 + [wire] + [pin] * 8, [wire] + [pin] * 5, [pin]]    # outer ring first
    level_b = [[pin] * 7 + [wire] * 2 + [pin] * 3, [pin] * 4 + [wire] + [pin], [wire]]
    if two_levels:
        lat.pitch, lat.center, lat.universes = (2.5, 6.0), (1.0, -2.0, 4.0), [level_a, level_b]
    else:
        lat.pitch, lat.center, lat.universes = (2.5,), (1.0, -2.0), level_a
    lat.outer = outer
    world = _world(20.0)
    can = openmc.ZCylinder(x0=1.0, y0=-2.0, r=6.5)
    lo, hi = openmc.ZPlane(-2.0), openmc.ZPlane(10.0)
    region = -can & +lo & -hi
    cells = [openmc.Cell(name="Lattice", fill=lat, region=region & world),
             openmc.Cell(name="Outside", region=world & ~region)]
    settings = openmc.Settings(run_mode="fixed source", particles=1000, batches=5, seed=1)
    settings.source = openmc.IndependentSource(space=openmc.stats.Point((1.0, -2.0, 4.0)))
    return openmc.Model(openmc.Geometry(cells), openmc.Materials([steel, water]), settings)


def lattice_tally_model(kind):
    """lattice_3d_model() or hex_lattice_model() with a flux + absorption tally on the steel rod in three lattice
    elements (a CellInstanceFilter), picked from the first, middle and last of the rod's instances."""
    model = lattice_3d_model() if kind == "rect" else hex_lattice_model("y", True)
    g = model.geometry
    rod = next(c for c in g.get_all_cells().values() if c.name in ("Rod", "Steel rod"))
    g.determine_paths()
    n = rod.num_instances
    t = openmc.Tally(name=f"{kind} rods")
    t.filters = [openmc.CellInstanceFilter([(rod, 0), (rod, n // 2), (rod, n - 1)])]
    t.scores = ["flux", "absorption"]
    model.tallies = openmc.Tallies([t])
    return model


def current_model():
    """Parts as OpenMC Studio writes them: box E (listed first) cuts into the +x face of box A; sphere B stands
    apart; World is the rest. Tally 'current out' is the current leaving A and B through their own surfaces
    (every surface x every part, with energy bins); tally 'net B' is the net current through B's sphere."""
    openmc.reset_auto_ids()
    steel, water = _lattice_materials()
    world = _world(20.0)
    E, A = _box((4.0, -2.0, -2.0), (8.0, 2.0, 2.0)), _box((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
    sphere = openmc.Sphere(x0=12.0, y0=10.0, z0=0.0, r=3.0)
    cells = [openmc.Cell(name="E", fill=steel, region=E & world),
             openmc.Cell(name="A", fill=water, region=A & world & ~E),
             openmc.Cell(name="B", fill=steel, region=-sphere & world)]
    cells.append(openmc.Cell(name="World", region=world & ~(E | A | -sphere)))
    out = openmc.Tally(name="current out")
    out.filters = [openmc.SurfaceFilter([h.surface for h in A] + [sphere]),
                   openmc.CellFromFilter([cells[1], cells[2]]), openmc.EnergyFilter([1e-5, 1e5, 2e7])]
    out.scores = ["current"]
    net = openmc.Tally(name="net B")
    net.filters = [openmc.SurfaceFilter([sphere])]
    net.scores = ["current"]
    settings = openmc.Settings(run_mode="fixed source", particles=1000, batches=5, seed=1)
    settings.source = openmc.IndependentSource(space=openmc.stats.Point((0.0, 0.0, 0.0)))
    model = openmc.Model(openmc.Geometry(cells), openmc.Materials([steel, water]), settings)
    model.tallies = openmc.Tallies([out, net])
    return model


def multi_source_model(kind):
    """A water box with a flux tally and two or three sources of different strength, energy, particle and
    direction. kind: 'sphere' (monodirectional point neutron + isotropic photon sphere shell + point), 'box'
    (box + point) or 'cylinder' (cylinder + point), or 'mixed' (box + sphere, which MCNP can't share)."""
    openmc.reset_auto_ids()
    steel, water = _lattice_materials()
    world = _world(20.0)
    cell = openmc.Cell(name="Water", fill=water, region=world)
    st = openmc.stats
    point = st.Point((1.0, 2.0, 3.0))
    point2 = st.Point((-4.0, 0.0, 1.0))
    sphere = st.spherical_uniform(r_outer=3.0, r_inner=1.0, origin=(5.0, 0.0, 0.0))
    box = st.Box((-6.0, -5.0, -4.0), (-2.0, 5.0, 4.0))
    cyl = st.CylindricalIndependent(r=st.PowerLaw(0.0, 2.0, 1), phi=st.Uniform(0.0, 2 * 3.141592653589793),
                                    z=st.Uniform(-1.5, 2.5), origin=(0.0, 6.0, -2.0))
    S = openmc.IndependentSource
    if kind == "sphere":
        sources = [S(space=point, energy=st.Discrete([2e6], [1.0]), angle=st.Monodirectional((0.0, 0.0, 1.0)), strength=1.0),
                   S(space=sphere, energy=st.Discrete([0.5e6, 1.2e6], [0.3, 0.7]), particle="photon", strength=3.0),
                   S(space=point2, energy=st.Watt(), strength=0.5)]
    elif kind == "box":
        sources = [S(space=box, energy=st.Uniform(1e6, 2e6), strength=2.0), S(space=point, energy=st.Maxwell(1.3e6), strength=1.0)]
    elif kind == "cylinder":
        sources = [S(space=point2, energy=st.Tabular([1e5, 1e6, 3e6], [0.2, 0.8], interpolation="histogram"), strength=1.0),
                   S(space=cyl, energy=st.Discrete([14.1e6], [1.0]), strength=4.0)]
    else:
        sources = [S(space=box, strength=1.0), S(space=sphere, strength=1.0)]
    settings = openmc.Settings(run_mode="fixed source", particles=1000, batches=5, seed=1)
    settings.source = sources
    model = openmc.Model(openmc.Geometry([cell]), openmc.Materials([water]), settings)
    t = openmc.Tally(name="flux")
    t.filters = [openmc.CellFilter([cell])]
    t.scores = ["flux"]
    model.tallies = openmc.Tallies([t])
    return model


def standalone_primitives_model():
    """Standalone box, standalone z-cylinder, and standalone sphere inside a world box."""
    openmc.reset_auto_ids()
    steel, water = _lattice_materials()
    box_reg = _box((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
    cyl = openmc.ZCylinder(x0=15.0, y0=0.0, r=4.0)
    top, bottom = openmc.ZPlane(8.0), openmc.ZPlane(-8.0)
    cyl_reg = -cyl & +bottom & -top
    sph = openmc.Sphere(x0=-15.0, y0=0.0, z0=0.0, r=4.0)
    sph_reg = -sph
    world = _world(30.0)

    cells = [
        openmc.Cell(name="Box", fill=steel, region=box_reg),
        openmc.Cell(name="Cyl", fill=water, region=cyl_reg),
        openmc.Cell(name="Sphere", fill=water, region=sph_reg),
        openmc.Cell(name="World", region=world & ~box_reg & ~cyl_reg & ~sph_reg),
    ]
    settings = openmc.Settings(run_mode="fixed source", particles=1000, batches=5, seed=1)
    settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
    return openmc.Model(openmc.Geometry(cells), openmc.Materials([steel, water]), settings)


def shared_face_box_model():
    """Two adjacent boxes sharing a plane face (x=0). Neither should be collapsed to RPP,
    preventing face mismatch / boundary corruption."""
    openmc.reset_auto_ids()
    steel, water = _lattice_materials()
    shared_px = openmc.XPlane(0.0)
    box_a = (+openmc.XPlane(-10.0) & -shared_px &
             +openmc.YPlane(-5.0) & -openmc.YPlane(5.0) &
             +openmc.ZPlane(-5.0) & -openmc.ZPlane(5.0))
    box_b = (+shared_px & -openmc.XPlane(10.0) &
             +openmc.YPlane(-5.0) & -openmc.YPlane(5.0) &
             +openmc.ZPlane(-5.0) & -openmc.ZPlane(5.0))
    world = _world(20.0)
    cells = [
        openmc.Cell(name="BoxA", fill=steel, region=box_a),
        openmc.Cell(name="BoxB", fill=water, region=box_b),
        openmc.Cell(name="World", region=world & ~box_a & ~box_b),
    ]
    settings = openmc.Settings(run_mode="fixed source", particles=1000, batches=5, seed=1)
    settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
    return openmc.Model(openmc.Geometry(cells), openmc.Materials([steel, water]), settings)


def export_model(model, work, name):
    d = os.path.join(work, name)
    os.makedirs(d)
    model.export_to_model_xml(os.path.join(d, "model.xml"))
    with contextlib.redirect_stdout(io.StringIO()):
        report = export(os.path.join(d, "model.xml"), d, name, samples=SAMPLES)
    return report


def _wrap_card(text, width=78):
    """One cell card over several lines (continuations start with 5 spaces), as MCNP input requires."""
    out, line = [], ""
    for tok in text.split():
        if line and len(line) + 1 + len(tok) > width:
            out.append(line)
            line = "     " + tok
        else:
            line = f"{line} {tok}" if line else tok
    return "\n".join(out + [line])


def validate_text(text, model, work):
    path = os.path.join(work, "mutant.mcnp")
    with open(path, "w") as f:
        f.write(text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = validate_deck(path, model=model, geometry_samples=SAMPLES)
    return ok, buf.getvalue()


def main():
    work = tempfile.mkdtemp(prefix="export_mcnp_selftest_")
    try:
        print("1. Valid exports")
        reports = {}
        for name, kw in [("fixed", {}), ("photon", {"photon": True}), ("eigen", {"eigen": True})]:
            r = export_model(shielding_model(**kw), work, name)
            reports[name] = r
            check(r["ok"], f"{name}: exported deck validates")
            check("geometry matches OpenMC" in r["validation"], f"{name}: geometry check ran and matched")

        r = export_model(rotated_model(), work, "rotated")
        check(r["ok"], "rotated box + cylinder (general planes, GQ): exported deck validates")
        check("geometry matches OpenMC" in r["validation"], "rotated: geometry check ran and matched")
        rot_text = open(r["runnable"]).read()
        check(re.search(r"^\d+ GQ ", rot_text, re.M) is not None, "rotated: cylinder written as a GQ card")
        rot_model = load_model(os.path.join(work, "rotated", "model.xml"))
        lines = rot_text.split("\n")
        gi = next(i for i, l in enumerate(lines) if re.match(r"^\d+ GQ ", l))
        gj = gi + 1
        while gj < len(lines) and lines[gj].startswith("     "):
            gj += 1
        head, last = lines[gj - 1].rstrip().rsplit(" ", 1)
        lines[gj - 1] = head + " " + repr(float(last) * 0.8)  # scale the GQ constant K
        broken = "\n".join(lines)
        ok, out = validate_text(broken, rot_model, work)
        check(broken != rot_text and not ok and "Geometry:" in out, "[rotated: GQ constant changed] rejected by the geometry check")

        photon_text = open(reports["photon"]["runnable"]).read()
        check("MODE N P" in photon_text and re.search(r"IMP:n,p=", photon_text, re.I) is not None,
              "photon: MODE N P and combined IMP:n,p importances written")
        eigen_text = open(reports["eigen"]["runnable"]).read()
        check("KCODE 1000 1.0 2 5" in eigen_text and "KSRC 0.0 0.0 0.0" in eigen_text and "SDEF" not in eigen_text,
              "eigen: KCODE/KSRC from settings, no SDEF")

        tab_m = shielding_model()
        tab_m.settings.source = [openmc.IndependentSource(
            space=openmc.stats.Point((0, 0, 0)),
            energy=openmc.stats.Tabular([0.0, 1e6, 2e6, 5e6], [0.2, 0.5, 0.3], interpolation="histogram")
        )]
        tab_rep = export_model(tab_m, work, "tabular")
        check(tab_rep["ok"], "tabular source: exported deck validates")
        tab_text = open(tab_rep["runnable"]).read()
        check("SI1 H 0.0 1.0 2.0 5.0" in tab_text and "SP1 D 0 0.2 0.5 0.3" in tab_text,
              "tabular source: SI H and SP D cards written correctly")

        print("2. Each validator check fires on a broken deck")
        base_path = reports["fixed"]["runnable"]
        base = open(base_path).read()
        model = load_model(os.path.join(work, "fixed", "model.xml"))
        graveyard = reports["fixed"]["graveyard_cell"]

        def mutate(desc, fn, expect):
            text = fn(base)
            check(text != base, f"[{desc}] mutation changed the deck")
            ok, out = validate_text(text, model, work)
            check(not ok and re.search(expect, out) is not None, f"[{desc}] rejected with /{expect}/")
            if ok or re.search(expect, out) is None:
                print("      validator said:\n" + "\n".join("      " + l for l in out.splitlines()))

        def drop_line(pattern):
            return lambda t: "\n".join(l for l in t.splitlines() if not re.match(pattern, l)) + "\n"

        def sub(pattern, repl, count=1):
            return lambda t: re.sub(pattern, repl, t, count=count, flags=re.M)

        mutate("graveyard removed", drop_line(rf"^{graveyard} 0"), r"no MCNP cell has IMP:N=0|in no MCNP cell")
        mutate("sphere radius changed", sub(r"^(\d+) SO 25\.0", r"\1 SO 26.0"), r"Geometry: point .* but MCNP cell")
        mutate("density changed", sub(r"^(\d+ \d+) -0\.93 ", r"\1 -0.95 "), r"density 0\.93 g/cm3 in OpenMC")
        mutate("complement removed (overlap)", sub(r"^(\d+ 0 [^\n]*?)\(-?\d+:-?\d+:-?\d+\)", r"\1"),
               r"several MCNP cells|but MCNP cell")
        mutate("MT card removed", drop_line(r"^MT\d+ "), r"no 'MT\d+' card")
        mutate("KCODE added to fixed source", lambda t: t + "KCODE 1000 1.0 2 5\n", r"fixed source but the deck has a KCODE")
        mutate("SDEF removed", lambda t: re.sub(r"^SDEF[^\n]*\n(     [^\n]*\n)*", "", t, flags=re.M), r"no SDEF card")
        mutate("NPS removed", drop_line(r"^NPS "), r"no NPS card")
        mutate("tally removed", drop_line(r"^F14:N "), r"Expected 3 tallies")
        mutate("tally cell renumbered", sub(r"^F4:N \d+", "F4:N 777"), r"refers to cells \['777'\]")
        mutate("universe 0 not assigned", lambda t: re.sub(r"(IMP:N=[0-9.]+)", r"U 1 \1", t), r"Universe 0|universe 1")

        pbase = photon_text
        pmodel = load_model(os.path.join(work, "photon", "model.xml"))
        text = pbase.replace("MODE N P", "MODE N")
        ok, out = validate_text(text, pmodel, work)
        check(text != pbase and not ok and "MODE has no P" in out, "[photon: P removed from MODE] rejected")


        print("4. Rectangular lattices as MCNP LAT=1 / FILL")
        lat_reports = {}
        for name, build in [("lattice3d", lattice_3d_model), ("lattice2d", lattice_2d_outer_model)]:
            r = export_model(build(), work, name)
            lat_reports[name] = r
            text = open(r["runnable"]).read()
            check(r["ok"], f"{name}: exported deck validates")
            check("geometry matches OpenMC" in r["validation"], f"{name}: geometry check followed the lattice and matched")
            check(re.search(r"\bLAT=1\b", text) is not None and "TRCL" not in text.upper(),
                  f"{name}: LAT=1 card written, no TRCL left from MCNPy")
            check(re.search(r"\bRPP\b", text) is not None,
                  f"{name}: RPP macrobody card written for base element")
        text2d = open(lat_reports["lattice2d"]["runnable"]).read()
        check(re.search(r"FILL=-2:2 -2:4 0:0 ", text2d) is not None,
              "lattice2d: FILL ranges -2:2 -2:4 reach past the 2 x 3 array to cover its cell (outer universe there)")

        base3 = open(lat_reports["lattice3d"]["runnable"]).read()
        model3 = load_model(os.path.join(work, "lattice3d", "model.xml"))
        lines3 = base3.split("\n")
        li = next(i for i, l in enumerate(lines3) if re.search(r"\bLAT=1\b", l))
        lj = li + 1
        while lj < len(lines3) and lines3[lj].startswith("     "):
            lj += 1
        card = " ".join(l.strip() for l in lines3[li:lj])

        def with_card(new_card):
            return "\n".join(lines3[:li] + [_wrap_card(new_card)] + lines3[lj:])

        def lat_mutate(desc, text, expect):
            ok, out = validate_text(text, model3, work)
            check(text != base3 and not ok and re.search(expect, out) is not None, f"[{desc}] rejected with /{expect}/")
            if ok or re.search(expect, out) is None:
                print("      validator said:\n" + "\n".join("      " + l for l in out.splitlines()[-8:]))

        # 1. RPP element bounds shifted by 1 cm in x
        sm = re.search(r"^(\s*\d+\s+RPP\s+)(-?[0-9.]+)\s+(-?[0-9.]+)(\s+.*)", base3, re.MULTILINE)
        x0, x1 = float(sm.group(2)), float(sm.group(3))
        shifted_rpp = base3[:sm.start()] + f"{sm.group(1)}{x0 + 1.0} {x1 + 1.0}{sm.group(4)}" + base3[sm.end():]
        lat_mutate("lattice: RPP element shifted in x", shifted_rpp, r"Geometry:")
        # 1b. RPP element has positive sense (+s instead of -s)
        pm = re.match(r"^(\d+ 0 )-(\d+)( .*\bLAT=1\b.*)", card)
        lat_mutate("lattice: RPP element has positive sense", with_card(f"{pm.group(1)}{pm.group(2)}{pm.group(3)}"),
                   r"RPP element must have negative sense")
        # 2. the filled cell's FILL origin moved by 1 cm in x
        mo = re.search(r"FILL=(\d+) \((\S+) (\S+) (\S+)\)", base3)
        moved = base3.replace(mo.group(0), f"FILL={mo.group(1)} ({float(mo.group(2)) + 1.0} {mo.group(3)} {mo.group(4)})", 1)
        lat_mutate("lattice: FILL origin shifted 1 cm", moved, r"Geometry:")
        # 3. the FILL array written j-fastest instead of i-fastest
        fm = re.search(r"FILL=0:2 0:1 0:1 ((?:\d+ ){12})", card + " ")
        vals = fm.group(1).split()
        transposed = [vals[k * 6 + j * 3 + i] for k in range(2) for i in range(3) for j in range(2)]  # j fastest
        lat_mutate("lattice: FILL array transposed", with_card(card.replace(fm.group(1), " ".join(transposed) + " ", 1)),
                   r"Geometry:")
        # 4. a TRCL put back on a unit-universe cell: the check must say it can't check, not pass
        ui = next(i for i, l in enumerate(lines3) if re.match(r"^\d+ \d+ -?[0-9.]+ .*\bU \d+", l))
        with_trcl = lines3[:]
        with_trcl[ui] = re.sub(r"(\bU \d+)", r"\1 TRCL (0 0 1)", with_trcl[ui], count=1)
        lat_mutate("lattice: TRCL on a unit cell", "\n".join(with_trcl), r"TRCL")


        print("5. Hexagonal lattices as MCNP LAT=2 / FILL")
        hex_reports = {}
        for name, kw in [("hexy2", {"orientation": "y", "two_levels": True}),
                         ("hexx1", {"orientation": "x", "two_levels": False})]:
            r = export_model(hex_lattice_model(**kw), work, name)
            hex_reports[name] = r
            text = open(r["runnable"]).read()
            check(r["ok"], f"{name}: exported deck validates")
            check("geometry matches OpenMC" in r["validation"], f"{name}: geometry check followed the LAT=2 lattice and matched")
            check(re.search(r"\bLAT=2\b", text) is not None and "TRCL" not in text.upper(),
                  f"{name}: LAT=2 card written, no TRCL left from MCNPy")

        baseh = open(hex_reports["hexy2"]["runnable"]).read()
        modelh = load_model(os.path.join(work, "hexy2", "model.xml"))
        linesh = baseh.split("\n")
        hi_ = next(i for i, l in enumerate(linesh) if re.search(r"\bLAT=2\b", l))
        hj = hi_ + 1
        while hj < len(linesh) and linesh[hj].startswith("     "):
            hj += 1
        hcard = " ".join(l.strip() for l in linesh[hi_:hj])
        hm = re.match(r"^(\d+ 0 )((?:-?\d+ ){8})", hcard)
        hsurf = hm.group(2).split()

        def hex_mutate(desc, surf_order, expect):
            new = hcard.replace(hm.group(0), hm.group(1) + " ".join(surf_order) + " ", 1)
            text = "\n".join(linesh[:hi_] + [_wrap_card(new)] + linesh[hj:])
            ok, out = validate_text(text, modelh, work)
            check(text != baseh and not ok and re.search(expect, out) is not None, f"[{desc}] rejected with /{expect}/")
            if ok or re.search(expect, out) is None:
                print("      validator said:\n" + "\n".join("      " + l for l in out.splitlines()[-6:]))

        a = hsurf
        # swapping one pair also breaks the [-1,1,0] rule, so either reason counts as caught
        hex_mutate("hex: [0,1,0] and [0,-1,0] faces swapped", [a[0], a[1], a[3], a[2]] + a[4:], r"Geometry:|LAT=2 order")
        hex_mutate("hex: i and j face pairs swapped", [a[2], a[3], a[0], a[1]] + a[4:], r"Geometry:|LAT=2 order")
        hex_mutate("hex: axial faces swapped", a[:6] + [a[7], a[6]], r"Geometry:")
        hex_mutate("hex: 5th face not [-1,1,0]", a[:4] + [a[5], a[4]] + a[6:], r"LAT=2 order")
        mo = re.search(r"FILL=(\d+) \((\S+) (\S+) (\S+)\)", baseh)
        moved = baseh.replace(mo.group(0), f"FILL={mo.group(1)} ({mo.group(2)} {float(mo.group(3)) + 0.7} {mo.group(4)})", 1)
        ok, out = validate_text(moved, modelh, work)
        check(not ok and "Geometry:" in out, "[hex: FILL origin shifted 0.7 cm in y] rejected by the geometry check")

        print("6. Tallies on cells inside lattice elements as MCNP chains")
        tal = {}
        for kind in ("rect", "hex"):
            r = export_model(lattice_tally_model(kind), work, f"tally_{kind}")
            text = open(r["runnable"]).read()
            tal[kind] = (text, load_model(os.path.join(work, f"tally_{kind}", "model.xml")))
            f4 = re.search(r"^F4:N (.*)$", text, re.M)
            check(r["ok"], f"{kind}: deck with lattice tally chains validates")
            check(f4 is not None and len(re.findall(r"\(\d+ < \d+\[-?\d+ -?\d+ -?\d+\] < \d+\)", f4.group(1))) == 3,
                  f"{kind}: F4 has three chains (rod < LAT cell[i j k] < filled cell)")
            check(re.search(r"^SD4 1 1 1$", text, re.M) is not None and re.search(r"^FM14 \(-1 \d+ -2\)", text, re.M),
                  f"{kind}: SD with one 1 per bin, absorption FM from the rod's material")
            check("3 of 3 lattice tally bins hold the same points" in r["validation"],
                  f"{kind}: every chain holds the same points as its OpenMC cell instance")

        def tally_mutate(kind, desc, change, expect):
            base, model = tal[kind]
            text = re.sub(r"^F4:N .*$", lambda m: change(m.group(0)), base, count=1, flags=re.M)
            ok, out = validate_text(text, model, work)
            check(text != base and not ok and re.search(expect, out) is not None, f"[{kind}: {desc}] rejected with /{expect}/")
            if ok or re.search(expect, out) is None:
                print("      validator said:\n" + "\n".join("      " + l for l in out.splitlines()[-6:]))

        def shift_first_index(card):  # element [i j k] -> [i+1 j k] in the first chain: a neighbour, often the same universe
            m = re.search(r"\[(-?\d+) (-?\d+) (-?\d+)\]", card)
            return card[:m.start()] + f"[{int(m.group(1)) + 1} {m.group(2)} {m.group(3)}]" + card[m.end():]

        for kind in ("rect", "hex"):
            tally_mutate(kind, "first bin's element index moved by one", shift_first_index, r"bin for cell \d+ instance \d+: point")
        rod_water = next(c.id for c in tal["rect"][1].geometry.get_all_cells().values() if c.name == "Rod water")
        tally_mutate("rect", "first bin ends in the rod's water, not the rod",
                     lambda card: re.sub(r"\((\d+) <", f"({rod_water} <", card, count=1), r"ends in MCNP cell")
        tally_mutate("rect", "last bin dropped", lambda card: card[:card.rindex("(")].rstrip(), r"has 2 bins but")
        tally_mutate("rect", "lattice index on a cell that isn't a lattice",
                     lambda card: re.sub(r"< (\d+)\)", r"< \1[0 0 0])", card, count=1), r"aren't lattice cells")

        print("7. Surface currents as MCNP F1 + C + FS")
        r = export_model(current_model(), work, "current")
        ctext = open(r["runnable"]).read()
        cmodel = load_model(os.path.join(work, "current", "model.xml"))
        check(r["ok"], "current: deck with F1/C/FS current tallies validates")
        check("current tallies: F1 surface, FS face and direction match OpenMC" in r["validation"],
              "current: the face check compared every F1 with OpenMC")
        f1s = re.findall(r"^F(\d*1):N (\d+)$", ctext, re.M)
        check(len(f1s) == 8, f"current: 8 F1 tallies (A's 6 faces, B's sphere, the net current); found {len(f1s)}")
        check(len(re.findall(r"^C\d*1 0 1$", ctext, re.M)) == 8 and len(re.findall(r"^E\d*1 ", ctext, re.M)) == 7,
              "current: a C 0 1 card on each, E cards on the 7 with energy bins")
        cut = re.search(r"^FC(\d+) .*\[S (\d+) C (\d+) SEG (\d+)-(\d+) COS (\d) X([+-]1)\]$", ctext, re.M)
        check(cut is not None, "current: A's +x face (cut by E) is the sum of several FS segments")
        check(re.search(r"\[S \d+ NET\]", ctext) is not None, "current: net current tagged [S s NET]")
        check(any("always 0 in OpenMC" in n for n in r.get("notes", [])), "current: bins that can't score are named in a note")

        def cur_mutate(desc, change, expect):
            text = change(ctext)
            ok, out = validate_text(text, cmodel, work)
            check(text != ctext and not ok and re.search(expect, out) is not None, f"[current: {desc}] rejected with /{expect}/")
            if ok or re.search(expect, out) is None:
                print("      validator said:\n" + "\n".join("      " + l for l in out.splitlines()[-6:]))

        n_cut = cut.group(1)
        plain = re.search(r"^FC(\d+) .*\[S (\d+) C \d+ SEG \d+ COS (\d) X([+-]1)\]$", ctext, re.M)
        n_pl = plain.group(1)

        def edit_card(text, prefix, fn):
            return re.sub(rf"^{prefix} (.*)$", lambda m: f"{prefix} " + fn(m.group(1)), text, count=1, flags=re.M)

        def flip_first(words):
            w = words.split()
            w[0] = w[0][1:] if w[0].startswith("-") else "-" + w[0]
            return " ".join(w)
        cur_mutate("first FS sign flipped on the cut face", lambda t: edit_card(t, f"FS{n_cut}", flip_first), r"FS card puts")
        cur_mutate("an FS surface dropped", lambda t: edit_card(t, f"FS{n_pl}", lambda w: " ".join(w.split()[:1] + w.split()[2:])),
                   r"FS card puts")
        swapped = "COS 1 X-1" if plain.group(3) == "2" else "COS 2 X+1"
        cur_mutate("direction swapped in the FC tag", lambda t: re.sub(rf"^(FC{n_pl} .*)COS \d X[+-]1", rf"\g<1>{swapped}", t, count=1, flags=re.M),
                   r"so leaving it is cosine bin")
        other = next(s for f, s in f1s if s != plain.group(2))
        cur_mutate("F1 on another surface", lambda t: re.sub(rf"^F{n_pl}:N \d+$", f"F{n_pl}:N {other}", t, count=1, flags=re.M),
                   r"isn't OpenMC surface")
        cur_mutate("C card removed", lambda t: re.sub(rf"^C{n_pl} 0 1\n", "", t, count=1, flags=re.M), r"needs the cosine card")

        m = current_model()
        cells = {c.name: c for c in m.geometry.get_all_cells().values()}
        e_plane = next(iter(cells["E"].region.get_surfaces().values()))
        bad = openmc.Tally(name="E plane from A")
        bad.filters = [openmc.SurfaceFilter([e_plane]), openmc.CellFromFilter([cells["A"]])]
        bad.scores = ["current"]
        m.tallies = openmc.Tallies([bad])
        try:
            export_model(m, work, "cur_both")
            check(False, "current: a surface of a part carved out of the cell refused")
        except UnsupportedFeature as e:
            check("both sides" in str(e), "current: a surface of a part carved out of the cell refused (cell on both sides)")
        m = current_model()
        world_plane = next(s for s in m.geometry.get_all_surfaces().values() if s.boundary_type == "vacuum")
        world_plane.boundary_type = "reflective"
        refl = openmc.Tally(name="reflective")
        refl.filters = [openmc.SurfaceFilter([world_plane]),
                        openmc.CellFromFilter([next(c for c in m.geometry.get_all_cells().values() if c.name == "World")])]
        refl.scores = ["current"]
        m.tallies = openmc.Tallies([refl])
        try:
            export_model(m, work, "cur_refl")
            check(False, "current: a reflective surface refused")
        except UnsupportedFeature as e:
            check("reflective" in str(e), "current: a reflective surface refused")
        m = current_model()
        m.tallies[0].scores = ["current", "flux"]
        try:
            export_model(m, work, "cur_flux")
            check(False, "current: current mixed with another score refused")
        except UnsupportedFeature as e:
            check("only the score 'current'" in str(e), "current: current mixed with another score refused")

        print("8. Several sources in one SDEF")
        src_texts = {}
        for kind in ("sphere", "box", "cylinder"):
            r = export_model(multi_source_model(kind), work, f"src_{kind}")
            text = open(r["runnable"]).read()
            src_texts[kind] = (text, load_model(os.path.join(work, f"src_{kind}", "model.xml")))
            check(r["ok"], f"sources ({kind}): deck validates")
            check("sources read back from the SDEF match OpenMC's" in r["validation"], f"sources ({kind}): every source read back and matched")
            check(re.search(r"^SI(\d+) S ", text, re.M) is not None and re.search(r"ERG=D\d+", text) is not None,
                  f"sources ({kind}): ERG picks the source (SI S)")
        stext = src_texts["sphere"][0]
        check("PAR=FERG=D" in stext and "MODE N P" in stext, "sources (sphere): particle depends on the source; MODE N P for the photon source")
        check("VEC=FERG=D" in stext and "DIR=FERG=D" in stext, "sources (sphere): direction depends on the source (one monodirectional)")
        check(re.search(r"X=FERG=D", src_texts["box"][0]) is not None, "sources (box): X/Y/Z depend on the source")
        check("AXS=0.0 0.0 1.0" in src_texts["cylinder"][0] and "EXT=FERG=D" in src_texts["cylinder"][0],
              "sources (cylinder): AXS fixed, RAD/EXT depend on the source")
        try:
            export_model(multi_source_model("mixed"), work, "src_mixed")
            check(False, "sources: a box and a sphere source refused")
        except UnsupportedFeature as e:
            check("box and sphere" in str(e), "sources: a box and a sphere source refused (one SDEF shape)")

        smodel = src_texts["sphere"][1]

        def src_mutate(desc, change, expect):
            text = change(stext)
            ok, out = validate_text(text, smodel, work)
            check(text != stext and not ok and re.search(expect, out) is not None, f"[sources: {desc}] rejected with /{expect}/")
            if ok or re.search(expect, out) is None:
                print("      validator said:\n" + "\n".join("      " + l for l in out.splitlines()[-6:]))

        def ds_of(key):
            return re.search(rf"{key}=FERG=D(\d+)", stext).group(1)

        def swap_groups(card, size, a=0, b=1):
            def f(m):
                w = m.group(2).split()
                g = [w[i:i + size] for i in range(0, len(w), size)]
                g[a], g[b] = g[b], g[a]
                return m.group(1) + " ".join(x for grp in g for x in grp)
            return lambda t: re.sub(rf"^({card} [LS] )(.*)$", f, t, count=1, flags=re.M)
        sel = re.search(r"(?<![A-Z])ERG=D(\d+)", stext).group(1)  # not the ERG in PAR=FERG=Dn
        src_mutate("first two positions swapped", swap_groups(f"DS{ds_of('POS')}", 3), r"Source \d: .*POS gives")
        src_mutate("strengths changed", lambda t: re.sub(rf"^SP{sel} .*$", f"SP{sel} 1 1 1", t, count=1, flags=re.M), rf"SP{sel} .* doesn't match")
        src_mutate("energies in the wrong order", lambda t: re.sub(rf"^(SI{sel} S )(\d+) (\d+)", r"\g<1>\3 \2", t, count=1, flags=re.M),
                   r"Source \d: energy")
        src_mutate("radius distributions swapped", swap_groups(f"DS{ds_of('RAD')}", 1), r"Source \d: (radius|a point source needs RAD)")
        src_mutate("particle list changed", lambda t: re.sub(rf"^DS{ds_of('PAR')} L .*$", f"DS{ds_of('PAR')} L 1 1 1", t, count=1, flags=re.M),
                   r"Source 2: MCNP particle")

        print("9. Standalone macrobody simplification (RPP, RCC)")
        r_prim = export_model(standalone_primitives_model(), work, "primitives")
        check(r_prim["ok"], "primitives: deck validates")
        prim_text = open(r_prim["runnable"]).read()
        check(re.search(r"^\s*\d+\s+RPP\s+-5\.?0*\s+5\.?0*\s+-5\.?0*\s+5\.?0*\s+-5\.?0*\s+5\.?0*", prim_text, re.M) is not None,
              "primitives: box simplified to RPP card with correct bounds")
        check(re.search(r"^\s*\d+\s+RCC\s+15\.?0*\s+0\.?0*\s+-8\.?0*\s+0\.?0*\s+0\.?0*\s+16\.?0*\s+4\.?0*", prim_text, re.M) is not None,
              "primitives: cylinder simplified to RCC card with correct base and height")
        check("Cell 1: simplified 6 primitive surfaces into RPP" in " ".join(r_prim["notes"]),
              "primitives: remediation note records box RPP simplification")
        check("Cell 2: simplified 3 primitive surfaces into RCC" in " ".join(r_prim["notes"]),
              "primitives: remediation note records cylinder RCC simplification")
        check("geometry matches OpenMC" in r_prim["validation"], "primitives: geometry check matched OpenMC with 0 lost particles")

        # Mutations for RPP and RCC
        prim_model = load_model(os.path.join(work, "primitives", "model.xml"))

        def prim_mutate(desc, change, expect):
            text = change(prim_text)
            ok, out = validate_text(text, prim_model, work)
            check(text != prim_text and not ok and re.search(expect, out) is not None, f"[{desc}] rejected with /{expect}/")
            if ok or re.search(expect, out) is None:
                print("      validator said:\n" + "\n".join("      " + l for l in out.splitlines()[-6:]))

        # 1. RPP bound perturbed
        prim_mutate("RPP: xmax shifted from 5 to 6",
                    lambda t: re.sub(r"(RPP\s+-5\.?0*\s+)5\.?0*", r"\g<1>6.0", t, count=1),
                    r"Geometry:")

        # 2. RCC radius perturbed
        prim_mutate("RCC: radius perturbed from 4 to 4.5",
                    lambda t: re.sub(r"(RCC\s+15\.?0*\s+0\.?0*\s+-8\.?0*\s+0\.?0*\s+0\.?0*\s+16\.?0*\s+)4\.?0*", r"\g<1>4.5", t, count=1),
                    r"Geometry:")

        # 3. Shared face preservation
        r_shared = export_model(shared_face_box_model(), work, "shared_boxes")
        check(r_shared["ok"], "shared boxes: deck validates")
        shared_text = open(r_shared["runnable"]).read()
        # Ensure neither box stole the shared plane or converted to RPP
        check(re.search(r"^\s*\d+\s+RPP\b", shared_text, re.M) is None,
              "shared boxes: adjacent boxes sharing face preserved as planes, no invalid RPP conversion")

        print("3. Unsupported features are refused")
        try:
            export_model(shielding_model(absorption_in_fuel=True), work, "absorb")
            check(False, "absorption in fuel refused")
        except UnsupportedFeature as e:
            check("absorption" in str(e) and "actinides" in str(e), "absorption in fuel refused with reason")
        m = shielding_model()
        m.materials[0]._sab = [("c_made_up_table", 1.0)]
        try:
            export_model(m, work, "badsab")
            check(False, "unknown S(a,b) refused")
        except UnsupportedFeature as e:
            check("c_made_up_table" in str(e) and "--sab" in str(e), "unknown S(a,b) refused with how to fix")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print("  - " + f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
