"""
Build MCNP run-control, source and tally cards from OpenMC objects.

MCNPy 0.0.7 translates geometry and materials only (see CLAUDE.md "Known quirks").
Everything here is derived from the OpenMC Settings/Tallies objects, never typed
from memory, and every deck that uses these cards is parsed by MontePy and checked
by src/validate_deck.py.

Anything this module can't translate faithfully raises UnsupportedFeature with a
message saying what and why, instead of emitting a card that would run but model
something different.
"""
import math
import re

import openmc
import openmc.stats

# OpenMC S(a,b) table name -> MCNP thermal-scattering (MT card) identifier.
# These are the classic MCNP library names. MCNP data releases name their tables
# differently (e.g. ENDF/B-VIII.0 tables distributed with MCNP6.3 use names like
# "h-h2o.40t"), so check them against your xsdir before running, and override with
# --sab NAME=ID in export_mcnp.py when they differ.
SAB_MCNP_MAP = {
    # Hydrogen / Water
    "c_H_in_H2O": "h-h2o.40t",
    "c_H_in_H2O_solid": "h-ice.40t",
    "c_ortho_H": "orthoH.40t",
    "c_para_H": "paraH.40t",
    # Heavy Water / Deuterium
    "c_D_in_D2O": "d-d2o.40t",
    "c_O_in_D2O": "o-d2o.40t",
    "c_ortho_D": "orthoD.40t",
    "c_para_D": "paraD.40t",
    # Carbon / Graphite
    "c_Graphite": "grph.40t",
    "c_Graphite_10p": "grph10.40t",
    "c_Graphite_30p": "grph30.40t",
    "c_C6H6": "benz.40t",
    # Polymers / Hydrocarbons
    "c_H_in_CH2": "h-poly.40t",
    "c_H_in_C5O2H8": "h-luci.40t",
    "c_H_in_CH4_liquid": "lmeth.40t",
    "c_H_in_CH4_solid": "smeth.40t",
    # Beryllium
    "c_Be": "be-met.40t",
    "c_Be_in_BeO": "be-beo.40t",
    "c_O_in_BeO": "o-beo.40t",
    # Hydrides
    "c_H_in_ZrH": "h-zrh.40t",
    "c_Zr_in_ZrH": "zr-zrh.40t",
    "c_H_in_YH2": "h-yh2.40t",
    "c_Y_in_YH2": "y-yh2.40t",
    # Fuels
    "c_U_in_UO2": "u-uo2.40t",
    "c_O_in_UO2": "o-uo2.40t",
    "c_U_in_UN": "u-un.40t",
    "c_N_in_UN": "n-un.40t",
    # Silicon / Oxides / Carbides
    "c_SiO2_alpha": "sio2.40t",
    "c_SiO2_beta": "sio2.40t",
    "c_Si_in_SiC": "si-sic.40t",
    "c_C_in_SiC": "c-sic.40t",
    # Metals
    "c_Al27": "al-27.40t",
    "c_Fe56": "fe-56.40t",
    "c_O_in_H2O_solid": "o-ice.40t",
}

# OpenMC reaction score -> MCNP FM reaction list (inside the parentheses of an FM bin).
# MCNP's -2 excludes fission, unlike OpenMC's "absorption". They're equal only for materials
# that can't fission, so absorption is exported only for those (see tally_cards). MCNP's
# "-2:-6" sum isn't used because MontePy 1.1.3 can't parse ':' in FM cards.
SCORE_FM = {
    "total": "-1",
    "absorption": "-2",
    "fission": "-6",
    "nu-fission": "-6 -7",
    "elastic": "2",
    "(n,2n)": "16",
    "(n,gamma)": "102",
    "(n,p)": "103",
    "(n,a)": "107",
}

TWO_PI = 2.0 * math.pi


class UnsupportedFeature(ValueError):
    """The OpenMC model uses something this exporter can't translate faithfully."""


def num(x):
    """Format a number for a deck: repr of the float, so 0 -> '0.0' (matches existing decks)."""
    return repr(float(x))


def _close(a, b, rel=1e-9, abs_tol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)


MAX_DIST = 999  # p. 397: SI/SP/SB/DS distribution numbers are 1 to 999


class _Dists:
    """Allocates SI/SP distribution numbers."""

    def __init__(self, start=1):
        self.next = start
        self.cards = []

    def reserve(self):
        n = self.next
        if n > MAX_DIST:
            raise UnsupportedFeature(f"This model needs more than {MAX_DIST} SI/SP/DS distributions, which is MCNP's "
                                     f"limit (manual p. 397). Export fewer sources, or give them fewer distributions.")
        self.next += 1
        return n

    def add(self, si, sp):
        n = self.reserve()
        if si is not None:
            self.cards.append(f"SI{n} {si}")
        self.cards.append(f"SP{n} {sp}")
        return f"D{n}"


# ── run control ────────────────────────────────────────────────────────────────

def uses_photons(settings, tallies=None):
    if settings.photon_transport:
        return True
    return any(getattr(s, "particle", "neutron") == "photon" for s in _sources(settings))


def mode_card(settings, tallies=None):
    return "MODE N P" if uses_photons(settings, tallies) else "MODE N"


def _sources(settings):
    src = settings.source
    if src is None:
        return []
    return list(src) if isinstance(src, (list, tuple)) else [src]


def ksrc_points(settings):
    """Initial fission-source points for KSRC, from the OpenMC source's spatial distribution."""
    pts = []
    for s in _sources(settings):
        space = getattr(s, "space", None)
        if space is None:
            pts.append((0.0, 0.0, 0.0))
        elif isinstance(space, openmc.stats.Point):
            pts.append(tuple(space.xyz))
        elif isinstance(space, openmc.stats.Box):
            pts.append(tuple((lo + hi) / 2 for lo, hi in zip(space.lower_left, space.upper_right)))
        elif isinstance(space, (openmc.stats.SphericalIndependent, openmc.stats.CylindricalIndependent)):
            pts.append(tuple(space.origin))
        else:
            raise UnsupportedFeature(f"KSRC: can't pick a start point from a {type(space).__name__} source.")
    return pts or [(0.0, 0.0, 0.0)]


def eigenvalue_cards(settings):
    """KSRC and KCODE. The 1.0 is MCNP's initial k-eff guess; OpenMC has no counterpart."""
    pts = ksrc_points(settings)
    ksrc = "KSRC " + " ".join(num(c) for p in pts for c in p)
    kcode = f"KCODE {int(settings.particles)} 1.0 {int(settings.inactive or 0)} {int(settings.batches)}"
    return ksrc, kcode


# ── fixed source ───────────────────────────────────────────────────────────────

def _energy(dist, dists):
    """Return the ERG= value (eV -> MeV)."""
    if dist is None:  # OpenMC's default source energy is a Watt spectrum with these parameters
        dist = openmc.stats.Watt(a=0.988e6, b=2.249e-6)
    if isinstance(dist, openmc.stats.Discrete):
        x, p = list(dist.x), list(dist.p)
        if len(x) == 1:
            return num(x[0] / 1e6)
        return dists.add("L " + " ".join(num(e / 1e6) for e in x), "D " + " ".join(num(w) for w in p))
    if isinstance(dist, openmc.stats.Watt):
        return dists.add(None, f"-3 {num(dist.a / 1e6)} {num(dist.b * 1e6)}")
    if isinstance(dist, openmc.stats.Maxwell):
        return dists.add(None, f"-2 {num(dist.theta / 1e6)}")
    if isinstance(dist, openmc.stats.Uniform):
        return dists.add(f"H {num(dist.a / 1e6)} {num(dist.b / 1e6)}", "0 1")
    if isinstance(dist, openmc.stats.Tabular):
        if dist.interpolation != "histogram":
            raise UnsupportedFeature(f"Tabular energy distribution with interpolation '{dist.interpolation}' isn't supported (only 'histogram').")
        edges = [num(e / 1e6) for e in dist.x]
        probs = [num(p) for p in dist.p]
        if len(probs) == len(edges) - 1:
            sp_vals = ["0"] + probs
        else:
            sp_vals = probs
        return dists.add("H " + " ".join(edges), "D " + " ".join(sp_vals))
    raise UnsupportedFeature(f"Source energy distribution {type(dist).__name__} isn't supported.")


def _require_full_angle(space, what):
    phi = space.phi
    if not (isinstance(phi, openmc.stats.Uniform) and _close(phi.a, 0) and _close(phi.b, TWO_PI, rel=1e-6)):
        raise UnsupportedFeature(f"{what} source: only a full 0-2π azimuthal range is supported.")


def _check_source(s, i=None):
    """Refusals shared by the single- and multi-source SDEF writers. Returns the MCNP particle number."""
    what = "Source" if i is None else f"Source {i + 1}"
    if not isinstance(s, openmc.IndependentSource):
        raise UnsupportedFeature(f"{type(s).__name__} sources aren't supported (only IndependentSource).")
    if getattr(s, "time", None) is not None:
        raise UnsupportedFeature(f"{what}: source time distributions aren't supported.")
    if getattr(s, "constraints", None):
        raise UnsupportedFeature(f"{what}: source domain constraints aren't supported.")
    particle = getattr(s, "particle", "neutron")
    if particle not in ("neutron", "photon"):
        raise UnsupportedFeature(f"{what}: source particle '{particle}' isn't supported.")
    angle = s.angle
    if angle is not None and not isinstance(angle, (openmc.stats.Isotropic, openmc.stats.Monodirectional)):
        raise UnsupportedFeature(f"{what}: source angle {type(angle).__name__} isn't supported.")
    return 1 if particle == "neutron" else 2


def _source_shape(s, i):
    """(kind, params) of a source's space: point (xyz), box (lo, hi), sphere (origin, r_in, r_out) or
    cylinder (origin, r_in, r_out, z_lo, z_hi), with the same limits as the single-source writer."""
    space = s.space
    if space is None:
        return "point", (0.0, 0.0, 0.0)
    if isinstance(space, openmc.stats.Point):
        return "point", tuple(float(c) for c in space.xyz)
    if isinstance(space, openmc.stats.Box):
        return "box", (tuple(map(float, space.lower_left)), tuple(map(float, space.upper_right)))
    if isinstance(space, openmc.stats.SphericalIndependent):
        r, ct = space.r, space.cos_theta
        if not (isinstance(r, openmc.stats.PowerLaw) and _close(r.n, 2)):
            raise UnsupportedFeature(f"Source {i + 1} (sphere): radius must be uniform in volume (PowerLaw n=2).")
        if not (isinstance(ct, openmc.stats.Uniform) and _close(ct.a, -1) and _close(ct.b, 1)):
            raise UnsupportedFeature(f"Source {i + 1} (sphere): only a full polar range (cos θ from -1 to 1) is supported.")
        _require_full_angle(space, f"Source {i + 1} (sphere)")
        return "sphere", (tuple(map(float, space.origin)), float(r.a), float(r.b))
    if isinstance(space, openmc.stats.CylindricalIndependent):
        r, z = space.r, space.z
        if not (isinstance(r, openmc.stats.PowerLaw) and _close(r.n, 1)):
            raise UnsupportedFeature(f"Source {i + 1} (cylinder): radius must be uniform in area (PowerLaw n=1).")
        if not isinstance(z, openmc.stats.Uniform):
            raise UnsupportedFeature(f"Source {i + 1} (cylinder): height must be a Uniform distribution.")
        _require_full_angle(space, f"Source {i + 1} (cylinder)")
        return "cylinder", (tuple(map(float, space.origin)), float(r.a), float(r.b), float(z.a), float(z.b))
    raise UnsupportedFeature(f"Source {i + 1}: source space {type(space).__name__} isn't supported.")


def _multi_source_sdef(sources):
    """One SDEF for several independent sources (manual p. 379-408).

    ERG is the independent variable: SI1 S lists one energy distribution per source and SP1 their strengths, so
    sampling ERG picks the source (SI option S, p. 397). Every other variable depends on ERG (KEY=FERG=Dn, p. 379)
    through a DS card with one entry per source: DS L for values (POS, VEC, PAR), DS S for distributions (RAD,
    EXT, X, Y, Z, DIR) (p. 402-403; the pattern of Examples 12-13, p. 408). ERG is the selector rather than POS
    because position keywords may not depend on POS (p. 379).

    MCNP picks one volume shape per SDEF from the keywords present (X/Y/Z: box, AXS: cylinder, else sphere around
    POS; p. 387), so point sources mix with any one of box, sphere or cylinder sources, but two different volume
    shapes can't share the card. Returns (sdef words, distribution cards)."""
    pars = [_check_source(s, i) for i, s in enumerate(sources)]
    shapes = [_source_shape(s, i) for i, s in enumerate(sources)]
    volume = sorted({k for k, _ in shapes} - {"point"})
    if len(volume) > 1:
        raise UnsupportedFeature(
            f"Sources mix {' and '.join(volume)} shapes. MCNP's single SDEF card has one volume shape (chosen by its "
            f"keywords: X/Y/Z box, AXS cylinder, POS/RAD sphere, manual p. 387), so point sources can go with any one "
            f"of them but two different volume shapes can't. Use one shape, or points.")
    family = volume[0] if volume else "point"
    dists = _Dists()
    fixed = lambda value: dists.add(f"L {value}", "1")[1:]  # a single value as a discrete distribution number
    energies = []
    for s in sources:
        e = _energy(s.energy, dists)
        energies.append(e[1:] if e.startswith("D") else fixed(e))
    sel = dists.reserve()
    dists.cards.append(f"SI{sel} S " + " ".join(energies))
    dists.cards.append(f"SP{sel} " + " ".join(num(float(s.strength)) for s in sources))

    def dep(option, values):
        n = dists.reserve()
        dists.cards.append(f"DS{n} {option} " + " ".join(values))
        return f"FERG=D{n}"

    words = [f"PAR={pars[0]}" if len(set(pars)) == 1 else "PAR=" + dep("L", [str(p) for p in pars]), f"ERG=D{sel}"]
    xyz = lambda v: " ".join(num(c) for c in v)
    if family == "box":
        for k, axis in enumerate("XYZ"):
            subs = [fixed(num(p[k])) if kind == "point" else dists.add(f"H {num(p[0][k])} {num(p[1][k])}", "0 1")[1:]
                    for kind, p in shapes]
            words.append(f"{axis}=" + dep("S", subs))
    else:
        words.append("POS=" + dep("L", [xyz(p if kind == "point" else p[0]) for kind, p in shapes]))
        if family == "cylinder":
            words.append("AXS=0.0 0.0 1.0")
        if family in ("sphere", "cylinder"):
            power = "2" if family == "sphere" else "1"
            words.append("RAD=" + dep("S", [fixed("0") if kind == "point" else dists.add(f"{num(p[1])} {num(p[2])}", f"-21 {power}")[1:]
                                            for kind, p in shapes]))
        if family == "cylinder":
            words.append("EXT=" + dep("S", [fixed("0") if kind == "point" else dists.add(f"{num(p[3])} {num(p[4])}", "-21 0")[1:]
                                            for kind, p in shapes]))
    mono = [isinstance(s.angle, openmc.stats.Monodirectional) for s in sources]
    if any(mono):
        words.append("VEC=" + dep("L", [xyz(s.angle.reference_uvw) if m else "0.0 0.0 1.0" for s, m in zip(sources, mono)]))
        words.append("DIR=" + dep("S", [fixed("1") if m else dists.add("H -1 1", "0 1")[1:] for m in mono]))
    return words, dists.cards


def fixed_source_cards(settings):
    """SDEF (+ SI/SP/DS) and NPS. Several sources go through _multi_source_sdef."""
    sources = _sources(settings)
    if not sources:
        sources = [openmc.IndependentSource()]  # OpenMC default: point at origin, isotropic, Watt
    nps = f"NPS {int(settings.particles) * int(settings.batches)}"
    if len(sources) > 1:
        words, cards = _multi_source_sdef(sources)
        return ["SDEF " + words[0] + "".join(f"\n     {w}" for w in words[1:])] + cards + [nps]
    s = sources[0]
    if not isinstance(s, openmc.IndependentSource):
        raise UnsupportedFeature(f"{type(s).__name__} sources aren't supported (only IndependentSource).")
    if getattr(s, "time", None) is not None:
        raise UnsupportedFeature("Source time distributions aren't supported.")
    if getattr(s, "constraints", None):
        raise UnsupportedFeature("Source domain constraints aren't supported.")
    particle = getattr(s, "particle", "neutron")
    if particle not in ("neutron", "photon"):
        raise UnsupportedFeature(f"Source particle '{particle}' isn't supported.")

    dists = _Dists()
    words = [f"PAR={1 if particle == 'neutron' else 2}"]
    space = s.space
    if space is None:
        words.append("POS=0.0 0.0 0.0")
    elif isinstance(space, openmc.stats.Point):
        words.append("POS=" + " ".join(num(c) for c in space.xyz))
    elif isinstance(space, openmc.stats.Box):
        for axis, lo, hi in zip("XYZ", space.lower_left, space.upper_right):
            words.append(f"{axis}={dists.add(f'H {num(lo)} {num(hi)}', '0 1')}")
    elif isinstance(space, openmc.stats.SphericalIndependent):
        r, ct = space.r, space.cos_theta
        if not (isinstance(r, openmc.stats.PowerLaw) and _close(r.n, 2)):
            raise UnsupportedFeature("Sphere source: radius must be uniform in volume (PowerLaw n=2).")
        if not (isinstance(ct, openmc.stats.Uniform) and _close(ct.a, -1) and _close(ct.b, 1)):
            raise UnsupportedFeature("Sphere source: only a full polar range (cos θ from -1 to 1) is supported.")
        _require_full_angle(space, "Sphere")
        words.append("POS=" + " ".join(num(c) for c in space.origin))
        words.append(f"RAD={dists.add(f'{num(r.a)} {num(r.b)}', '-21 2')}")
    elif isinstance(space, openmc.stats.CylindricalIndependent):
        r, z = space.r, space.z
        if not (isinstance(r, openmc.stats.PowerLaw) and _close(r.n, 1)):
            raise UnsupportedFeature("Cylinder source: radius must be uniform in area (PowerLaw n=1).")
        if not isinstance(z, openmc.stats.Uniform):
            raise UnsupportedFeature("Cylinder source: height must be a Uniform distribution.")
        _require_full_angle(space, "Cylinder")
        words.append("POS=" + " ".join(num(c) for c in space.origin))
        words.append("AXS=0.0 0.0 1.0")
        words.append(f"RAD={dists.add(f'{num(r.a)} {num(r.b)}', '-21 1')}")
        words.append(f"EXT={dists.add(f'{num(z.a)} {num(z.b)}', '-21 0')}")
    else:
        raise UnsupportedFeature(f"Source space {type(space).__name__} isn't supported.")

    angle = s.angle
    if isinstance(angle, openmc.stats.Monodirectional):
        words.append("VEC=" + " ".join(num(c) for c in angle.reference_uvw) + " DIR=1")
    elif angle is not None and not isinstance(angle, openmc.stats.Isotropic):
        raise UnsupportedFeature(f"Source angle {type(angle).__name__} isn't supported.")

    words.append(f"ERG={_energy(s.energy, dists)}")
    sdef = "SDEF " + words[0] + "".join(f"\n     {w}" for w in words[1:])
    nps = f"NPS {int(settings.particles) * int(settings.batches)}"
    return [sdef] + dists.cards + [nps]


# ── tallies ────────────────────────────────────────────────────────────────────

def _can_fission(material):
    """True if the material has any actinide (Z >= 90) nuclide."""
    from openmc.data import zam
    return any(zam(n.name)[0] >= 90 for n in material.nuclides)



def _detector_fm(t, materials, detector_responses):
    """FM multiplier (C, m, R) for a tally with an EnergyFunctionFilter (a detector response).

    The filter holds only numbers, so the response comes from OpenMC Studio's `detector_responses`
    ({tally id: {"material", "nuclide", "mt", "scale"}}) in model.py. MCNP's FM (C m R) multiplies the flux
    by C times material m's cross section for reaction R, summed over m's nuclides by atom fraction:
      macro (N_i * sigma_i, 1/cm): C = total atom density of m (atoms/b-cm)
      micro (sigma of one nuclide, barns): C = 1 / that nuclide's atom fraction in m
    Returns (card_args, note)."""
    info = (detector_responses or {}).get(t.id)
    if not info:
        raise UnsupportedFeature(
            f"Tally '{t.name}': its EnergyFunctionFilter has no detector description, so the matching MCNP FM "
            f"card is unknown. Export it from OpenMC Studio (model.py's detector_responses) or remove the filter.")
    mats = {m.id: m for m in (materials or [])}
    mat = mats.get(info.get("material"))
    if mat is None:
        raise UnsupportedFeature(f"Tally '{t.name}': its detector material isn't in the model's materials.")
    dens = mat.get_nuclide_atom_densities()
    total = sum(dens.values())
    nuc, mt, scale = info.get("nuclide") or "all", int(info["mt"]), info.get("scale", "macro")
    others = [n for n in dens if n != nuc] if nuc != "all" else []
    if scale == "micro":
        if nuc not in dens:
            raise UnsupportedFeature(f"Tally '{t.name}': {nuc} isn't in detector material {mat.id}, so MCNP can't "
                                     f"give its microscopic cross section.")
        c = total / dens[nuc]
    else:
        c = total
        if nuc != "all" and nuc not in dens:
            raise UnsupportedFeature(f"Tally '{t.name}': {nuc} isn't in detector material {mat.id}.")
    note = (f"Tally '{t.name}': detector response ({scale}, MT {mt}) as FM with material {mat.id}; C = "
            + (f"1/atom fraction of {nuc}" if scale == "micro" else "its atom density") + ".")
    if others:
        note += (f" MCNP sums reaction {mt} over every nuclide in material {mat.id} ({', '.join(sorted(dens))}); "
                 f"OpenMC used only {nuc}. They agree if the others don't have that reaction.")
    return f"{num(c)} {mat.id} {mt}", note


def _cyl_fmesh(n, t, m):
    """FMESH GEOM=CYL for an openmc.CylindricalMesh about z: I = radius, J = height, K = angle (revolutions)."""
    r, phi, z = (list(map(float, g)) for g in (m.r_grid, m.phi_grid, m.z_grid))
    if abs(r[0]) > 1e-12 or abs(phi[0]) > 1e-12:
        raise UnsupportedFeature(f"Tally '{t.name}': MCNP cylindrical meshes start at r = 0 and angle 0.")
    ox, oy, oz = (float(v) for v in m.origin)
    lst = lambda vals: " ".join(num(v) for v in vals)
    ones = lambda vals: " ".join("1" for _ in vals)
    return (f"FMESH{n}:N GEOM=CYL ORIGIN={num(ox)} {num(oy)} {num(oz + z[0])} AXS=0 0 1 VEC=1 0 0"
            f"\n     IMESH={lst(r[1:])} IINTS={ones(r[1:])}"
            f"\n     JMESH={lst(v - z[0] for v in z[1:])} JINTS={ones(z[1:])}"
            f"\n     KMESH={lst(v / (2 * 3.141592653589793) for v in phi[1:])} KINTS={ones(phi[1:])}")


def _instance_bins(t, f, geometry, lattices):
    """MCNP tally bins for a CellInstanceFilter. Each (cell, instance) becomes the path from that cell up to
    universe 0, (c < L[i j k] < ... < c0) (manual p. 452-455), read from OpenMC's instance path
    u4->c4->l2(1,0,0)->u1->c1. L is the lattice's MCNP LAT cell and [i j k] its MCNP index, both from
    lattice_cards.rewrite(). Returns [(bin text, openmc cell)]."""
    cells = geometry.get_all_cells()
    geometry.determine_paths()
    out = []
    for cid, inst in f.bins:
        cell = cells.get(int(cid))
        if cell is None:
            raise UnsupportedFeature(f"Tally '{t.name}' refers to cell {cid}, which isn't in the geometry.")
        paths = cell.paths
        if not 0 <= int(inst) < len(paths):
            raise UnsupportedFeature(f"Tally '{t.name}': cell {cid} has no instance {inst} ({len(paths)} instances).")
        levels = []
        for part in paths[int(inst)].split("->"):
            if part[0] == "c":
                levels.append(part[1:])
            elif part[0] == "l":
                m = re.fullmatch(r"l(\d+)\(([-\d,]+)\)", part)
                L, idx = int(m.group(1)), tuple(int(v) for v in m.group(2).split(","))
                idx += (0,) * (3 - len(idx))
                info = (lattices or {}).get(L)
                if info is None or idx not in info["index"]:
                    raise UnsupportedFeature(f"Tally '{t.name}': lattice {L} element {idx} (cell {cid}, instance {inst}) "
                                             f"has no MCNP lattice element to tally.")
                levels.append(f"{info['cell']}[{' '.join(map(str, info['index'][idx]))}]")
        out.append((levels[0] if len(levels) == 1 else "(" + " < ".join(reversed(levels)) + ")", cell))
    return out


def _bin_card(head, bins, width=78):
    """A tally card whose bins don't fit on one line: continuation lines start with 5 spaces, and a bin
    (which may be a chain with spaces) is never split."""
    lines, line = [], head
    for b in bins:
        if len(line) + 1 + len(b) > width and line.strip():
            lines.append(line)
            line = "     " + b
        else:
            line += " " + b
    return "\n".join(lines + [line])


def _flip(side):
    return "-" if side == "+" else "+"


def _literals(region):
    """A cell region as (literals, carved): literals [(surface, side)] that are all ANDed, and carved, a list of
    convex shapes (each a literal list) subtracted from it, as Studio writes `shape & world & ~(a | b)`.
    None if the region has a union that isn't under a complement."""
    if region is None:
        return [], []
    if isinstance(region, openmc.Halfspace):
        return [(region.surface, region.side)], []
    if isinstance(region, openmc.Intersection):
        lits, carved = [], []
        for r in region:
            sub = _literals(r)
            if sub is None:
                return None
            lits += sub[0]
            carved += sub[1]
        return lits, carved
    if isinstance(region, openmc.Complement):
        inner = region.node
        if isinstance(inner, openmc.Halfspace):
            return [(inner.surface, _flip(inner.side))], []
        if isinstance(inner, openmc.Complement):
            return _literals(inner.node)
        if isinstance(inner, openmc.Union):  # ~(a | b) = ~a & ~b
            return _literals(openmc.Intersection([openmc.Complement(r) for r in inner]))
        sub = _literals(inner)
        if sub is None or sub[1]:
            return None
        return [], [sub[0]]
    if isinstance(region, openmc.Union):  # a union of half-spaces is a convex shape carved out: a | b = ~(~a & ~b)
        halves = _union_halfspaces(region)
        if halves is not None:
            return [], [[(s, _flip(sd)) for s, sd in halves]]
    return None


def _union_halfspaces(region):
    """[(surface, side)] if the region is a union of half-spaces (OpenMC writes ~(box) this way), else None."""
    if isinstance(region, openmc.Halfspace):
        return [(region.surface, region.side)]
    if isinstance(region, openmc.Complement) and isinstance(region.node, openmc.Halfspace):
        return [(region.node.surface, _flip(region.node.side))]
    if isinstance(region, openmc.Union):
        out = []
        for r in region:
            sub = _union_halfspaces(r)
            if sub is None:
                return None
            out += sub
        return out
    return None


def _box(lits):
    return openmc.Intersection([+s if sd == "+" else -s for s, sd in lits]).bounding_box


def _boxes_meet(a, b, tol=1e-9):
    return all(a[0][k] <= b[1][k] + tol and b[0][k] <= a[1][k] + tol for k in range(3))


def _current_patch(t, s, c):
    """How MCNP picks out the crossings OpenMC counts in bin (surface s, CellFromFilter cell c).

    OpenMC scores every crossing of s by a particle that was in c, +1 towards s's positive side and -1 towards
    its negative side. When c lies on one side of s (its region has s as an ANDed half-space), those are the
    particles leaving c through its face on s. MCNP's F1 counts crossings of all of s in both directions
    (manual p. 119, 460), so the face is cut out with FS and the direction with C 0 1 (p. 459-460, 474-475):
    FS lists the complement of each of c's other half-spaces, so segment K+1 is the face. A carved-out shape
    (an earlier Studio part) that reaches the face adds its complemented half-spaces too, and the face is then
    the sum of those segments. Returns (fs literals, face segments, cosine bin, sign), or None when s doesn't
    bound c (the bin is always 0)."""
    where = f"Tally '{t.name}': surface {s.id} leaving cell {c.id} ({c.name})"
    flat = _literals(c.region)
    if flat is None:
        raise UnsupportedFeature(f"{where}: the cell's region has a union, so its face on the surface can't be "
                                 f"cut out with an FS card.")
    lits, carved = flat
    sides = {sd for surf, sd in lits if surf.id == s.id}
    in_carved = any(surf.id == s.id for shape in carved for surf, _ in shape)
    if not sides:
        if in_carved:
            raise UnsupportedFeature(
                f"{where}: the surface belongs to a part carved out of this cell, so the cell lies on both sides "
                f"of it and OpenMC counts crossings of the whole surface inside the cell. MCNP can't reproduce "
                f"that; tally these parts in separate surface tallies.")
        return None
    if len(sides) > 1:
        raise UnsupportedFeature(f"{where}: the cell uses both sides of the surface.")
    side = sides.pop()
    others, seen = [], set()
    for surf, sd in lits:
        if surf.id != s.id and (surf.id, sd) not in seen:
            seen.add((surf.id, sd))
            others.append((surf, sd))
    lo, hi = (list(v) for v in _box(others + [(s, side)]))
    axis = {openmc.XPlane: (0, "x0"), openmc.YPlane: (1, "y0"), openmc.ZPlane: (2, "z0")}.get(type(s))
    if axis:  # an axis plane's face is flat along its axis
        lo[axis[0]] = hi[axis[0]] = getattr(s, axis[1])
    touching = []
    for shape in carved:
        own = [sd for surf, sd in shape if surf.id == s.id]
        if own and own[0] != side:
            continue  # the carved part sits across the surface: it borders the face but doesn't cut it
        rest = [(surf, sd) for surf, sd in shape if surf.id != s.id]
        if not rest:
            raise UnsupportedFeature(f"{where}: a carved-out part covers the whole face.")
        if _boxes_meet((lo, hi), _box(rest)):
            touching.append(rest)
    if len(touching) > 1:
        raise UnsupportedFeature(f"{where}: {len(touching)} other parts cut into this face; the FS card can "
                                 f"exclude only one.")
    fs = [(surf, _flip(sd)) for surf, sd in others]
    segs = [len(fs) + 1]
    if touching:
        fs += [(surf, _flip(sd)) for surf, sd in touching[0]]
        segs = list(range(segs[0], len(fs) + 1))
    cos_bin, sign = (2, 1) if side == "-" else (1, -1)
    return fs, segs, cos_bin, sign


def current_bins(t, geometry):
    """The (surface, cell or None) pairs of a current tally that get an MCNP tally, in order, and those that are
    always 0 in OpenMC (the surface doesn't bound the cell)."""
    surf_f = next(f for f in t.filters if isinstance(f, openmc.SurfaceFilter))
    from_f = next((f for f in t.filters if isinstance(f, openmc.CellFromFilter)), None)
    surfaces, cells = geometry.get_all_surfaces(), geometry.get_all_cells()
    pairs, zero = [], []
    for sid in surf_f.bins:
        for cid in (from_f.bins if from_f is not None else [None]):
            c = cells.get(int(cid)) if cid is not None else None
            if c is not None and int(sid) not in (c.region.get_surfaces() if c.region is not None else {}):
                zero.append((int(sid), c.id))
            else:
                pairs.append((surfaces.get(int(sid)), c))
    return pairs, zero


def _current_cards(t, geometry, e_card, k, notes):
    """F1 + FC + C + FS (+ E) per bin of a current tally. The FC card ends with a tag the validator reads:
    [S s C c SEG a-b COS n X+1] = OpenMC's value is the sum of FS segments a-b in cosine bin n, times +1;
    [S s NET] = cosine bin 2 minus bin 1 (a SurfaceFilter with no CellFromFilter). Returns (cards, next k)."""
    name = t.name or f"tally {t.id}"
    pairs, zero = current_bins(t, geometry)
    out = []
    for s, c in pairs:
        if s is None:
            raise UnsupportedFeature(f"Tally '{name}' refers to a surface that isn't in the geometry.")
        if s.boundary_type in ("reflective", "periodic", "white"):
            raise UnsupportedFeature(f"Tally '{name}': surface {s.id} is {s.boundary_type}; current tallies on it "
                                     f"aren't exported (MCNP and OpenMC count reflected crossings differently).")
        patch = _current_patch(t, s, c) if c is not None else None
        n = 10 * k + 1
        k += 1
        if c is None:
            tag = f"[S {s.id} NET]"
        else:
            fs, segs, cos_bin, sign = patch
            seg = f"{segs[0]}-{segs[-1]}" if len(segs) > 1 else str(segs[0])
            tag = f"[S {s.id} C {c.id} SEG {seg} COS {cos_bin} X{sign:+d}]"
        out += [f"F{n}:N {s.id}", f"FC{n} {name[:78 - 7 - len(tag)]} {tag}", f"C{n} 0 1"]
        if c is not None and fs:
            out.append(_bin_card(f"FS{n}", [("-" if sd == "-" else "") + str(surf.id) for surf, sd in fs]))
        if e_card:
            out.append(f"E{n} {e_card}")
    notes.append(f"Tally '{name}': one MCNP F1 per (surface, part) bin. F1 counts crossings without a sign, so each "
                 f"FC card ends with where OpenMC's value is: [S s C c SEG a-b COS n X+1] = the sum of FS segments "
                 f"a-b in cosine bin n (1 = towards the surface's negative side, 2 = positive) times the sign; "
                 f"[S s NET] = cosine bin 2 minus bin 1.")
    if zero:
        notes.append(f"Tally '{name}': (surface, cell) bins {zero} are always 0 in OpenMC (the surface doesn't "
                     f"bound that cell), so they have no MCNP tally.")
    return out, k


def _e_card(t, energy_f, notes):
    if energy_f is None:
        return None
    edges = [float(e) for e in energy_f.values]
    if edges[0] > 0:
        notes.append(f"Tally '{t.name}': MCNP energy bins start at 0, so there is an extra "
                     f"bin below {edges[0]:g} eV that OpenMC doesn't have.")
    return " ".join(num(e / 1e6) for e in edges[1:])


def tally_cards(tallies, geometry, materials=None, detector_responses=None, lattices=None):
    """F4/E4/FM/SD for cell tallies and FMESH for regular-mesh tallies.

    Returns (cards, notes). Cell tallies get SD=1 so MCNP reports volume-integrated
    values like OpenMC (MCNP's F4 divides by volume by default). One MCNP tally per
    OpenMC score, since flux and each reaction need their own multiplier.
    A CellInstanceFilter (one cell in one lattice element) becomes chain bins; `lattices` is the index map
    lattice_cards.rewrite() fills.
    """
    cards, notes = [], []
    cells = geometry.get_all_cells()
    k = 0
    for t in tallies or []:
        if t.nuclides and list(t.nuclides) != ["total"]:
            raise UnsupportedFeature(f"Tally '{t.name}': per-nuclide tallies aren't supported.")
        cell_f = inst_f = energy_f = mesh_f = energy_fn_f = surf_f = from_f = None
        for f in t.filters:
            if isinstance(f, openmc.CellFilter):
                cell_f = f
            elif isinstance(f, openmc.CellInstanceFilter):
                inst_f = f
            elif isinstance(f, openmc.EnergyFilter):
                energy_f = f
            elif isinstance(f, openmc.MeshFilter):
                mesh_f = f
            elif isinstance(f, openmc.ParticleFilter) and list(f.bins) == ["neutron"]:
                pass
            elif isinstance(f, openmc.EnergyFunctionFilter):
                energy_fn_f = f
            elif isinstance(f, openmc.SurfaceFilter):
                surf_f = f
            elif isinstance(f, openmc.CellFromFilter):
                from_f = f
            else:
                raise UnsupportedFeature(f"Tally '{t.name}': {type(f).__name__} isn't supported.")
        if surf_f is not None or from_f is not None or "current" in t.scores:
            if surf_f is None or any(f is not None for f in (cell_f, inst_f, mesh_f, energy_fn_f)) or \
                    list(t.scores) != ["current"]:
                raise UnsupportedFeature(f"Tally '{t.name}': a current tally needs a SurfaceFilter (a CellFromFilter "
                                         f"and an EnergyFilter are optional), only the score 'current', and no cell, "
                                         f"mesh or detector filter.")
            new, k = _current_cards(t, geometry, _e_card(t, energy_f, notes), k, notes)
            cards += new
            continue
        if sum(f is not None for f in (cell_f, inst_f, mesh_f)) != 1:
            raise UnsupportedFeature(f"Tally '{t.name}': needs exactly one CellFilter, CellInstanceFilter or MeshFilter.")
        bins = None  # [(MCNP bin text, openmc cell)]
        if cell_f is not None:
            ids = [int(c) for c in cell_f.bins]
            missing = [c for c in ids if c not in cells]
            if missing:
                raise UnsupportedFeature(f"Tally '{t.name}' refers to cells {missing} that aren't in the geometry.")
            bins = [(str(c), cells[c]) for c in ids]
        elif inst_f is not None:
            bins = _instance_bins(t, inst_f, geometry, lattices)

        e_card = _e_card(t, energy_f, notes)

        for score in t.scores:
            n = 10 * k + 4
            k += 1
            label = f"{t.name or 'tally ' + str(t.id)} ({score})"
            if score != "flux" and score not in SCORE_FM:
                raise UnsupportedFeature(f"Tally '{t.name}': score '{score}' isn't supported "
                                         f"(supported: flux, {', '.join(SCORE_FM)}).")
            if bins is not None:
                cards.append(_bin_card(f"F{n}:N", [b for b, _ in bins]))
                cards.append(f"FC{n} {label}")
                if score != "flux":
                    mats = {c.fill.id if isinstance(c.fill, openmc.Material) else None for _, c in bins}
                    if len(mats) != 1 or None in mats:
                        raise UnsupportedFeature(f"Tally '{t.name}' score '{score}': all tallied cells must share "
                                                 f"one material (found {sorted(str(m) for m in mats)}).")
                    mat_id = mats.pop()
                    if score == "absorption" and _can_fission(bins[0][1].fill):
                        raise UnsupportedFeature(
                            f"Tally '{t.name}': 'absorption' in material {mat_id} can't be exported faithfully. MCNP's "
                            f"absorption (-2) excludes fission and this material has actinides; tally 'fission' and "
                            f"'(n,gamma)' separately instead.")
                    cards.append(f"FM{n} (-1 {mat_id} {SCORE_FM[score]})")
                elif energy_fn_f is not None:
                    fm, note = _detector_fm(t, materials, detector_responses)
                    cards.append(f"FM{n} ({fm})")
                    notes.append(note)
                if e_card:
                    cards.append(f"E{n} {e_card}")
                cards.append(_bin_card(f"SD{n}", ["1"] * len(bins)))
            else:
                if score != "flux":
                    raise UnsupportedFeature(f"Tally '{t.name}': mesh tallies support only 'flux' "
                                             f"(a mesh spans several materials).")
                m = mesh_f.mesh
                if isinstance(m, openmc.CylindricalMesh):
                    card = _cyl_fmesh(n, t, m)
                elif isinstance(m, openmc.RegularMesh) and len(m.dimension) == 3:
                    (nx, ny, nz), lo, hi = m.dimension, m.lower_left, m.upper_right
                    card = (f"FMESH{n}:N GEOM=XYZ ORIGIN={num(lo[0])} {num(lo[1])} {num(lo[2])}"
                            f"\n     IMESH={num(hi[0])} IINTS={int(nx)} JMESH={num(hi[1])} JINTS={int(ny)}"
                            f"\n     KMESH={num(hi[2])} KINTS={int(nz)}")
                else:
                    raise UnsupportedFeature(f"Tally '{t.name}': only 3D RegularMesh and CylindricalMesh are supported.")
                if energy_fn_f is not None:
                    fm, note = _detector_fm(t, materials, detector_responses)
                    card += f"\n     FM={fm}"
                    notes.append(note)
                if e_card:
                    card += f"\n     EMESH={e_card}"
                cards.append(card)
                notes.append(f"Tally '{t.name}' (FMESH{n}): MCNP mesh results are per cm² (divided by voxel "
                             f"volume); OpenMC's are not. Multiply MCNP values by the voxel volume to compare.")
    return cards, notes
