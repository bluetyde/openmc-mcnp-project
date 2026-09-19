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


class _Dists:
    """Allocates SI/SP distribution numbers."""

    def __init__(self, start=1):
        self.next = start
        self.cards = []

    def add(self, si, sp):
        n = self.next
        self.next += 1
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


def fixed_source_cards(settings):
    """SDEF (+ SI/SP) and NPS for a single independent source."""
    sources = _sources(settings)
    if not sources:
        sources = [openmc.IndependentSource()]  # OpenMC default: point at origin, isotropic, Watt
    if len(sources) > 1:
        raise UnsupportedFeature(f"{len(sources)} sources: only a single source is supported for SDEF export.")
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


def tally_cards(tallies, geometry, materials=None, detector_responses=None):
    """F4/E4/FM/SD for cell tallies and FMESH for regular-mesh tallies.

    Returns (cards, notes). Cell tallies get SD=1 so MCNP reports volume-integrated
    values like OpenMC (MCNP's F4 divides by volume by default). One MCNP tally per
    OpenMC score, since flux and each reaction need their own multiplier.
    """
    cards, notes = [], []
    cells = geometry.get_all_cells()
    k = 0
    for t in tallies or []:
        if t.nuclides and list(t.nuclides) != ["total"]:
            raise UnsupportedFeature(f"Tally '{t.name}': per-nuclide tallies aren't supported.")
        cell_f = energy_f = mesh_f = energy_fn_f = None
        for f in t.filters:
            if isinstance(f, openmc.CellFilter):
                cell_f = f
            elif isinstance(f, openmc.EnergyFilter):
                energy_f = f
            elif isinstance(f, openmc.MeshFilter):
                mesh_f = f
            elif isinstance(f, openmc.ParticleFilter) and list(f.bins) == ["neutron"]:
                pass
            elif isinstance(f, openmc.EnergyFunctionFilter):
                energy_fn_f = f
            elif isinstance(f, (openmc.SurfaceFilter, openmc.CellFromFilter)):
                raise UnsupportedFeature(f"Tally '{t.name}': surface current tallies aren't exported to MCNP yet "
                                         f"(MCNP F1 counts crossings anywhere on a surface, not only on one cell's face). "
                                         f"Remove the tally to export the rest.")
            else:
                raise UnsupportedFeature(f"Tally '{t.name}': {type(f).__name__} isn't supported.")
        if bool(cell_f) == bool(mesh_f):
            raise UnsupportedFeature(f"Tally '{t.name}': needs exactly one CellFilter or MeshFilter.")

        e_card = None
        if energy_f is not None:
            edges = [float(e) for e in energy_f.values]
            e_card = " ".join(num(e / 1e6) for e in edges[1:])
            if edges[0] > 0:
                notes.append(f"Tally '{t.name}': MCNP energy bins start at 0, so there is an extra "
                             f"bin below {edges[0]:g} eV that OpenMC doesn't have.")

        for score in t.scores:
            n = 10 * k + 4
            k += 1
            label = f"{t.name or 'tally ' + str(t.id)} ({score})"
            if score != "flux" and score not in SCORE_FM:
                raise UnsupportedFeature(f"Tally '{t.name}': score '{score}' isn't supported "
                                         f"(supported: flux, {', '.join(SCORE_FM)}).")
            if cell_f is not None:
                ids = [int(c) for c in cell_f.bins]
                missing = [c for c in ids if c not in cells]
                if missing:
                    raise UnsupportedFeature(f"Tally '{t.name}' refers to cells {missing} that aren't in the geometry.")
                cards.append(f"F{n}:N " + " ".join(str(c) for c in ids))
                cards.append(f"FC{n} {label}")
                if score != "flux":
                    mats = {cells[c].fill.id if isinstance(cells[c].fill, openmc.Material) else None for c in ids}
                    if len(mats) != 1 or None in mats:
                        raise UnsupportedFeature(f"Tally '{t.name}' score '{score}': all tallied cells must share "
                                                 f"one material (found {sorted(str(m) for m in mats)}).")
                    mat_id = mats.pop()
                    if score == "absorption" and _can_fission(cells[ids[0]].fill):
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
                cards.append(f"SD{n} " + " ".join("1" for _ in ids))
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
