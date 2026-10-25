"""Tori in the MCNP deck: MCNPy has no torus, MCNP has TX, TY and TZ.

MCNPy's translator stops on an OpenMC XTorus, YTorus or ZTorus (it finds no MCNP card for the class and fails with an AttributeError on None). An OpenMC torus

    XTorus / YTorus / ZTorus (x0, y0, z0, a, b, c):   (s - s0)^2 / b^2  +  (sqrt(rho^2) - a)^2 / c^2  =  1

(s along the torus axis, rho across it) is exactly what MCNP's TX / TY / TZ card describes (manual 6.3, section 5.3, equation 5.1: "s^2/B^2 + (r - A)^2/C^2 = 1" rotated about
the s axis), with A = a, B = b (half-height along the axis) and C = c (half-width across it):

    n  TZ  x0 y0 z0  a b c

plan() finds the tori; mcnpy_view() hands MCNPy a sphere with the torus's number in its place (one unique radius each, never written to the deck); patch_deck() rewrites that
sphere's card into the torus card. Like world_complement and cell_regions this changes how the deck is written, not what it means: validate_deck then compares the written deck with
the OpenMC model point by point, torus cells included.
"""
import contextlib
import re

import openmc

TORI = {openmc.XTorus: "TX", openmc.YTorus: "TY", openmc.ZTorus: "TZ"}
BASE_RADIUS = 7777.0   # the stand-in spheres have radii BASE_RADIUS + 1, + 2, ... (only used to find their cards again)
LINE = 78


def plan(geometry):
    """None, or a list of {'id', 'card', 'values', 'radius', 'surface'} for every torus in the geometry (sorted by id)."""
    tori = sorted((s for s in geometry.get_all_surfaces().values() if type(s) in TORI), key=lambda s: s.id)
    if not tori:
        return None
    out = []
    for k, s in enumerate(tori, 1):
        values = [float(v) for v in (s.x0, s.y0, s.z0, s.a, s.b, s.c)]
        if not (values[3] > 0 and values[4] > 0 and values[5] > 0 and all(v == v and abs(v) != float("inf") for v in values)):
            raise ValueError(f"torus {s.id} needs a, b and c above zero, not {values[3:]}")
        if values[3] < values[5]:
            raise ValueError(f"torus {s.id} is degenerate (major radius a={values[3]:g} below its half-width c={values[5]:g}): its tube closes on itself")
        out.append({"id": s.id, "card": TORI[type(s)], "values": values, "radius": BASE_RADIUS + k, "surface": s})
    return out


def _halfspaces(node):
    if isinstance(node, openmc.Halfspace):
        yield node
    elif isinstance(node, openmc.Complement):
        yield from _halfspaces(node.node)
    elif isinstance(node, (openmc.Intersection, openmc.Union)):
        for child in node:
            yield from _halfspaces(child)


@contextlib.contextmanager
def mcnpy_view(tp, geometry):
    """While MCNPy translates: every torus in a cell region is a stand-in sphere with the torus's surface number."""
    if tp is None:
        yield
        return
    stand = {}
    for t in tp:
        s = t["surface"]
        v = t["values"]
        sph = openmc.Sphere(surface_id=s.id, x0=v[0], y0=v[1], z0=v[2], r=t["radius"], boundary_type=s.boundary_type, name=s.name)
        stand[s.id] = sph
    swapped = []
    try:
        for cell in geometry.get_all_cells().values():
            if cell.region is None:
                continue
            for h in _halfspaces(cell.region):
                if h.surface.id in stand and h.surface is not stand[h.surface.id]:
                    swapped.append((h, h.surface))
                    h.surface = stand[h.surface.id]
        yield
    finally:
        for h, s in swapped:
            h.surface = s


def _fmt(v):
    return "0" if v == 0 else repr(round(float(v), 10))


def patch_deck(text, tp):
    """The deck text with each stand-in sphere's surface card replaced by the torus card of the same number."""
    lines = text.split("\n")
    for t in tp:
        n = t["id"]
        start = next((i for i, ln in enumerate(lines) if re.match(rf"^{n}\s+\S", ln) and _is_sphere_card(ln, t["radius"])), None)
        if start is None:
            raise ValueError(f"the stand-in card of torus {n} is not in the translated deck")
        end = start + 1
        while end < len(lines) and re.match(r"^\s{5,}\S", lines[end]):
            end += 1
        tokens = [str(n), t["card"]] + [_fmt(v) for v in t["values"]]
        out, cur = [], ""
        for tok in tokens:
            if cur and len(cur) + 1 + len(tok) > LINE:
                out.append(cur)
                cur = "     " + tok
            else:
                cur = f"{cur} {tok}" if cur else tok
        out.append(cur)
        lines[start:end] = out
    return "\n".join(lines)


def _is_sphere_card(line, radius):
    tokens = line.replace("$", " $ ").split("$")[0].split()
    try:
        return len(tokens) >= 3 and tokens[1].upper() in ("S", "SO", "SX", "SY", "SZ") and abs(float(tokens[-1]) - radius) < 1e-6
    except ValueError:
        return False


def patch_file(path, tp):
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(patch_deck(text, tp))
