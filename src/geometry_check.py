"""
Geometry equivalence check: does the MCNP deck describe the same geometry as the OpenMC model?

No MCNP executable is needed. Random points are sampled over the model (plus points
inside every cell's bounding box, so small cells aren't missed). For each point:
  - OpenMC says which cell (and material) contains it, or that it's outside.
  - Every MCNP cell's region is evaluated from the MontePy-parsed deck.
Each point must be inside exactly one MCNP cell (0 = undefined space -> lost particles,
2+ = overlapping cells), that cell must be the same cell number as OpenMC's (MCNPy keeps
OpenMC IDs), and points outside the OpenMC model must land in a cell with IMP:N=0.
Cell materials and densities are compared too.

Supported MCNP surfaces: P (4-constant form), PX/PY/PZ, SO, S, SX/SY/SZ, CX/CY/CZ,
C/X, C/Y, C/Z, GQ (rotated cylinders and other general quadrics). Decks with other surface types, transformations, universes/fills or
lattices are reported as not checkable rather than silently passed.
"""
import math

import numpy as np
import openmc
from montepy.surfaces.half_space import HalfSpace, UnitHalfSpace
from montepy.surfaces.half_space import Operator

SURFACE_TOL = 1e-6  # cm; points closer than this to any surface are skipped (on-surface ambiguity)


class NotCheckable(Exception):
    pass


def _surface_fn(s):
    t = str(s.surface_type).upper()
    c = [float(v) for v in s.surface_constants]
    if getattr(s, "transform", None) is not None:
        raise NotCheckable(f"surface {s.number} has a transformation")
    # each returns (signed value, approximate distance scale) for points array P (N,3)
    if t == "P" and len(c) == 4:
        a, b, cc, d = c
        norm = math.sqrt(a * a + b * b + cc * cc)
        return lambda P: (P @ np.array([a, b, cc]) - d, norm)
    if t in ("PX", "PY", "PZ"):
        i = "XYZ".index(t[1])
        return lambda P: (P[:, i] - c[0], 1.0)
    if t in ("SO", "S", "SX", "SY", "SZ"):
        if t == "SO":
            ctr, r = (0, 0, 0), c[0]
        elif t == "S":
            ctr, r = c[:3], c[3]
        else:
            ctr = [0.0, 0.0, 0.0]
            ctr["XYZ".index(t[1])] = c[0]
            r = c[1]
        ctr = np.array(ctr, dtype=float)
        return lambda P: (((P - ctr) ** 2).sum(axis=1) - r * r, 2 * r)
    if t in ("CX", "CY", "CZ", "C/X", "C/Y", "C/Z"):
        axis = "XYZ".index(t[-1])
        a, b = [i for i in range(3) if i != axis]
        if t.startswith("C/"):
            ca, cb, r = c
        else:
            ca, cb, r = 0.0, 0.0, c[0]
        return lambda P: ((P[:, a] - ca) ** 2 + (P[:, b] - cb) ** 2 - r * r, 2 * r)
    if t == "GQ" and len(c) == 10:
        # MCNP order: A x^2 + B y^2 + C z^2 + D xy + E yz + F zx + G x + H y + J z + K
        A, B, C, D, E, F, G, H, J, K = c

        def gq(P):
            x, y, z = P[:, 0], P[:, 1], P[:, 2]
            v = A * x * x + B * y * y + C * z * z + D * x * y + E * y * z + F * z * x + G * x + H * y + J * z + K
            gx = 2 * A * x + D * y + F * z + G
            gy = 2 * B * y + D * x + E * z + H
            gz = 2 * C * z + E * y + F * x + J
            return v, np.sqrt(gx * gx + gy * gy + gz * gz)  # |v| / |grad| ~ distance to the surface
        return gq
    raise NotCheckable(f"surface {s.number} type {t} isn't supported by the geometry check")


def _evaluate(node, P, surf_vals, cell_masks, cells_by_num, stack):
    if isinstance(node, UnitHalfSpace):
        if node.is_cell:
            return _cell_mask(node.divider.number, P, surf_vals, cell_masks, cells_by_num, stack)
        v = surf_vals[node.divider.number]
        return v > 0 if node.side else v < 0
    op = node.operator
    if op in (Operator.GROUP,):
        return _evaluate(node.left, P, surf_vals, cell_masks, cells_by_num, stack)
    if op == Operator.COMPLEMENT:
        return ~_evaluate(node.left, P, surf_vals, cell_masks, cells_by_num, stack)
    left = _evaluate(node.left, P, surf_vals, cell_masks, cells_by_num, stack)
    right = _evaluate(node.right, P, surf_vals, cell_masks, cells_by_num, stack)
    if op == Operator.INTERSECTION:
        return left & right
    if op == Operator.UNION:
        return left | right
    raise NotCheckable(f"geometry operator {op} isn't supported")


def _cell_mask(num, P, surf_vals, cell_masks, cells_by_num, stack):
    if num in cell_masks:
        return cell_masks[num]
    if num in stack:
        raise NotCheckable(f"cell {num} refers to itself through complements")
    stack.add(num)
    mask = _evaluate(cells_by_num[num].geometry, P, surf_vals, cell_masks, cells_by_num, stack)
    stack.discard(num)
    cell_masks[num] = mask
    return mask


def _domain(geometry, has_vacuum):
    lo, hi = geometry.bounding_box
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    finite = np.isfinite(lo) & np.isfinite(hi)
    span = np.where(finite, hi - lo, 0.0)
    ref = span[finite].max() if finite.any() else 10.0
    lo = np.where(finite, lo, -ref)
    hi = np.where(finite, hi, ref)
    if has_vacuum:  # reach a little outside to test the graveyard cell
        pad = np.where(finite, 0.1 * (hi - lo), 0.0)
        lo, hi = lo - pad, hi + pad
    return lo, hi


def check_geometry(problem, geometry, n_samples=20000, per_cell=500, seed=12345):
    """Return dict(ok, checked_points, errors, skipped_near_surface, reason)."""
    result = {"ok": False, "checked_points": 0, "errors": [], "skipped_near_surface": 0, "reason": None}
    try:
        cells = geometry.get_all_cells()
        if any(not isinstance(c.fill, (openmc.Material, type(None))) for c in cells.values()):
            raise NotCheckable("the OpenMC model uses universes or lattices")
        for c in problem.cells:
            if getattr(c, "fill", None) is not None and getattr(c.fill, "universe", None) is not None:
                raise NotCheckable(f"MCNP cell {c.number} has a FILL")
            if c.universe is not None and c.universe.number != 0:
                raise NotCheckable(f"MCNP cell {c.number} is in universe {c.universe.number}")
        surf_fns = {s.number: _surface_fn(s) for s in problem.surfaces}
    except NotCheckable as e:
        result["reason"] = str(e)
        return result

    rng = np.random.default_rng(seed)
    has_vacuum = any(s.boundary_type == "vacuum" for s in geometry.get_all_surfaces().values())
    lo, hi = _domain(geometry, has_vacuum)
    pts = [rng.uniform(lo, hi, size=(n_samples, 3))]
    for cell in geometry.root_universe.cells.values():  # make sure every cell gets hit
        clo, chi = cell.region.bounding_box if cell.region is not None else (lo, hi)
        clo = np.maximum(np.nan_to_num(np.array(clo, dtype=float), neginf=-np.inf), lo)
        chi = np.minimum(np.nan_to_num(np.array(chi, dtype=float), posinf=np.inf), hi)
        if np.all(chi > clo):
            pts.append(rng.uniform(clo, chi, size=(per_cell, 3)))
    P = np.vstack(pts)

    surf_vals, near = {}, np.zeros(len(P), dtype=bool)
    for num, fn in surf_fns.items():
        v, scale = fn(P)
        surf_vals[num] = v
        near |= np.abs(v) / np.maximum(scale, 1e-12) < SURFACE_TOL
    keep = ~near
    P = P[keep]
    surf_vals = {k: v[keep] for k, v in surf_vals.items()}
    result["skipped_near_surface"] = int(near.sum())

    cells_by_num = {c.number: c for c in problem.cells}
    masks = {}
    try:
        for num in cells_by_num:
            _cell_mask(num, P, surf_vals, masks, cells_by_num, set())
    except NotCheckable as e:
        result["reason"] = str(e)
        return result
    nums = sorted(masks)
    M = np.stack([masks[n] for n in nums], axis=1)  # (points, cells)
    hits = M.sum(axis=1)

    errors = []

    def add(msg):
        if len(errors) < 20:
            errors.append(msg)

    for i, p in enumerate(P):
        found = geometry.find(tuple(p))
        omc_cell = found[-1] if found and isinstance(found[-1], openmc.Cell) else None
        inside = [nums[j] for j in np.flatnonzero(M[i])]
        where = f"({p[0]:.4g}, {p[1]:.4g}, {p[2]:.4g})"
        if hits[i] == 0:
            add(f"point {where} is in no MCNP cell (OpenMC: {'cell ' + str(omc_cell.id) if omc_cell else 'outside'}) -> lost particles")
            continue
        if hits[i] > 1:
            add(f"point {where} is in several MCNP cells {inside} (overlap)")
            continue
        mcnp = cells_by_num[inside[0]]
        if omc_cell is None:
            if mcnp.importance.neutron != 0:
                add(f"point {where} is outside the OpenMC model but in MCNP cell {mcnp.number} with IMP:N={mcnp.importance.neutron}")
        elif mcnp.number != omc_cell.id:
            add(f"point {where}: OpenMC cell {omc_cell.id} ({omc_cell.name}) but MCNP cell {mcnp.number}")
    result["checked_points"] = int(len(P))

    # materials and densities, per cell
    for cid, cell in cells.items():
        mc = cells_by_num.get(cid)
        if mc is None:
            add(f"OpenMC cell {cid} ({cell.name}) has no MCNP cell {cid}")
            continue
        fill = cell.fill
        mnum = mc.material.number if mc.material is not None else 0
        if fill is None:
            if mnum != 0:
                add(f"cell {cid}: void in OpenMC but material {mnum} in MCNP")
            continue
        if mnum != fill.id:
            add(f"cell {cid}: material {fill.id} in OpenMC but {mnum} in MCNP")
            continue
        if fill.density_units == "g/cm3":
            if mc.is_atom_dens or not math.isclose(mc.mass_density, fill.density, rel_tol=1e-6):
                add(f"cell {cid}: density {fill.density} g/cm3 in OpenMC but "
                    f"{'atom density ' + str(mc.atom_density) if mc.is_atom_dens else str(mc.mass_density) + ' g/cm3'} in MCNP")
    result["errors"] = errors
    result["ok"] = not errors and len(P) > 0
    return result
