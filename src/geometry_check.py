"""
Geometry equivalence check: does the MCNP deck describe the same geometry as the OpenMC model?

No MCNP executable is needed. Random points are sampled over the model (plus points
inside every cell's bounding box, so small cells aren't missed). For each point:
  - OpenMC says which cell (and material) contains it, or that it's outside.
  - The MCNP deck is followed down from universe 0: inside each universe, every cell's region is evaluated
    from the MontePy-parsed deck; a cell with FILL passes the point on to the filling universe (shifted by the
    FILL displacement), and a LAT=1 lattice picks the element from the order of its surfaces.
Each point must be inside exactly one cell at every level (0 = undefined space -> lost particles,
2+ = overlapping cells), the cell it ends in must be the same cell number as OpenMC's deepest cell
(MCNPy keeps OpenMC IDs), and points outside the OpenMC model must land in a cell with IMP:N=0.
Cell materials and densities are compared too.

Tally chains: given `chains` [(label, MCNP path, OpenMC instance path)], each point's MCNP path (the cells it
passes through from universe 0 down, with lattice indices) and OpenMC path (u4->c4->l2(1,0,0)->u1->c1, as
Geometry.determine_paths writes it) are recorded, and a bin's MCNP path must hold exactly the points its
OpenMC instance holds. This follows the lattice cards as MCNP reads them, so a wrong [i j k] in a tally bin
is caught even when the element it names holds the same universe.

Lattice rules follow the MCNP 6.3.0 manual (LA-UR-22-30006 Rev. 1, PDF pages): beyond the 1st surface listed
on a LAT=1 cell is element [1,0,0], beyond the 2nd [-1,0,0], and so on for j and k (p. 290); a FILL array
lists universes with i varying fastest and elements outside its ranges don't exist (p. 291-292); a value equal
to the lattice's own universe fills that element with the lattice cell's material (p. 292); FILL=n (o1 o2 o3)
places the filling universe's origin at o in the filled cell's coordinates (p. 291).

Supported MCNP surfaces: P (4-constant form), PX/PY/PZ, SO, S, SX/SY/SZ, CX/CY/CZ,
C/X, C/Y, C/Z, GQ (rotated cylinders and other general quadrics). Lattice elements must be bounded by planes:
LAT=1 rectangular boxes, or LAT=2 hexagonal prisms along z whose faces are listed in the manual's order
(p. 290); a LAT=2 point is placed in the nearest hexagon. Decks with other surface types, surface
transformations, TRCL or rotated fills are reported as not checkable rather than silently passed.
"""
import math
from collections import defaultdict

import numpy as np
import openmc
from montepy.surfaces.half_space import HalfSpace, UnitHalfSpace
from montepy.surfaces.half_space import Operator

SURFACE_TOL = 1e-6  # cm; points closer than this to any surface are skipped (on-surface ambiguity)
LOST, OVERLAP, NEAR = -1, -2, -3


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
    if t in ("SO", "S", "SX", "SY", "SZ", "SPH"):
        if t == "SO":
            ctr, r = (0, 0, 0), c[0]
        elif t in ("S", "SPH"):
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
    if t == "RPP" and len(c) == 6:
        x0, x1, y0, y1, z0, z1 = c

        def rpp(P):
            dx = np.maximum(np.maximum(x0 - P[:, 0], P[:, 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - P[:, 1], P[:, 1] - y1), 0.0)
            dz = np.maximum(np.maximum(z0 - P[:, 2], P[:, 2] - z1), 0.0)
            dist_out = np.sqrt(dx * dx + dy * dy + dz * dz)
            dist_in = np.minimum(
                np.minimum(P[:, 0] - x0, x1 - P[:, 0]),
                np.minimum(np.minimum(P[:, 1] - y0, y1 - P[:, 1]), np.minimum(P[:, 2] - z0, z1 - P[:, 2])),
            )
            v = np.where(dist_out > 0, dist_out, -dist_in)
            return v, 1.0

        return rpp
    if t == "RCC" and len(c) == 7:
        vx, vy, vz, hx, hy, hz, r = [float(v) for v in c]
        V = np.array([vx, vy, vz])
        H = np.array([hx, hy, hz])
        h_len2 = float(np.dot(H, H))
        h_len = math.sqrt(h_len2)
        u = H / h_len

        def rcc(P):
            w = P - V
            t_proj = np.dot(w, u)
            w_perp = w - np.outer(t_proj, u)
            rho = np.sqrt(np.sum(w_perp ** 2, axis=1))

            d_axial_out = np.maximum(np.maximum(-t_proj, t_proj - h_len), 0.0)
            d_radial_out = np.maximum(rho - r, 0.0)
            dist_out = np.sqrt(d_axial_out ** 2 + d_radial_out ** 2)

            dist_in = np.minimum(np.minimum(t_proj, h_len - t_proj), r - rho)
            v = np.where(dist_out > 0, dist_out, -dist_in)
            return v, 1.0

        return rcc
    if t == "BOX" and len(c) == 12:
        c = [float(v) for v in c]
        V = np.array(c[0:3])
        A1 = np.array(c[3:6])
        A2 = np.array(c[6:9])
        A3 = np.array(c[9:12])
        l1, l2, l3 = np.linalg.norm(A1), np.linalg.norm(A2), np.linalg.norm(A3)
        u1, u2, u3 = A1 / l1, A2 / l2, A3 / l3

        def box_fn(P):
            w = P - V
            t1 = np.dot(w, u1)
            t2 = np.dot(w, u2)
            t3 = np.dot(w, u3)
            d1_out = np.maximum(np.maximum(-t1, t1 - l1), 0.0)
            d2_out = np.maximum(np.maximum(-t2, t2 - l2), 0.0)
            d3_out = np.maximum(np.maximum(-t3, t3 - l3), 0.0)
            dist_out = np.sqrt(d1_out ** 2 + d2_out ** 2 + d3_out ** 2)
            dist_in = np.minimum(
                np.minimum(t1, l1 - t1),
                np.minimum(np.minimum(t2, l2 - t2), np.minimum(t3, l3 - t3)),
            )
            v = np.where(dist_out > 0, dist_out, -dist_in)
            return v, 1.0

        return box_fn
    raise NotCheckable(f"surface {s.number} type {t} isn't supported by the geometry check")


def _evaluate(node, sv, masks, cells_by_num, stack):
    if isinstance(node, UnitHalfSpace):
        if node.is_cell:
            return _cell_mask(node.divider.number, sv, masks, cells_by_num, stack)
        v = sv(node.divider.number)
        return v > 0 if node.side else v < 0
    op = node.operator
    if op in (Operator.GROUP,):
        return _evaluate(node.left, sv, masks, cells_by_num, stack)
    if op == Operator.COMPLEMENT:
        return ~_evaluate(node.left, sv, masks, cells_by_num, stack)
    left = _evaluate(node.left, sv, masks, cells_by_num, stack)
    right = _evaluate(node.right, sv, masks, cells_by_num, stack)
    if op == Operator.INTERSECTION:
        return left & right
    if op == Operator.UNION:
        return left | right
    raise NotCheckable(f"geometry operator {op} isn't supported")


def _cell_mask(num, sv, masks, cells_by_num, stack):
    if num in masks:
        return masks[num]
    if num in stack:
        raise NotCheckable(f"cell {num} refers to itself through complements")
    stack.add(num)
    mask = _evaluate(cells_by_num[num].geometry, sv, masks, cells_by_num, stack)
    stack.discard(num)
    masks[num] = mask
    return mask


def _leaves(node):
    """Half-spaces of a cell's geometry in the order they are written."""
    if isinstance(node, UnitHalfSpace):
        return [node]
    out = _leaves(node.left)
    if getattr(node, "right", None) is not None:
        out += _leaves(node.right)
    return out


def _card_text(cell):
    for attr in ("input_lines", "_input_lines"):
        lines = getattr(cell, attr, None)
        if lines:
            return " ".join(lines)
    inp = getattr(cell, "_input", None)
    return " ".join(getattr(inp, "input_lines", []) or []) if inp is not None else ""


def _universe_number(cell):
    return abs(cell.universe.number) if cell.universe is not None else 0


def _halfspace(leaf, surfaces_by_num, cell):
    """A lattice cell's half-space as (unit normal n, d) with the element on the side n . x < d."""
    s = surfaces_by_num[leaf.divider.number]
    t = str(s.surface_type).upper()
    c = [float(v) for v in s.surface_constants]
    if t in ("PX", "PY", "PZ"):
        n, d = np.eye(3)["XYZ".index(t[1])], c[0]
    elif t == "P" and len(c) == 4:
        n, d = np.array(c[:3]), c[3]
    else:
        raise NotCheckable(f"lattice cell {cell.number}: element surfaces must be planes (P, PX, PY, PZ)")
    norm = float(np.linalg.norm(n))
    n, d = n / norm, d / norm
    return (-n, -d) if leaf.side else (n, d)  # positive sense means the element is on the n . x > d side


class _Element:
    """The [0,0,0] element of a LAT=1 or LAT=2 cell, read from its surfaces in the order written (p. 290):
    each pair of opposite faces (a, b) gives an index step T = n_a (d_a + d_b), and element [1,0,0] lies
    beyond the first surface. `index(P)` returns lattice indices, points in the element's own coordinates, and
    which points sit on an element face."""

    def __init__(self, cell, surfaces_by_num):
        self.cell = cell
        lt = int(getattr(cell.lattice_type, "value", cell.lattice_type))
        leaves = _leaves(cell.geometry)
        if any(l.is_cell for l in leaves):
            raise NotCheckable(f"lattice cell {cell.number} uses a cell complement")
        if lt == 1 and len(leaves) == 1:
            leaf = leaves[0]
            s = surfaces_by_num[leaf.divider.number]
            if str(s.surface_type).upper() == "RPP":
                if leaf.side:
                    raise NotCheckable(f"lattice cell {cell.number}: RPP element must have negative sense (-s)")
                x0, x1, y0, y1, z0, z1 = [float(v) for v in s.surface_constants]
                self.pairs = [
                    (np.array([1.0, 0.0, 0.0]), x1, -x0, np.array([x1 - x0, 0.0, 0.0])),
                    (np.array([0.0, 1.0, 0.0]), y1, -y0, np.array([0.0, y1 - y0, 0.0])),
                    (np.array([0.0, 0.0, 1.0]), z1, -z0, np.array([0.0, 0.0, z1 - z0])),
                ]
                self.hex = False
                return
        hs = [_halfspace(l, surfaces_by_num, cell) for l in leaves]
        if lt == 1 and len(hs) not in (4, 6) or lt == 2 and len(hs) not in (6, 8) or lt not in (1, 2):
            raise NotCheckable(f"lattice cell {cell.number}: LAT={lt} with {len(hs)} surfaces")
        self.pairs = []
        for (na, da), (nb, db) in zip(hs[0::2], hs[1::2]):
            if not np.allclose(nb, -na, atol=1e-9):
                raise NotCheckable(f"lattice cell {cell.number}: surfaces {len(self.pairs) * 2 + 1} and "
                                   f"{len(self.pairs) * 2 + 2} aren't opposite faces")
            self.pairs.append((na, da, db, na * (da + db)))
        self.hex = lt == 2
        if self.hex:
            (n1, d1, e1, T1), (n3, d3, e3, T3), (_, _, _, T5) = self.pairs[:3]
            if not np.allclose(T5, T3 - T1, atol=1e-6 * float(np.linalg.norm(T1))):
                raise NotCheckable(f"lattice cell {cell.number}: faces aren't in MCNP's LAT=2 order "
                                   f"([1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [-1,1,0], [1,-1,0])")
            if abs(T1[2]) > 1e-9 or abs(T3[2]) > 1e-9:
                raise NotCheckable(f"lattice cell {cell.number}: hexagonal faces must be parallel to z")
            A = np.array([[n1[0], n1[1]], [n3[0], n3[1]]])
            self.c0 = np.linalg.solve(A, [(d1 - e1) / 2, (d3 - e3) / 2])  # centre of element [0,0,0] in x, y
            self.B = np.linalg.inv(np.array([[T1[0], T3[0]], [T1[1], T3[1]]]))
            self.faces = [(n, d) for n, d, _, _ in self.pairs[:3]] + [(-n, e) for n, _, e, _ in self.pairs[:3]]
            if len(self.pairs) == 4 and abs(abs(self.pairs[3][0][2]) - 1) > 1e-9:
                raise NotCheckable(f"lattice cell {cell.number}: the base planes must be normal to z")
        else:
            N = np.array([n for n, _, _, _ in self.pairs])
            if not np.allclose(N @ N.T, np.eye(len(N)), atol=1e-9):
                raise NotCheckable(f"lattice cell {cell.number}: only rectangular (orthogonal) LAT=1 elements are checked")

    def index(self, P):
        idx = np.zeros((len(P), 3), dtype=np.int64)
        local = P.copy()
        near = np.zeros(len(P), dtype=bool)
        axial = self.pairs[3:] if self.hex else self.pairs
        first = 2 if self.hex else 0
        for k, (n, d, e, T) in enumerate(axial):  # element i spans n . x in (-e + i w, d + i w)
            w = d + e
            t = (P @ n + e) / w
            i = np.floor(t).astype(np.int64)
            near |= np.abs(t - np.round(t)) * w < SURFACE_TOL
            idx[:, first + k] = i
            local -= np.outer(i, T)
        if self.hex:  # nearest hexagon centre: round the (i, j) coordinates as cube coordinates
            T1, T3 = self.pairs[0][3], self.pairs[1][3]
            ab = (local[:, :2] - self.c0) @ self.B.T
            q, r = ab[:, 0], ab[:, 1]
            s = -q - r
            rq, rr, rs = np.round(q), np.round(r), np.round(s)
            dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
            fix_q = (dq > dr) & (dq > ds)
            fix_r = ~fix_q & (dr > ds)
            rq = np.where(fix_q, -rr - rs, rq)
            rr = np.where(fix_r, -rq - rs, rr)
            i, j = rq.astype(np.int64), rr.astype(np.int64)
            idx[:, 0], idx[:, 1] = i, j
            local -= np.outer(i, T1) + np.outer(j, T3)
            for n, d in self.faces:
                near |= np.abs(local @ n - d) < SURFACE_TOL
        return idx, local, near


class _Deck:
    def __init__(self, problem):
        self.cells_by_num = {c.number: c for c in problem.cells}
        self.surfaces_by_num = {s.number: s for s in problem.surfaces}
        self.surf_fns = {n: _surface_fn(s) for n, s in self.surfaces_by_num.items()}
        self.by_universe = defaultdict(list)
        for c in problem.cells:
            self.by_universe[_universe_number(c)].append(c)
            if "TRCL" in _card_text(c).upper():
                raise NotCheckable(f"MCNP cell {c.number} has a TRCL")
            f = c.fill
            tr = getattr(f, "transform", None) if f is not None else None
            if tr is not None:
                rot = getattr(tr, "rotation_matrix", None)
                rot = np.zeros(0) if rot is None else np.asarray(rot, dtype=float).ravel()
                if rot.size and not np.allclose(rot, np.eye(3).ravel()):
                    raise NotCheckable(f"MCNP cell {c.number} fills with a rotation")
                if not getattr(tr, "is_main_to_aux", True):
                    raise NotCheckable(f"MCNP cell {c.number}: FILL displacement given in the universe's system (m = -1)")
        self.elements = {c.number: _Element(c, self.surfaces_by_num)
                         for c in problem.cells if getattr(c, "lattice_type", None) is not None}

    def locate(self, P, universe=0, depth=0, paths=None):
        """MCNP leaf cell number for each point of P (in `universe`'s coordinates), or LOST / OVERLAP / NEAR.
        If `paths` (one list per point) is given, each point's (cell number, lattice index or None) levels are
        appended to it, top level first."""
        out = np.full(len(P), LOST, dtype=np.int64)
        if len(P) == 0:
            return out
        if depth > 20:
            raise NotCheckable("more than 20 universe levels")
        cells = self.by_universe.get(universe, [])
        lat = [c for c in cells if c.number in self.elements]
        if lat:  # a lattice is the only cell of its universe and repeats forever (p. 289): index every point
            if len(cells) != 1:
                raise NotCheckable(f"universe {universe} has a lattice cell and other cells")
            return self._lattice(lat[0], P, depth, paths)
        cache, near = {}, np.zeros(len(P), dtype=bool)

        def sv(num):
            if num not in cache:
                v, scale = self.surf_fns[num](P)
                cache[num] = v
                np.logical_or(near, np.abs(v) / np.maximum(scale, 1e-12) < SURFACE_TOL, out=near)
            return cache[num]
        masks = {}
        for c in cells:
            _cell_mask(c.number, sv, masks, self.cells_by_num, set())
        M = np.stack([masks[c.number] for c in cells], axis=1) if cells else np.zeros((len(P), 0), bool)
        hits = M.sum(axis=1)
        out[hits > 1] = OVERLAP
        for j, c in enumerate(cells):
            sel = np.flatnonzero(M[:, j] & (hits == 1))
            if not len(sel):
                continue
            f = c.fill
            fu = getattr(f, "universe", None) if f is not None else None
            sub = None
            if paths is not None and c.number not in self.elements:
                for p in sel:
                    paths[p].append((c.number, None))
            if paths is not None:
                sub = [paths[p] for p in sel]
            if c.number in self.elements:
                out[sel] = self._lattice(c, P[sel], depth, sub)
            elif fu is not None:
                tr = getattr(f, "transform", None)
                shift = np.asarray(tr.displacement_vector, dtype=float) if tr is not None else np.zeros(3)
                out[sel] = self.locate(P[sel] - shift, abs(fu.number), depth + 1, sub)
            else:
                out[sel] = c.number
        out[near] = NEAR
        return out

    def _lattice(self, c, P, depth, paths=None):
        idx, local, near = self.elements[c.number].index(P)
        if paths is not None:
            for p in range(len(P)):
                paths[p].append((c.number, tuple(int(v) for v in idx[p])))
        f = c.fill
        own = _universe_number(c)
        if getattr(f, "multiple_universes", False):
            lo = np.asarray(f.min_index, dtype=np.int64)
            arr = f.universes
            rel = idx - lo
            shape = np.array(arr.shape)
            inside = np.all((rel >= 0) & (rel < shape), axis=1)
            unum = np.full(len(P), -10**9, dtype=np.int64)
            for p in np.flatnonzero(inside):
                unum[p] = abs(arr[tuple(rel[p])].number)
        else:
            inside = np.ones(len(P), dtype=bool)
            unum = np.full(len(P), abs(f.universe.number), dtype=np.int64)
        out = np.full(len(P), LOST, dtype=np.int64)  # outside the FILL array: the element doesn't exist (p. 291)
        for u in np.unique(unum[inside]):
            sel = np.flatnonzero(inside & (unum == u))
            sub = [paths[p] for p in sel] if paths is not None else None
            out[sel] = c.number if u == own else self.locate(local[sel], int(u), depth + 1, sub)
        out[near] = NEAR
        return out


def _on_surface(s, P, iters=30):
    """Newton-project points onto OpenMC surface s (any type, via evaluate()); returns the points that converged."""
    out = []
    for p in P:
        x = np.array(p, dtype=float)
        for _ in range(iters):
            f = s.evaluate(x)
            g = np.array([(s.evaluate(x + h) - s.evaluate(x - h)) / 2e-6 for h in np.eye(3) * 1e-6])
            gg = float(g @ g)
            if gg < 1e-24:
                break
            x = x - f * g / gg
            if abs(f) / math.sqrt(gg) < 1e-10:
                out.append((x, g / math.sqrt(gg)))
                break
    return out


def check_current_faces(problem, geometry, faces, n_points=600, seed=4321):
    """Check the F1 + FS + C cards of current tallies against OpenMC.

    `faces` = [(label, F1 surface number, OpenMC surface id, OpenMC cell id or None, [(FS surface number, sign)],
    face segments, cosine bin, sign)]. Points are put on the OpenMC surface (around the cell and over the whole
    model), and at each one:
      - the deck's F1 surface must pass through it (the F1 is on the same surface as OpenMC's bin);
      - OpenMC says whether the cell is just across the surface there (a small step to each side) and on which
        side; MCNP's FS card, read from the deck's surfaces, gives the segment; the point must be in one of the
        FS segments declared for the face exactly when OpenMC has the cell there;
      - the cell's side must match the declared cosine bin (2 = leaving towards the positive side) and sign.
    Returns dict(errors, reason, checked)."""
    result = {"errors": [], "reason": None, "checked": 0}
    try:
        surfs = {s.number: s for s in problem.surfaces}
        fns = {}

        def fn(num):
            if num not in fns:
                fns[num] = _surface_fn(surfs[num])
            return fns[num]
    except NotCheckable as e:
        result["reason"] = str(e)
        return result
    rng = np.random.default_rng(seed)
    has_vacuum = any(s.boundary_type == "vacuum" for s in geometry.get_all_surfaces().values())
    dlo, dhi = _domain(geometry, has_vacuum)
    all_s, all_c = geometry.get_all_surfaces(), geometry.get_all_cells()
    errors = []

    def add(msg):
        if len(errors) < 20:
            errors.append(msg)

    def in_cell(found, cid):
        return any(isinstance(x, openmc.Cell) and x.id == cid for x in found)

    for label, f1, sid, cid, fs, segs, cos_bin, sign in faces:
        s = all_s.get(sid)
        if s is None or f1 not in surfs or any(n not in surfs for n, _ in fs):
            add(f"{label}: its surfaces aren't all in the deck and the OpenMC model")
            continue
        try:
            fn(f1)
            for n, _ in fs:
                fn(n)
        except NotCheckable as e:
            add(f"{label}: can't evaluate its surfaces ({e})")
            continue
        lo, hi = dlo, dhi
        if cid is not None and all_c[cid].region is not None:
            clo, chi = (np.array(v, dtype=float) for v in all_c[cid].region.bounding_box)
            pad = np.where(np.isfinite(chi - clo), 0.2 * (chi - clo), 0.0)
            lo, hi = np.maximum(np.nan_to_num(clo - pad, neginf=-np.inf), dlo), np.minimum(np.nan_to_num(chi + pad, posinf=np.inf), dhi)
        starts = np.vstack([rng.uniform(lo, hi, size=(n_points * 2 // 3, 3)), rng.uniform(dlo, dhi, size=(n_points // 3, 3))])
        pts = [(x, nrm) for x, nrm in _on_surface(s, starts) if np.all(x >= dlo - 1e-9) and np.all(x <= dhi + 1e-9)]
        if not pts:
            add(f"{label}: no point on surface {sid} inside the model")
            continue
        X = np.array([p for p, _ in pts])
        v, scale = fn(f1)(X)
        off = np.abs(v) / np.maximum(scale, 1e-12) > 1e-6
        if off.any():
            add(f"{label}: the F1 surface {f1} isn't OpenMC surface {sid} (point {tuple(np.round(X[off][0], 4))} is on "
                f"{sid} but not on {f1})")
            continue
        if cid is None:
            result["checked"] += len(X)
            continue
        senses = []
        for n, sgn in fs:
            val, sc = fn(n)(X)
            senses.append(((val, np.broadcast_to(sc, val.shape)), sgn))  # planes give one scale for all points
        hit = 0
        for i, (x, nrm) in enumerate(pts):
            if any(abs(val[i]) / max(sc[i], 1e-12) < 1e-6 for (val, sc), _ in senses):
                continue  # on a segmenting surface: which segment is ambiguous
            eps = 1e-5
            up, down = in_cell(geometry.find(tuple(x + eps * nrm)), cid), in_cell(geometry.find(tuple(x - eps * nrm)), cid)
            if up and down:
                add(f"{label}: cell {cid} is on both sides of surface {sid} at {tuple(np.round(x, 4))}")
                break
            seg = len(fs) + 1
            for k, ((val, _), sgn) in enumerate(senses):
                if (val[i] < 0) == (sgn < 0):
                    seg = k + 1
                    break
            face = up or down
            if face != (seg in segs):
                add(f"{label}: at {tuple(np.round(x, 4))} OpenMC has cell {cid} {'across' if face else 'not across'} "
                    f"surface {sid}, but the FS card puts the point in segment {seg} (face segments {segs})")
                break
            if face:
                hit += 1
                want = (1, -1) if up else (2, 1)  # cell on the positive side: leaving means moving to the negative side
                if (cos_bin, sign) != want:
                    add(f"{label}: cell {cid} is on the {'positive' if up else 'negative'} side of surface {sid}, so "
                        f"leaving it is cosine bin {want[0]} with sign {want[1]:+d}, not bin {cos_bin} x{sign:+d}")
                    break
            result["checked"] += 1
        if not hit and not any(e.startswith(label) for e in errors):
            add(f"{label}: no sampled point on surface {sid} is next to cell {cid}")
    result["errors"] = errors
    return result


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


def _openmc_path(found):
    """Geometry.find()'s result as an instance path, in Geometry.determine_paths()'s format."""
    parts = []
    for item in found:
        if isinstance(item, tuple):
            lat, idx = item
            parts.append(f"l{lat.id}({','.join(str(int(v)) for v in idx)})")
        elif isinstance(item, openmc.Cell):
            parts.append(f"c{item.id}")
        else:
            parts.append(f"u{item.id}")
    return "->".join(parts)


def chain_text(chain):
    """An MCNP path [(cell, index or None)], top level first, written as a tally bin."""
    levels = [f"{c}[{' '.join(map(str, i))}]" if i is not None else str(c) for c, i in reversed(chain)]
    return levels[0] if len(levels) == 1 else "(" + " < ".join(levels) + ")"


def check_geometry(problem, geometry, n_samples=20000, per_cell=500, seed=12345, chains=None):
    """Return dict(ok, checked_points, errors, skipped_near_surface, reason, unhit_chains, unsampled_cells).

    unsampled_cells lists the material/void cells no sample point landed in. Their geometry was NOT
    compared: OpenMC can't bound a region made of tilted planes or quadrics, so such a cell's "box" is
    the whole domain, and a small cell in a large world can get no points at all. A pass with
    unsampled cells is a pass for the rest of the model only, and callers must say so."""
    result = {"ok": False, "checked_points": 0, "errors": [], "skipped_near_surface": 0, "reason": None,
              "unhit_chains": [], "unsampled_cells": []}
    try:
        deck = _Deck(problem)
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

    paths = [[] for _ in range(len(P))] if chains else None
    try:
        leaf = deck.locate(P, paths=paths)
    except NotCheckable as e:
        result["reason"] = str(e)
        return result
    keep = leaf != NEAR
    result["skipped_near_surface"] = int((~keep).sum())
    P, leaf = P[keep], leaf[keep]
    if chains:
        paths = [tuple(paths[i]) for i in np.flatnonzero(keep)]
        by_chain = {tuple(ch): b for b, (_, ch, _) in enumerate(chains)}
        by_path = {op: b for b, (_, _, op) in enumerate(chains)}
        hits = [0] * len(chains)

    errors = []

    def add(msg):
        if len(errors) < 20:
            errors.append(msg)

    cell_hits = {}
    for pi, (p, n) in enumerate(zip(P, leaf)):
        found = geometry.find(tuple(p))
        omc_cell = found[-1] if found and isinstance(found[-1], openmc.Cell) else None
        if omc_cell is not None:
            cell_hits[omc_cell.id] = cell_hits.get(omc_cell.id, 0) + 1
        where = f"({p[0]:.4g}, {p[1]:.4g}, {p[2]:.4g})"
        if chains:
            bm, bo = by_chain.get(paths[pi]), by_path.get(_openmc_path(found))
            if bm != bo:
                for b in {bm, bo} - {None}:
                    label, ch, op = chains[b]
                    add(f"{label}: point {where} is in " + (f"MCNP bin {chain_text(ch)} but not in OpenMC instance {op}"
                        if b == bm else f"OpenMC instance {op} but not in MCNP bin {chain_text(ch)}")
                        + f" (MCNP path there: {chain_text(paths[pi]) if paths[pi] else 'none'})")
            elif bm is not None:
                hits[bm] += 1
        if n == LOST:
            add(f"point {where} is in no MCNP cell (OpenMC: {'cell ' + str(omc_cell.id) if omc_cell else 'outside'}) -> lost particles")
            continue
        if n == OVERLAP:
            add(f"point {where} is in several MCNP cells of one universe (overlap)")
            continue
        mcnp = deck.cells_by_num[int(n)]
        if omc_cell is None:
            if mcnp.importance.neutron != 0:
                add(f"point {where} is outside the OpenMC model but in MCNP cell {mcnp.number} with IMP:N={mcnp.importance.neutron}")
        elif mcnp.number != omc_cell.id:
            add(f"point {where}: OpenMC cell {omc_cell.id} ({omc_cell.name}) but MCNP cell {mcnp.number}")
    result["checked_points"] = int(len(P))
    result["cell_hits"] = cell_hits
    result["unsampled_cells"] = sorted(
        (cid, c.name) for cid, c in geometry.get_all_cells().items()
        if (c.fill is None or isinstance(c.fill, openmc.Material)) and not cell_hits.get(cid))
    if chains:
        result["unhit_chains"] = [chains[b][0] for b in range(len(chains)) if not hits[b]]

    # materials and densities, per cell
    for cid, cell in geometry.get_all_cells().items():
        fill = cell.fill
        if fill is not None and not isinstance(fill, openmc.Material):
            continue  # filled with a universe or lattice: its cells are compared instead
        mc = deck.cells_by_num.get(cid)
        if mc is None:
            add(f"OpenMC cell {cid} ({cell.name}) has no MCNP cell {cid}")
            continue
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
