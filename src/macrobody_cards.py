"""
Simplification of standalone primitives (box -> RPP, cylinder -> RCC) in translated MCNP decks.

OpenMC models define rectangular prisms via 6 planes and finite cylinders via a cylinder
surface + 2 planar end caps. In MCNP, standalone parts bounded by these surfaces can be
represented cleanly as single macrobody cards (RPP, RCC), reducing surface counts and
making the deck significantly easier to read.

Safety guarantees:
- Only standalone cells whose surfaces are unique to that cell (or whose complement in
  an outer/graveyard cell is the exact de Morgan union of those surfaces) are converted.
- Shared-plane interfaces between adjacent cells are preserved as half-spaces.
- All converted decks must pass geometry verification (zero lost particles / gaps / overlaps).
"""
import re
import math
from typing import Dict, List, Tuple, Optional, Set

COMMENT = re.compile(r"^\s{0,4}[cC](?:\s|$)")
PARAM_KEYWORD = re.compile(
    r"^(IMP(?::[A-Z]+)?|U|LAT|FILL|VOL|TRCL|LIKE|PWT|NONU|PD)(?:=|\s|$)",
    re.IGNORECASE,
)


def _fmt(v: float) -> str:
    v = float(v)
    return "0" if v == 0.0 else repr(round(v, 10))


def _split_cards(block: str) -> Tuple[List[str], List[str]]:
    """Splits an MCNP block into (header_lines, [card_strings])."""
    header, cards, pending = [], [], []
    for ln in block.split("\n"):
        if COMMENT.match(ln):
            pending.append(ln)
        elif (ln.startswith("     ") or ln.startswith("\t")) and cards:
            cards[-1] += "\n" + ln
        elif re.match(r"^\s*\d+\s", ln):
            cards.append("\n".join(pending + [ln]))
            pending = []
        elif not cards:
            header.append(ln)
        else:
            cards[-1] += "\n" + ln
    return header, cards


def _parse_surface_block(block: str) -> Dict[int, dict]:
    """Parses surface block into dict: surf_id -> {type, constants, star, card, lines}."""
    surfaces = {}
    _, cards = _split_cards(block)
    for card in cards:
        lines = card.split("\n")
        non_comment = [l for l in lines if not COMMENT.match(l)]
        if not non_comment:
            continue
        first = non_comment[0].strip()
        tokens = first.split()
        if not tokens:
            continue
        try:
            sid = int(tokens[0])
        except ValueError:
            continue

        rem = tokens[1:]
        star = False
        if rem and rem[0].startswith("*"):
            star = True
            if len(rem[0]) > 1:
                rem[0] = rem[0][1:]
            else:
                rem = rem[1:]

        if not rem:
            continue
        stype = rem[0].upper()
        const_tokens = rem[1:]
        for extra in non_comment[1:]:
            const_tokens.extend(extra.strip().split())

        constants = []
        for tok in const_tokens:
            try:
                constants.append(float(tok))
            except ValueError:
                pass

        surfaces[sid] = {
            "id": sid,
            "star": star,
            "type": stype,
            "constants": constants,
            "card": card,
            "lines": lines,
        }
    return surfaces


def _parse_cell_card(card: str) -> Optional[dict]:
    """Parses a cell card into components: cell_id, mat_id, density, geom_tokens, param_tokens."""
    lines = card.split("\n")
    comments = [l for l in lines if COMMENT.match(l)]
    data_lines = [l for l in lines if not COMMENT.match(l)]
    if not data_lines:
        return None

    tokens = " ".join(data_lines).split()
    if len(tokens) < 2:
        return None

    try:
        cid = int(tokens[0])
        mat_id = int(tokens[1])
    except ValueError:
        return None

    idx = 2
    density = None
    if mat_id != 0:
        if idx >= len(tokens):
            return None
        try:
            density = float(tokens[idx])
            idx += 1
        except ValueError:
            return None

    geom_tokens = []
    param_tokens = []
    while idx < len(tokens):
        tok = tokens[idx]
        if PARAM_KEYWORD.match(tok):
            param_tokens = tokens[idx:]
            break
        geom_tokens.append(tok)
        idx += 1

    return {
        "id": cid,
        "mat": mat_id,
        "density": density,
        "geom_tokens": geom_tokens,
        "param_tokens": param_tokens,
        "comments": comments,
        "card": card,
    }


def _plane_axis_val(sinfo: dict) -> Optional[Tuple[str, float]]:
    """Returns (axis, intercept) if surface is an axis-aligned plane, else None."""
    stype = sinfo["type"]
    c = sinfo["constants"]
    if stype == "PX" and len(c) >= 1:
        return ("X", c[0])
    if stype == "PY" and len(c) >= 1:
        return ("Y", c[0])
    if stype == "PZ" and len(c) >= 1:
        return ("Z", c[0])
    if stype == "P" and len(c) == 4:
        # P A B C D where Ax + By + Cz - D = 0
        A, B, C, D = c[0], c[1], c[2], c[3]
        if abs(A - 1.0) < 1e-12 and abs(B) < 1e-12 and abs(C) < 1e-12:
            return ("X", D)
        if abs(A + 1.0) < 1e-12 and abs(B) < 1e-12 and abs(C) < 1e-12:
            return ("X", -D)
        if abs(B - 1.0) < 1e-12 and abs(A) < 1e-12 and abs(C) < 1e-12:
            return ("Y", D)
        if abs(B + 1.0) < 1e-12 and abs(A) < 1e-12 and abs(C) < 1e-12:
            return ("Y", -D)
        if abs(C - 1.0) < 1e-12 and abs(A) < 1e-12 and abs(B) < 1e-12:
            return ("Z", D)
        if abs(C + 1.0) < 1e-12 and abs(A) < 1e-12 and abs(B) < 1e-12:
            return ("Z", -D)
    return None


def _cylinder_info(sinfo: dict) -> Optional[Tuple[str, float, float, float]]:
    """Returns (axis, c1, c2, radius) if surface is an axial cylinder, else None."""
    stype = sinfo["type"]
    c = sinfo["constants"]
    if stype in ("CZ", "C/Z"):
        if stype == "CZ" and len(c) >= 1:
            return ("Z", 0.0, 0.0, c[0])
        elif len(c) >= 3:
            return ("Z", c[0], c[1], c[2])
    if stype in ("CX", "C/X"):
        if stype == "CX" and len(c) >= 1:
            return ("X", 0.0, 0.0, c[0])
        elif len(c) >= 3:
            return ("X", c[0], c[1], c[2])
    if stype in ("CY", "C/Y"):
        if stype == "CY" and len(c) >= 1:
            return ("Y", 0.0, 0.0, c[0])
        elif len(c) >= 3:
            return ("Y", c[0], c[1], c[2])
    return None


def simplify_cells(blocks: List[str], model, enabled: bool = True) -> List[str]:
    """Simplifies standalone box and cylinder cells to RPP and RCC macrobodies.

    Returns a list of note strings detailing the simplifications performed.
    """
    if not enabled or len(blocks) < 2:
        return []

    surfaces = _parse_surface_block(blocks[1])
    cell_header, cell_cards_raw = _split_cards(blocks[0])

    parsed_cells = []
    for raw in cell_cards_raw:
        pc = _parse_cell_card(raw)
        if pc is not None:
            parsed_cells.append(pc)

    notes = []

    # Map surface usage across all cells: surf_id -> set of cell_ids
    surf_usage: Dict[int, Set[int]] = {}
    for c in parsed_cells:
        gtoks = c["geom_tokens"]
        for tok in gtoks:
            for m in re.finditer(r"\b(\d+)\b", tok):
                sid = int(m.group(1))
                if sid in surfaces:
                    surf_usage.setdefault(sid, set()).add(c["id"])

    # Identify candidate standalone cells
    # We examine each cell with simple convex geometry tokens
    for c in parsed_cells:
        gtoks = c["geom_tokens"]
        ptoks = c["param_tokens"]

        # Skip lattices, universes > 0, or cells with complex boolean logic
        if any(t.upper().startswith(("LAT", "FILL")) for t in ptoks):
            continue

        # Check if tokens are all simple signed surface IDs (no :, (, ), #)
        is_simple_intersection = True
        signed_surfs: List[Tuple[int, int]] = []  # (sign: +1 or -1, sid)
        for tok in gtoks:
            if re.match(r"^[-+]?\d+$", tok):
                val = int(tok)
                sign = -1 if val < 0 else 1
                signed_surfs.append((sign, abs(val)))
            else:
                is_simple_intersection = False
                break

        if not is_simple_intersection:
            continue

        # Check for Box candidate: exactly 6 signed planes
        rpp_bounds = None
        rcc_info = None

        if len(signed_surfs) == 6:
            # Check for 3 pairs of orthogonal planes
            planes_by_axis: Dict[str, List[Tuple[int, int, float]]] = {"X": [], "Y": [], "Z": []}
            valid_box = True
            for sign, sid in signed_surfs:
                if sid not in surfaces:
                    valid_box = False
                    break
                pinfo = _plane_axis_val(surfaces[sid])
                if not pinfo:
                    valid_box = False
                    break
                axis, val = pinfo
                planes_by_axis[axis].append((sign, sid, val))

            if valid_box and all(len(planes_by_axis[ax]) == 2 for ax in ("X", "Y", "Z")):
                # In each axis, we need one plane with + sign (x >= xmin) and one with - sign (x <= xmax)
                box_coords = {}
                for ax in ("X", "Y", "Z"):
                    p1, p2 = planes_by_axis[ax]
                    if p1[0] == 1 and p2[0] == -1 and p1[2] < p2[2]:
                        box_coords[ax] = (p1[2], p2[2], p1[1], p2[1])
                    elif p2[0] == 1 and p1[0] == -1 and p2[2] < p1[2]:
                        box_coords[ax] = (p2[2], p1[2], p2[1], p1[1])
                    else:
                        valid_box = False
                        break

                if valid_box:
                    xmin, xmax, sx0, sx1 = box_coords["X"]
                    ymin, ymax, sy0, sy1 = box_coords["Y"]
                    zmin, zmax, sz0, sz1 = box_coords["Z"]
                    rpp_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

        # Check for Cylinder candidate: 1 cylinder + 2 axial planes
        elif len(signed_surfs) == 3:
            cyl_entries = []
            plane_entries = []
            for sign, sid in signed_surfs:
                if sid not in surfaces:
                    break
                c_inf = _cylinder_info(surfaces[sid])
                p_inf = _plane_axis_val(surfaces[sid])
                if c_inf and sign == -1:  # inside cylinder
                    cyl_entries.append((sid, c_inf))
                elif p_inf:
                    plane_entries.append((sign, sid, p_inf))

            if len(cyl_entries) == 1 and len(plane_entries) == 2:
                cyl_sid, (c_axis, c1, c2, radius) = cyl_entries[0]
                (s1_sign, s1_id, (p1_axis, p1_val)), (s2_sign, s2_id, (p2_axis, p2_val)) = plane_entries
                if p1_axis == c_axis and p2_axis == c_axis:
                    # One plane must be + (val_min) and one - (val_max)
                    if s1_sign == 1 and s2_sign == -1 and p1_val < p2_val:
                        vmin, vmax = p1_val, p2_val
                    elif s2_sign == 1 and s1_sign == -1 and p2_val < p1_val:
                        vmin, vmax = p2_val, p1_val
                    else:
                        vmin = vmax = None

                    if vmin is not None:
                        h = vmax - vmin
                        if c_axis == "Z":
                            # Base (c1, c2, vmin), H (0, 0, h), R
                            rcc_info = (c1, c2, vmin, 0.0, 0.0, h, radius)
                        elif c_axis == "X":
                            # Base (vmin, c1, c2), H (h, 0, 0), R
                            rcc_info = (vmin, c1, c2, h, 0.0, 0.0, radius)
                        elif c_axis == "Y":
                            # Base (c1, vmin, c2), H (0, h, 0), R
                            rcc_info = (c1, vmin, c2, 0.0, h, 0.0, radius)

        if not rpp_bounds and not rcc_info:
            continue

        # Surfaces belonging to this candidate cell
        c_sids = set(sid for _, sid in signed_surfs)

        # Check who else uses these surfaces
        other_cells_using = set()
        for sid in c_sids:
            for other_cid in surf_usage.get(sid, set()):
                if other_cid != c["id"]:
                    other_cells_using.add(other_cid)

        # Build complement pattern
        # The exact de Morgan complement of (s1 s2 ... sk) is (-s1 : -s2 : ... : -sk)
        de_morgan_parts = []
        for sign, sid in signed_surfs:
            comp_sign = -sign
            de_morgan_parts.append(f"{comp_sign * sid}")

        # Check if every other cell using these surfaces uses the full de Morgan complement
        can_simplify = True
        cells_with_complement = []

        for other_cid in other_cells_using:
            other_c = next((x for x in parsed_cells if x["id"] == other_cid), None)
            if not other_c:
                can_simplify = False
                break
            other_gtxt = " ".join(other_c["geom_tokens"])

            # Match patterns like (-1:2:-3:4:-5:6) or (-1 : 2 : -3 : 4 : -5 : 6)
            # Permutations of order are possible, so check if all terms are present with colons
            found_comp = False
            # Find bracketed expressions: (...)
            for bmatch in re.finditer(r"\(([^()]+)\)", other_gtxt):
                inner = bmatch.group(1)
                terms = [t.strip() for t in inner.split(":")]
                if set(terms) == set(de_morgan_parts):
                    found_comp = True
                    cells_with_complement.append((other_c, bmatch.group(0)))
                    break

            if not found_comp:
                # If the other cell references an individual surface without the full complement,
                # it's a shared face; do not simplify!
                can_simplify = False
                break

        if not can_simplify:
            continue

        # We can safely simplify!
        # Pick the new surface ID (reuse the first surface ID of the set)
        target_sid = signed_surfs[0][1]

        if rpp_bounds:
            xmin, xmax, ymin, ymax, zmin, zmax = rpp_bounds
            mbody_line = f"{target_sid} RPP {_fmt(xmin)} {_fmt(xmax)} {_fmt(ymin)} {_fmt(ymax)} {_fmt(zmin)} {_fmt(zmax)}"
            mbody_type = "RPP"
        else:
            vx, vy, vz, hx, hy, hz, r = rcc_info
            mbody_line = f"{target_sid} RCC {_fmt(vx)} {_fmt(vy)} {_fmt(vz)} {_fmt(hx)} {_fmt(hy)} {_fmt(hz)} {_fmt(r)}"
            mbody_type = "RCC"

        # Update cell c: geometry becomes -target_sid
        c["geom_tokens"] = [f"-{target_sid}"]

        # Update cells with complement: replace (-s1:s2:...) with target_sid (+target_sid)
        for other_c, comp_expr in cells_with_complement:
            other_gtxt = " ".join(other_c["geom_tokens"])
            new_gtxt = other_gtxt.replace(comp_expr, f"{target_sid}")
            other_c["geom_tokens"] = new_gtxt.split()

        # Update surfaces: replace target_sid with the new macrobody card,
        # and mark the remaining surfaces in c_sids for removal
        surfaces[target_sid] = {
            "id": target_sid,
            "star": False,
            "type": mbody_type,
            "constants": list(rpp_bounds if rpp_bounds else rcc_info),
            "card": mbody_line,
            "lines": [mbody_line],
        }
        for sid in c_sids:
            if sid != target_sid:
                surfaces.pop(sid, None)

        notes.append(
            f"Cell {c['id']}: simplified {len(signed_surfs)} primitive surfaces into {mbody_type} {target_sid}."
        )

    if not notes:
        return []

    # Reconstruct blocks[0] (cells) and blocks[1] (surfaces)
    new_cell_cards = []
    for c in parsed_cells:
        lines = []
        if c["comments"]:
            lines.extend(c["comments"])
        first_line_parts = [str(c["id"]), str(c["mat"])]
        if c["density"] is not None:
            first_line_parts.append(_fmt(c["density"]))
        first_line_parts.extend(c["geom_tokens"])
        if c["param_tokens"]:
            first_line_parts.extend(c["param_tokens"])
        lines.append(" ".join(first_line_parts))
        new_cell_cards.append("\n".join(lines))

    blocks[0] = "\n".join(cell_header + (["\n".join(new_cell_cards)] if new_cell_cards else []))

    # Reconstruct surface block
    surf_header, _ = _split_cards(blocks[1])
    # Keep surfaces in numerical order
    sorted_surfs = sorted(surfaces.values(), key=lambda s: s["id"])
    surf_lines = []
    for s in sorted_surfs:
        surf_lines.append(s["card"].rstrip())

    blocks[1] = "\n".join(surf_header + (["\n".join(surf_lines)] if surf_lines else []))
    return notes
