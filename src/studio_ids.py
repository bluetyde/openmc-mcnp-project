"""Optional Studio provenance, encoded entirely as MCNP comments (schema v1).

Every record addresses a card explicitly, so comments between continuation lines
cannot change ownership. Payloads are percent-encoded JSON split into short ASCII
chunks. Ordinary c/$ comments and all physical data lines remain untouched.
"""
import json
import re
from urllib.parse import quote


def comment_cards(kind, number, owner):
    payload = quote(json.dumps(owner, ensure_ascii=True, separators=(',', ':')), safe='')
    chunks = [payload[i:i + 64] for i in range(0, len(payload), 64)]
    return [f'c @studio-v1 {kind} {number} {i + 1}/{len(chunks)} {chunk}'
            for i, chunk in enumerate(chunks)]


def annotate(text, ids, lattice_maps=None, tally_map=None, graveyard=None):
    if ids is None:
        return text
    maps = {kind: dict(ids.get(kind, {})) for kind in ('cell', 'surface', 'material')}
    for lid, layout in (lattice_maps or {}).items():
        owner = ids.get('lattice', {}).get(lid)
        if owner:
            maps['cell'][layout['cell']] = owner
            for sid in layout.get('surfaces', []):
                maps['surface'][sid] = owner
    if graveyard is not None:
        maps['cell'][graveyard] = {'kind': 'world', 'id': 'world'}
    maps['tally'] = {n: ids.get('tally', {}).get(tid) for n, tid in (tally_map or {}).items()}
    maps['data'] = {}
    lines = text.splitlines()
    out = lines[:1] + ['c @studio-map-v1']
    block = 0
    for line in lines[1:]:
        if not line.strip():
            block += 1
        elif not re.match(r'^ {0,4}[cC](?:\s|$)|^\s*\$', line) and not line.startswith('     '):
            tokens = line.split('$', 1)[0].split()
            head = tokens[0].upper() if tokens else ''
            kind, number = None, None
            if block < 2 and re.fullmatch(r'[*+]?\d+', head):
                kind, number = ('cell' if block == 0 else 'surface'), int(head.lstrip('*+'))
            elif block >= 2:
                match = re.fullmatch(r'M(\d+)', head)
                tally = re.fullmatch(r'(?:F|FMESH)(\d+)(?::[A-Z,]+)?', head)
                if match:
                    kind, number = 'material', int(match[1])
                elif tally:
                    kind, number = 'tally', int(tally[1])
                elif head in {'MODE', 'NPS', 'KCODE', 'NONU'}:
                    kind, number = 'data', head
                    maps['data'][head] = {'kind': 'settings', 'id': 'settings'}
                elif re.fullmatch(r'SDEF|KSRC|S[IP]\d+|DS\d+', head) and len(ids.get('source', {})) == 1:
                    kind, number = 'data', head
                    maps['data'][head] = next(iter(ids['source'].values()))
            owner = maps.get(kind, {}).get(number)
            if owner:
                owner = dict(owner)
                # Derived surfaces do not have a directly editable OpenMC primitive.
                if kind == 'surface' and len(tokens) > 1 and tokens[1].upper() in {'RPP', 'RCC', 'BOX', 'HEX'}:
                    owner.pop('slot', None)
                out.extend(comment_cards(kind, number, owner))
        out.append(line)
    return '\n'.join(out) + ('\n' if text.endswith('\n') else '')
