"""Turn the campaign logs into rows.

Each step in a campaign log is a `########` header followed by whatever the
harness printed, which for a measurement is one JSON object. Everything else in
the log -- notices, tracebacks, progress lines -- is skipped, so a step that
failed simply yields no row rather than a plausible-looking wrong one.
"""

import json
import re
import sys
from pathlib import Path

RESULTS = Path('./results')
HEADER = re.compile(r'^######## (\S+) \[([^\]]+)\] (.*)$')


def steps(path: Path):
    header, buf = None, []
    for line in path.read_text(errors='replace').splitlines():
        m = HEADER.match(line)
        if m:
            if header:
                yield header, buf
            header, buf = (m.group(1), m.group(2), m.group(3).strip()), []
        elif header:
            buf.append(line)
    if header:
        yield header, buf


def json_blocks(lines):
    """Every top-level JSON object printed by a step."""
    out, depth, start = [], 0, None
    for i, line in enumerate(lines):
        if depth == 0 and line.rstrip() == '{':
            start, depth = i, 1
            continue
        if depth:
            depth += line.count('{') - line.count('}')
            if depth == 0:
                try:
                    out.append(json.loads('\n'.join(lines[start : i + 1])))
                except json.JSONDecodeError:
                    pass
    return out


def load(campaign: str):
    rows = []
    for (when, tag, what), lines in steps(RESULTS / f'{campaign}.log'):
        for obj in json_blocks(lines):
            rows.append({'when': when, 'campaign': tag, 'step': what, **obj})
    return rows


if __name__ == '__main__':
    for campaign in sys.argv[1:] or [p.stem for p in sorted(RESULTS.glob('*.log'))]:
        rows = load(campaign)
        kinds = {}
        for r in rows:
            kind = (
                'measure'
                if 'recall_at_k' in r and 'target' in r
                else 'needle'
                if 'exact_top1' in r
                else 'multikb'
                if 'by_collection_count' in r
                else 'sizes'
                if 'vector_index_mb' in r and 'target' not in r
                else 'index'
                if 'partitions_indexed' in r
                else 'churn'
                if 'delete_collection_seconds' in r
                else 'vacuum'
                if 'rows_deleted' in r
                else 'other'
            )
            kinds.setdefault(kind, []).append(r)
        print(f'{campaign}: {len(rows)} objets -> ' + ', '.join(f'{k}={len(v)}' for k, v in sorted(kinds.items())))
