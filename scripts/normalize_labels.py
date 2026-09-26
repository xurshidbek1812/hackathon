"""Normalise hand-made labels to the organisers' conventions.

    python scripts/normalize_labels.py dev_labels/C3896_team_labels.json dev_labels/ground_truth.json

Same-class events that overlap become one segment covering both (task FAQ:
"two events of the same class at once -> one segment covering both").
"""
import json
import sys


def normalize(gt: dict) -> dict:
    out = {}
    for vid, v in gt.items():
        by_class: dict[str, list] = {}
        for s, e, label in sorted(v["events"]):
            segs = by_class.setdefault(label, [])
            if segs and s < segs[-1][1]:
                segs[-1][1] = max(segs[-1][1], e)
            else:
                segs.append([s, e])
        events = sorted([s, e, label] for label, segs in by_class.items() for s, e in segs)
        out[vid] = {**v, "events": events}
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    json.dump(normalize(json.load(open(src))), open(dst, "w"), indent=1)
