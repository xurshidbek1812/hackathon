"""PLACEHOLDER - replace with the official evaluate.py from the starter kit.

Re-implemented from the metric description in the task PDF, for local
development only:
  Score A = macro over classes of mean F1 at temporal IoU {0.3, 0.5, 0.7}
  Score B = 0.4 AP(chance-normalised) + 0.4 F1_alarm + 0.2 mTTA / W
  M       = 0.7 A + 0.3 B
"""
import argparse
import json
import sys

import numpy as np

CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]
TAUS = (0.3, 0.5, 0.7)
H, W, THETA, MERGE_GAP = 5.0, 10.0, 0.5, 2.0


def validate(pred) -> list[str]:
    errors = []
    for vid, v in pred.get("videos", {}).items():
        last = {}
        for ev in sorted(v.get("events", []), key=lambda e: e[0]):
            if len(ev) != 3 or ev[2] not in CLASSES or not float(ev[0]) < float(ev[1]) or float(ev[0]) < 0:
                errors.append(f"{vid}: bad event {ev}")
            elif ev[2] in last and ev[0] < last[ev[2]]:
                errors.append(f"{vid}: same-class overlap {ev}")
            else:
                last[ev[2]] = ev[1]
    return errors


def tiou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_counts(gt, pr, tau):
    pairs = sorted(((tiou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pr)), reverse=True)
    mg, mp, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < tau:
            break
        if i not in mg and j not in mp:
            mg.add(i)
            mp.add(j)
            tp += 1
    return tp, len(pr) - tp, len(gt) - tp


def score_a(pred, gt):
    classes = set()
    for vid, g in gt.items():
        classes |= {e[2] for e in g["events"]}
        classes |= {e[2] for e in pred["videos"].get(vid, {}).get("events", [])}
    table = {}
    for c in sorted(classes):
        f1s = []
        for tau in TAUS:
            TP = FP = FN = 0
            for vid, g in gt.items():
                gs = [e for e in g["events"] if e[2] == c]
                ps = [e for e in pred["videos"].get(vid, {}).get("events", []) if e[2] == c]
                tp, fp, fn = match_counts(gs, ps, tau)
                TP, FP, FN = TP + tp, FP + fp, FN + fn
            f1s.append(2 * TP / (2 * TP + FP + FN) if TP else 0.0)
        table[c] = f1s
    score = float(np.mean([np.mean(v) for v in table.values()])) if table else 0.0
    return score, table


def average_precision(scores, labels) -> float:
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]     # threshold boundaries (ties grouped)
    prec, rec = tp[last] / (tp[last] + fp[last]), tp[last] / max(y.sum(), 1)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def score_b(pred, gt):
    all_s, all_y = [], []
    n_acc = matched_alarms = n_alarms = 0
    tta = []
    for vid, g in gt.items():
        risk = np.asarray(pred["videos"].get(vid, {}).get("risk", []), float).reshape(-1, 2)
        acc = [e for e in g["events"] if e[2] == "accident"]
        nm = [e for e in g["events"] if e[2] == "near_miss"]
        n_acc += len(acc)
        if len(risk) == 0:
            tta += [0.0] * len(acc)
            continue
        t, s = risk[:, 0], risk[:, 1]
        pos = np.zeros(len(t), bool)
        ign = np.zeros(len(t), bool)
        for a0, a1, _ in acc:
            pos |= (t >= a0 - H) & (t < a0)
            ign |= (t >= a0) & (t <= a1)
        for n0, n1, _ in nm:
            ign |= (t >= n0 - H) & (t <= n1)
        pos &= ~ign
        all_s.append(s[~ign])
        all_y.append(pos[~ign].astype(float))
        # alarms
        alarms = []
        on = s >= THETA
        i = 0
        while i < len(t):
            if on[i]:
                j = i
                while j + 1 < len(t) and on[j + 1]:
                    j += 1
                if alarms and t[i] - alarms[-1][1] < MERGE_GAP:
                    alarms[-1][1] = t[j]
                else:
                    alarms.append([t[i], t[j]])
                i = j + 1
            else:
                i += 1
        alarms = [a for a in alarms if not ign[np.searchsorted(t, a[0])]]
        n_alarms += len(alarms)
        unmatched = sorted(a[0] for a in acc)
        got = {}
        for a in alarms:
            for s0 in unmatched:
                if s0 - W <= a[0] < s0 and s0 not in got:
                    got[s0] = a[0]
                    matched_alarms += 1
                    break
        tta += [s0 - got[s0] if s0 in got else 0.0 for s0 in unmatched]
    if n_acc == 0:
        return 0.0, {}
    s, y = np.concatenate(all_s), np.concatenate(all_y)
    r = y.mean() if len(y) else 0.0
    ap_raw = average_precision(s, y) if y.sum() else 0.0
    ap = max(0.0, (ap_raw - r) / (1 - r)) if r < 1 else 0.0
    prec = matched_alarms / n_alarms if n_alarms else 0.0
    rec = len([x for x in tta if x > 0]) / n_acc
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    mtta = float(np.mean(tta)) if tta else 0.0
    return 0.4 * ap + 0.4 * f1 + 0.2 * mtta / W, {"AP": ap, "F1_alarm": f1, "mTTA": mtta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    pred = json.load(open(args.pred, encoding="utf-8"))
    errors = validate(pred)
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("format OK")
    if args.validate_only or not args.gt:
        return
    gt = json.load(open(args.gt, encoding="utf-8"))
    gt = gt.get("videos", gt)
    a, table = score_a(pred, gt)
    b, parts = score_b(pred, gt)
    print(f"{'class':22s} F1@0.3  F1@0.5  F1@0.7")
    for c, f in table.items():
        print(f"{c:22s} {f[0]:.3f}   {f[1]:.3f}   {f[2]:.3f}")
    print(f"Score A = {a:.4f}")
    print(f"Score B = {b:.4f}  {parts}")
    print(f"Model score M = {0.7 * a + 0.3 * b:.4f}")


if __name__ == "__main__":
    main()
