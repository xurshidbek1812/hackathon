# Team Infinity — WIUT Hackathon 2026, Computer Vision track

Traffic event detection (Part A) and causal accident anticipation (Part B) for one fixed CCTV road camera.

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
```

No internet is needed at run time: the detector weights ship in `weights/yolo11m.pt`
(`weights/download.sh` restores them if missing and checks the SHA-256).
`run_submission.py`, `evaluate.py` and `examples/` are the organizers' starter-kit files, unchanged.

**Runtime** on the 340 s, 3840×2160 @ 29.97 fps sample video, measured with the official harness on a
laptop GTX 1650 Ti (fp32; several times slower than the T4 used for evaluation):
Part A 246 s + Part B 303 s = **549 s of a 1021 s budget**. Decoding 4K H.264 is the bottleneck, so
decoding runs in a reader thread that overlaps with inference (Part A), and Part B's detector runs in a
worker thread with a fixed one-sample lag (see *Determinism*).

---

## Repository layout

```
solution.py                  competition interface: detect_events(), RiskEstimator
run_submission.py            starter-kit harness (unchanged)
evaluate.py                  starter-kit metric (unchanged)
requirements.txt             inference dependencies (requirements-dev.txt adds demo + tests)
weights/                     yolo11m.pt, download.sh, SHA256SUMS
configs/scene.json           scene layout drawn from the sample view (normalised coordinates)
configs/flow_prior.npz       lane directions + queue zones learned from the sample videos
src/traffic_events/
  detector.py                YOLO wrapper, rider / vehicle-occupant suppression
  tracker.py                 ByteTrack-style tracker (ours, deterministic)
  tracks.py                  smoothing, body-unit kinematics, parked-vehicle re-linking
  flow.py                    learned flow field (lane direction, road mask, queue zones)
  scene.py                   scene config -> pixel geometry
  signal_state.py            traffic-signal colour from the signal-head region
  frame_features.py          background model: static blobs (obstacles), smoke, fire
  rules/                     one module per event family (motion, signal, pedestrian, collision, hazards)
  segments.py                frame flags -> segments, merging, same-class overlap handling
  pipeline.py                Part A orchestration + time budget
  risk.py                    Part B causal risk model
  export.py                  JSON export for the website and the demo
scripts/
  analyze_videos.py          run everything on the samples, export website data, build flow prior
  eda.py                     EDA figures and statistics for the website
  render_annotated.py        annotated MP4s for the website
demo/server.py               FastAPI live demo (serves website/ + upload API)
website/                     static team website (GitHub Pages ready) + tools/labeler.html, tools/scene_editor.html
tests/                       unit tests: every rule on synthetic trajectories, tracker, metric
```

## Approach

| Stage | What | Learned or rule-based |
|---|---|---|
| Detection | YOLO11-m, COCO classes car/bus/truck/motorcycle/bicycle/person/animals/loose objects, 960 px, every 2nd frame (12.5 Hz), batched | learned (pre-trained, not fine-tuned) |
| Tracking | ByteTrack-style two-stage IoU matching with constant-velocity prediction on timestamps | rule-based |
| Trajectories | bottom-centre ground point, 0.6 s centred smoothing, speeds in **body units/s** (px ÷ √(w·h)) | rule-based |
| Scene | lanes, crossings, stop lines, solid lines, signal head from `camera.md` → `configs/scene.json`; lane directions, road mask and queue zones learned from sample trajectories → `configs/flow_prior.npz` | hand-coded scene facts + statistics |
| Events | one rule per class (see table), segments post-processed for IoU 0.7 | rule-based |
| Part B | own causal tracker; logistic score of TTC, hard braking, swerve, wrong-way, pedestrian on road; fast rise, slow decay | rule-based, hand-set weights |

| Class | Rule |
|---|---|
| accident | boxes overlap with ground points at the same depth, after closing ≥ 0.8 body/s, plus hard braking / swerve / both staying stopped / pedestrian fall |
| near_miss | TTC < 1.2 s and predicted miss < 1 body, with hard braking or swerve, no contact |
| red_light | front of vehicle crosses a stop line in its approach direction while the signal reads red |
| wrong_way | ≥ 1.5 s and ≥ 2 body lengths against the lane direction (config lanes or learned flow) |
| illegal_u_turn | heading reverses ≥ 150° within 20 s, outside U-turn-allowed zones |
| stopped_vehicle | stationary ≥ 10 s on the road, not in a learned queue zone, not released together with a queue |
| jaywalking | pedestrian feet inside the carriageway (with margin), outside every crossing, ≥ 1 s |
| failure_to_yield | vehicle drives through a crossing while a pedestrian is on it |
| illegal_turn | entry zone → exit zone listed in `prohibited_turns` |
| solid_line_crossing | box bottom corners cross a solid polyline and stay across ≥ 0.8 s |
| stop_line | vehicle stops just past the stop line on red; ends when the signal turns green |
| congestion | ≥ 4 vehicles, ≥ 75 % crawling, per carriageway, ≥ 30 s |
| road_obstacle | detected animal / static loose object on the road, or a static foreground blob nobody detected |
| fire_smoke | flickering fire-coloured pixels (static lamps removed), or a grey, smooth, growing foreground blob |

Rules whose scene information is missing switch themselves off: a class we predict that never occurs in the
test set costs a full class of macro-F1.

### Scene configuration

The starter kit arrived without `camera.md`, so `configs/scene.json` was drawn by hand from sample frames
(with `website/tools/scene_editor.html`): the carriageway outline, the three pedestrian crossings, the
pedestrian islands and median (excluded), the main approach's stop line, and the junction box. The signal
heads of the main approach face away from the camera, so that stop line has no signal region: its phase is
**inferred from the queue** (`rules/signal.py`, `QueuePhase`) — red when at least two other vehicles have been
standing at the line for 5 s and no other vehicle has crossed it in the last 4 s. Lane directions, the road
mask and signal-queue zones are learned from the sample trajectories (`configs/flow_prior.npz`).

### Calibration on the sample video

With no labels, every rule was checked by rendering its candidate events (tracks drawn on the frame) and
inspecting them. The first full run produced 86 events on 340 s of ordinary traffic; the fixes that brought it
to 13 plausible events were:

| Problem seen on the real video | Fix |
|---|---|
| NaN detections on GTX 16xx in fp16 | fp16 off on those cards; non-finite boxes dropped |
| identity switches in dense queues (fake U-turns, near misses) | IoU gate 0.3, split tracks at impossible jumps, re-link parked fragments only within 8 s |
| boxes cut by the frame edge fake a heading reversal | observations touching the border are dropped |
| platoons and queue-joining counted as near misses | near miss needs crossing paths (45–135° between vehicles, 30–150° with a pedestrian), both moving, danger held ≥ 0.3 s |
| far, tiny vehicles overlap through perspective | collision rules ignore road users smaller than 2.5 % of frame height |
| turn-lane queue fooled the inferred red phase | red needs ≥ 2 vehicles waiting ≥ 5 s and no traffic flowing through |
| pedestrians on the kerb next to crossings | margins scale with person height; failure-to-yield needs the pedestrian on the carriageway and within 3 vehicle lengths |
| legal junction turns flagged as wrong-way | wrong-way only judged outside the junction box, on cells with ≥ 80 % consistent direction |

The Part B cues were recalibrated the same way; on the sample (no accidents) the risk now stays below 0.2.

## Reproducing the results

```bash
pip install -r requirements-dev.txt
python scripts/analyze_videos.py --videos samples --build-prior    # website data + configs/flow_prior.npz
python scripts/eda.py --videos samples                              # EDA figures
python scripts/render_annotated.py samples/*.mp4                    # annotated videos for the website
python run_submission.py --videos samples --out predictions_samples.json
python evaluate.py --pred predictions_samples.json --gt dev_labels.json   # our own dev labels
pytest -q
```

Live demo locally: `uvicorn demo.server:app --port 7860`, then open http://localhost:7860.

## Determinism

Seeds are fixed (`random`, `numpy`, `torch`), cuDNN benchmarking is off, the tracker and rules are
deterministic. Part B's worker thread does not make the output timing-dependent: the detection submitted at
sample *k* is always collected at sample *k+1*, so the score lags by exactly one sample (0.1 s) on every run. The only wall-clock-dependent behaviour is an emergency frame skip that triggers when a machine
runs at more than **twice** the time budget; on the target GPU it never triggers.

## Datasets and models

| Asset | Use | Licence |
|---|---|---|
| YOLO11-m (Ultralytics), pre-trained on COCO | detector, used as is | AGPL-3.0 |
| COCO 2017 (via the pre-trained weights) | — | CC BY 4.0 |
| Competition sample videos | EDA, flow prior, dev labels | competition use |

No other external data is used. No hosted/paid model is called at any point.

## Open-source code

- Ultralytics YOLO (AGPL-3.0) — detector runtime.
- The tracker follows the ByteTrack algorithm (Zhang et al., ECCV 2022); the implementation in `src/traffic_events/tracker.py` is ours.

## Team

Team **Infinity**

| Member | Role | Contributions |
|---|---|---|
| Suxrob Muminov ([GitHub](https://github.com/suxrobmuminov2007-creator), [LinkedIn](https://www.linkedin.com/in/suxrob-muminov-b3a8a0395/)) | Team leader | managed the team, coordinated the work and the submission |
| Xurshidbek Bekchonov ([GitHub](https://github.com/xurshidbek1812), [LinkedIn](https://www.linkedin.com/in/xurshidbek-bekchonov-98461235b/)) | Website developer | team website, live demo page and upload flow, results / EDA / dashboard views |
| Nodirbek Ro'ziqulov ([GitHub](https://github.com/quixtorm), [LinkedIn](https://www.linkedin.com/in/nadir-ruz-a8898443a/)) | Computer vision & analysis | detection and tracking pipeline, event rules and their calibration, EDA, accident-risk model (Part B), evaluation and time budget |
