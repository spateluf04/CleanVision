# RoomScan Live Dashboard — Test Plan & Validation Checklist

Covers `roomscan_dashboard.py`, `roomscan_live.py` (`LiveScanController`), and
the shared pipeline they drive (`aria_capture.py` -> `energy_detector.py` ->
`energy_estimator.py` -> `roomscan.py:finalize_scan()` -> `energy_sessions.py`).

Run automated checks from the repo root with the project venv active:

```bash
source ~/aria-venv/bin/activate
python -m pytest tests/ -q
```

All commands below assume that has been done once already.

---

## 0. Isolating "camera vs. detector" with `--debug-camera-only`

If the live feed appears broken, `--debug-camera-only` rules out the
detector as the cause in one run: it skips constructing `EnergyDetector`
entirely (no YOLO/ultralytics load at all) and skips subscribing the
camera-rgb detect/aggregate callback, while `AriaCapture`'s frame delivery
and the dashboard's frame rendering are completely untouched (they never
depended on a detection subscriber existing).

```bash
python roomscan_dashboard.py --debug-camera-only --start-streaming --device-ip <glasses-ip> --interface usb --profile profile18
# or, for the standalone manual-test CLI:
python roomscan_live.py --debug-camera-only --start-streaming --device-ip <glasses-ip> --interface usb
```

- Window title gains a `[DEBUG: camera-only, detector disabled]` suffix.
- Once frames are confirmed flowing, the status pill reads `DEBUG: camera
  OK, detector disabled` instead of `Live feed active` / `no appliances
  detected yet` — this status only appears after `_poll_frame()` has
  already confirmed a real frame arrived, so seeing it is itself
  confirmation the camera pipeline is healthy.
- The Detected Devices table stays on the empty-state placeholder and
  totals read `0 W`/`0 kWh`/`$0` for the whole scan; Save Report still
  works and writes a zero-device report (expected — nothing ran detection).

Reading the result:
- Feed still doesn't appear with this flag on -> the bug is upstream of
  detection (`AriaCapture` not delivering frames, or `LiveScanController`/
  dashboard not exposing or rendering them).
- Feed appears with this flag on but not with it off -> the bug is in
  `EnergyDetector`/YOLO (failing to load, hanging, or throwing) blocking
  the pipeline.

Turn the flag back off (drop `--debug-camera-only`) to re-enable detector
overlays once the camera side is confirmed working.

Automated coverage: `tests/test_roomscan_e2e_live.py::test_debug_camera_only_shows_frames_with_detection_disabled`
drives this mode against the real `LiveScanController`/`AriaCapture`,
asserting `disable_detection`/`_detector is None`, that frames still
render, and that the device table/totals stay in their empty state.

## 1. Testing with a live Aria stream

Requires paired glasses and `aria streaming install-certs` already run once.

```bash
aria streaming stop && aria recording stop
python roomscan_dashboard.py --start-streaming --device-ip <glasses-ip> --interface usb --profile profile18
```

(`--interface wifi` also works without `--device-ip` if the glasses are
already streaming to this host.)

Checklist:
- [ ] Window opens with "Idle" status and an empty camera feed placeholder.
- [ ] Enter a room name, click **Start Scan** — status changes to
      `Scanning: <room>`, **Start** disables, **Stop**/**Save** enable.
- [ ] Live RGB feed appears in the center panel within ~1–2s, upright
      (not rotated 90°) and not mirrored.
- [ ] Point the glasses at a real TV/laptop/microwave — within a few
      seconds it appears in the **Detected Devices** table with a
      plausible confidence (>0.5).
- [ ] Energy Summary panel's watts/kWh/cost values move away from `--`
      and update roughly once per second (`ROOMSCAN_LIVE_TICK_SECONDS`).
- [ ] Pan away from a detected device and back — count does not double
      (max-simultaneous rule in `ApplianceScanAggregator`); pan across two
      of the same device type at once and confirm the count reflects the
      higher simultaneous number, not a running total.
- [ ] Click **Stop Scan** — feed freezes, status shows `Stopped: <room>
      (not yet saved)`, **Stop** disables, **Save** stays enabled.
- [ ] Click **Save Report** — status shows `Saved: .../roomscan_report.html`,
      **Start**/room-name field re-enable, the new session appears at the
      top of **Previous Sessions**.
- [ ] Run `capture_healthcheck.py --live` separately beforehand if a scan
      looks wrong, to isolate a bad stream/sensor from a dashboard bug:
      `python3 capture_healthcheck.py --live --start-streaming --device-ip <ip> --interface usb`.

## 2. Testing with VRS playback

`roomscan_dashboard.py`/`LiveScanController` are live-only (they own an
`AriaCapture(source="live", ...)`); VRS playback is exercised through the
CLI orchestrator instead, which shares 100% of the detection/estimation/
report code the dashboard uses via `finalize_scan()`:

```bash
python roomscan.py --vrs /path/to/walkthrough.vrs --room-name "Living room" --out roomscan_out
```

Checklist:
- [ ] Exit code is `0` if any device was found, `2` if none (per
      `roomscan.py:main()` — a non-zero/non-2 exit means a crash, not "no
      devices").
- [ ] Console summary lists per-device counts/confidence and totals.
- [ ] `roomscan_out/roomscan_report.json` and `roomscan_report.html` exist;
      opening the HTML in a browser shows the same devices/totals as the
      console summary plus embedded crop thumbnails.
- [ ] The new session appears in the dashboard's **Previous Sessions**
      list next time `roomscan_dashboard.py` is launched with the same
      `--out` directory (both write through the same
      `energy_sessions.py` index).
- [ ] Re-run against the same VRS file — a second, distinctly-timestamped
      session folder is created (never overwrites the first).

## 3. Verifying device detection is stable

- [ ] Unit-level: `python -m pytest tests/test_energy.py -q -k Aggregator or Stabilizer` —
      covers max-simultaneous counting, best-confidence crop selection, and
      (this session's fix) that concurrent writer/reader access from
      separate threads never raises (`ThreadSafetyTests`).
- [ ] Live/VRS: watch the **Detected Devices** table for several seconds
      once a device is in frame — the count for a class should not
      flicker (e.g. `tv: 1` one tick, `tv: 0` the next, `tv: 1` again)
      once `DetectionStabilizer`'s `min_hits`/`window_seconds` window has
      elapsed; occasional confidence-value jitter is fine, count jitter is
      not.
- [ ] Briefly occlude a detected device (hand in front of camera) — the
      stabilized count should hold steady rather than immediately drop to
      zero (this is the point of `stabilized_counts()` vs
      `instantaneous_counts()` in `LiveScanController.snapshot()`).
- [ ] Two physically distinct instances of the same class in frame at
      once (e.g. two laptops) should count as 2, not 1.

## 4. Verifying energy estimates update correctly

- [ ] Unit-level: `python -m pytest tests/test_energy.py -q -k Estimator` —
      confirms `estimate_room()`/`estimate_device()` math against
      `config.ENERGY_CATALOG` priors and `DAYS_PER_YEAR`.
- [ ] While a live scan is running, cross-check one detected device by
      hand: `watts_active * count` should equal the per-device row's
      watts column in **Detected Devices**; `kwh_per_day = watts_active *
      hours_active_per_day / 1000` per the catalog entry, and
      `cost_per_year_usd = kwh_per_year * cost_per_kwh_usd`.
- [ ] **Energy Summary** panel's `Total watts` should equal the sum of
      every device row's watts column, and should change (not freeze) as
      devices enter/leave frame.
- [ ] **Top Energy Drains** list should stay sorted by `kwh_per_year`
      descending as new devices are detected (relies on
      `energy_estimator.estimate_room()` already returning devices
      pre-sorted — the dashboard does not re-sort).
- [ ] Adding a device with a catalog `cost_per_kwh_usd` different from
      the config default (if one is set in the catalog entry) should be
      reflected in that device's own `$/yr` value, not the global default.

## 5. Verifying reports export correctly

- [ ] After **Save Report** (dashboard) or a completed `roomscan.py --vrs`
      run, confirm on disk in the session's `<room>_<timestamp>/` folder:
      - `roomscan_report.json` — valid JSON, `scan.room_name` matches what
        was typed/passed, `devices`/`totals`/`recommendations` all present.
      - `roomscan_report.html` — opens standalone in a browser (no local
        server needed — crops are base64-inlined), shows the same room
        name/devices/totals as the JSON.
      - `crops/` — one JPEG per detected instance slot, matching the
        counts in the report.
- [ ] Regenerate HTML only, from an existing JSON, and confirm it's
      byte-for-byte re-derivable: `python energy_report.py --json
      roomscan_out/<session>/roomscan_report.json`.
- [ ] Unit-level: `python -m pytest tests/test_energy.py -q -k FinalizeScan` —
      confirms `finalize_scan()` (the single code path both `roomscan.py`
      and the dashboard's Save button use) writes both artifact files and
      registers the session, without needing YOLO/torch/Qt.

## 6. Verifying session history works

- [ ] Unit-level: `python -m pytest tests/test_energy_sessions.py -q` —
      register/list/get/compare/CSV-export round trips against a temp
      index file.
- [ ] Dashboard-level smoke test (headless, no hardware):
      `python -m pytest tests/test_roomscan_dashboard.py -q` — drives the
      real `RoomScanDashboard` widget tree against a fake
      `LiveScanController` through Start -> poll -> Stop -> Save, and
      asserts the saved session reloads into **Previous Sessions**.
- [ ] Manually: after two or more saved sessions, confirm
      **Previous Sessions** lists most-recent-first.
- [ ] Select exactly one session, click **Review** — opens
      `roomscan_report.html` in the system browser via `webbrowser.open()`.
- [ ] Select zero, one, or three+ sessions and click **Compare** — shows
      an info dialog asking for exactly two, rather than crashing.
- [ ] Select exactly two sessions, click **Compare** — dialog shows a
      per-device count/kWh/cost table (devices present in only one
      session show `0` on the other side) plus a totals delta line
      (`B - A`) with a correct sign.
- [ ] Click **Export Summary...**, save to a path — resulting CSV has one
      row per registered session with a `session_id`, `room_name`, and
      `kwh_per_year` column matching the index.

---

## Automated coverage summary

| Area | Test file | Hardware-free? |
|---|---|---|
| Detection counting/stabilization + threading | `tests/test_energy.py` | Yes |
| Energy math | `tests/test_energy.py` | Yes |
| Report writing + session registration (`finalize_scan`) | `tests/test_energy.py` | Yes |
| Session index CRUD + compare + CSV export | `tests/test_energy_sessions.py` | Yes |
| Session folder naming contract | `tests/test_energy_sessions.py` | Yes |
| Full dashboard lifecycle (Start/Stop/Save/reload) | `tests/test_roomscan_dashboard.py` | Yes (fake `LiveScanController`) |
| Full live-scan flow end-to-end (Start -> feed -> detect -> stabilize -> totals -> Stop -> Save -> Past Scans) through the REAL `LiveScanController`/`AriaCapture`/dashboard, faking only the Aria SDK and YOLO boundary | `tests/test_roomscan_e2e_live.py` | Yes |
| `--debug-camera-only` mode: frames still render with detection fully disabled (no `EnergyDetector` constructed) | `tests/test_roomscan_e2e_live.py::test_debug_camera_only_shows_frames_with_detection_disabled` | Yes |
| Live Aria stream capture/detection | — | No — manual, Section 1 (still the only way to catch real-hardware-specific issues: cert handshake, actual sensor timing/skew, real appliance recognition) |
| VRS playback capture/detection | — | Partially (`roomscan.py --vrs` manual run, Section 2); pipeline logic itself is covered hardware-free by `tests/test_energy.py` |
