"""Browser report page generator for RoomScan energy audits.

Turns a roomscan_report.json dict into a single self-contained HTML file:
crop JPEGs are inlined as base64 data URIs so the page can be opened,
projected, or shared with no server and no sibling files. Also usable
standalone to regenerate the page after hand-editing the JSON:

    python energy_report.py --json roomscan_out/roomscan_report.json
"""

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Dict, List

from config import ENERGY_CATALOG, ROOMSCAN_OUTPUT_DIR, ROOMSCAN_REPORT_HTML_NAME, ROOMSCAN_REPORT_JSON_NAME
from logging_utils import get_logger

logger = get_logger(__name__)

_PAGE_STYLE = """
:root { color-scheme: dark; }
* { box-sizing: border-box; margin: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0e1116; color: #e6e9ef; padding: 32px; }
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.9rem; margin-bottom: 4px; }
.sub { color: #8a93a5; margin-bottom: 28px; font-size: 0.95rem; }
.totals { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; }
.stat { background: #171c26; border: 1px solid #232a38; border-radius: 12px; padding: 18px 24px; min-width: 170px; }
.stat .v { font-size: 1.7rem; font-weight: 600; color: #5dd0a0; }
.stat .k { color: #8a93a5; font-size: 0.85rem; margin-top: 4px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
.card { background: #171c26; border: 1px solid #232a38; border-radius: 12px; overflow: hidden; }
.card .imgs { display: flex; gap: 2px; background: #0b0e13; }
.card .imgs img { flex: 1; min-width: 0; object-fit: cover; height: 170px; }
.card .body { padding: 16px 18px; }
.card h2 { font-size: 1.1rem; margin-bottom: 2px; }
.card .cnt { color: #5dd0a0; font-weight: 600; }
.row { display: flex; justify-content: space-between; font-size: 0.9rem; padding: 3px 0; color: #b7bfcd; }
.row b { color: #e6e9ef; font-weight: 600; }
.row.note { display: block; color: #f5b942; font-size: 0.82rem; padding-bottom: 8px; }
.noimg { height: 170px; display: flex; align-items: center; justify-content: center; color: #4a5265; background: #0b0e13; }
.recs { margin: 28px 0; }
.recs h2 { font-size: 1.2rem; margin-bottom: 12px; }
.recs ul { list-style: none; display: flex; flex-direction: column; gap: 8px; }
.recs li { background: #171c26; border: 1px solid #232a38; border-left: 3px solid #5dd0a0; border-radius: 8px; padding: 12px 16px; font-size: 0.92rem; color: #dbe1ea; }
.gemini-note { color: #f5b942; font-size: 0.85rem; margin: 8px 0 28px; }
.ai-badge { display: inline-block; background: #2a2410; color: #f5b942; border: 1px solid #f5b942; border-radius: 6px; font-size: 0.7rem; font-weight: 600; padding: 1px 7px; margin-left: 8px; vertical-align: middle; }
.cats { margin-bottom: 28px; }
.cats h2 { font-size: 1.2rem; margin-bottom: 12px; }
.cat-row { display: flex; align-items: center; gap: 12px; background: #171c26; border: 1px solid #232a38; border-radius: 8px; padding: 10px 16px; margin-bottom: 6px; font-size: 0.9rem; }
.cat-row .cat-name { flex: 1; font-weight: 600; color: #e6e9ef; }
.cat-row .cat-count { color: #8a93a5; min-width: 90px; }
.cat-row .cat-kwh { color: #5dd0a0; min-width: 90px; text-align: right; }
.cat-row .cat-cost { color: #e6e9ef; min-width: 70px; text-align: right; }
footer { margin-top: 36px; color: #5a6375; font-size: 0.8rem; }
"""

# Keyword buckets for the report's "Breakdown by Category" section -- checked
# in order, first match wins, so more specific terms come first. Covers both
# catalog appliances (ENERGY_CATALOG's fixed COCO classes) and open-vocabulary
# Gemini-discovered devices (outlets/lights/vents/etc.) the same way, since
# both device shapes carry class_name/display_name/notes.
_CATEGORY_KEYWORDS = [
    ("Lighting", ("light", "lamp", "bulb", "fixture", "chandelier", "sconce")),
    (
        "HVAC & Climate",
        ("vent", "duct", "register", "air conditioner", "fan", "heater", "thermostat", "humidifier", "dehumidifier"),
    ),
    (
        "Kitchen & Major Appliances",
        ("refrigerator", "oven", "microwave", "toaster", "dishwasher", "washer", "dryer", "water heater", "hair dr"),
    ),
    (
        "Electronics & Standby",
        (
            "tv", "television", "laptop", "computer", "monitor", "cell phone", "phone", "clock",
            "speaker", "console", "router", "modem", "hub", "printer", "charger", "power strip",
            "outlet", "receptacle", "socket",
        ),
    ),
]
_OTHER_CATEGORY = "Other"


def _categorize_device(device: Dict[str, object]) -> str:
    parts = [str(device.get("class_name", "")), str(device.get("display_name", ""))]
    parts.extend(str(n) for n in device.get("notes", []) or [])
    text = " ".join(parts).lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return category
    return _OTHER_CATEGORY


def _crop_data_uris(device: Dict[str, object], out_dir: Path) -> List[str]:
    uris: List[str] = []
    for rel in device.get("crops", []):
        path = out_dir / str(rel)
        try:
            uris.append("data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii"))
        except OSError as exc:
            logger.warning("Crop %s unreadable, omitting from page: %s", path, exc)
    return uris


def _device_card(device: Dict[str, object], out_dir: Path) -> str:
    uris = _crop_data_uris(device, out_dir)
    if uris:
        imgs = "".join(f'<img src="{u}" alt="detection crop">' for u in uris[:3])
        img_block = f'<div class="imgs">{imgs}</div>'
    else:
        img_block = '<div class="noimg">no snapshot</div>'
    name = html.escape(str(device["display_name"]))
    notes = [html.escape(str(n)) for n in device.get("notes", []) if n]
    notes_block = f'<div class="row note">{" &middot; ".join(notes)}</div>' if notes else ""
    badge = '<span class="ai-badge">AI</span>' if device.get("source") == "gemini_discovered" else ""
    return f"""
  <div class="card">
    {img_block}
    <div class="body">
      <h2>{name} <span class="cnt">&times;{device['count']}</span>{badge}</h2>
      {notes_block}
      <div class="row"><span>Active draw</span><b>{device['watts_active']:.0f} W</b></div>
      <div class="row"><span>Assumed use</span><b>{device['hours_per_day']:g} h/day</b></div>
      <div class="row"><span>Energy</span><b>{device['kwh_per_day']:.2f} kWh/day &middot; {device['kwh_per_year']:.0f} kWh/yr</b></div>
      <div class="row"><span>Est. cost</span><b>${device['cost_per_year_usd']:.2f} /yr</b></div>
    </div>
  </div>"""


def _category_breakdown_block(report: Dict[str, object]) -> str:
    devices = report.get("devices", [])
    if not devices:
        return ""
    totals_by_category: Dict[str, Dict[str, float]] = {}
    for device in devices:
        bucket = totals_by_category.setdefault(
            _categorize_device(device), {"count": 0.0, "kwh_per_year": 0.0, "cost_per_year_usd": 0.0}
        )
        bucket["count"] += float(device["count"])
        bucket["kwh_per_year"] += float(device["kwh_per_year"])
        bucket["cost_per_year_usd"] += float(device["cost_per_year_usd"])
    ordered = sorted(totals_by_category.items(), key=lambda kv: kv[1]["kwh_per_year"], reverse=True)
    rows = "".join(
        f'<div class="cat-row"><span class="cat-name">{html.escape(category)}</span>'
        f'<span class="cat-count">{int(vals["count"])} device{"s" if vals["count"] != 1 else ""}</span>'
        f'<span class="cat-kwh">{vals["kwh_per_year"]:.0f} kWh/yr</span>'
        f'<span class="cat-cost">${vals["cost_per_year_usd"]:.0f}/yr</span></div>'
        for category, vals in ordered
    )
    return f'<div class="cats"><h2>Breakdown by Category</h2>{rows}</div>'


def _recommendations_block(report: Dict[str, object]) -> str:
    items = report.get("recommendations", [])
    if not items:
        return ""
    lis = "".join(f"<li>{html.escape(str(s))}</li>" for s in items)
    return f'<div class="recs"><h2>Recommended Actions</h2><ul>{lis}</ul></div>'


def _gemini_rejected_note(report: Dict[str, object]) -> str:
    classes = report.get("scan", {}).get("gemini_rejected_classes", [])
    if not classes:
        return ""
    names = ", ".join(html.escape(ENERGY_CATALOG.get(c, {}).get("display", c)) for c in classes)
    return f'<div class="gemini-note">Gemini vision auto-corrected a likely misclassification and excluded it from: {names}.</div>'


def render_html(report: Dict[str, object], out_dir: Path, html_path: Path) -> None:
    """Write the self-contained report page for one roomscan report dict."""
    scan = report["scan"]
    totals = report["totals"]
    cards = "".join(_device_card(d, out_dir) for d in report["devices"])
    if not cards:
        cards = '<div class="noimg" style="border-radius:12px">No appliances detected in this scan.</div>'
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoomScan &mdash; {html.escape(str(scan['room_name']))}</title>
<style>{_PAGE_STYLE}</style></head>
<body><div class="wrap">
<h1>RoomScan Energy Audit &mdash; {html.escape(str(scan['room_name']))}</h1>
<div class="sub">{html.escape(str(scan['source']))} &middot; {scan['frames_sampled']} frames analyzed &middot; {html.escape(str(scan['generated_at']))}</div>
<div class="totals">
  <div class="stat"><div class="v">{totals['device_count']}</div><div class="k">devices found</div></div>
  <div class="stat"><div class="v">{totals['kwh_per_day']:.1f}</div><div class="k">kWh / day</div></div>
  <div class="stat"><div class="v">{totals['kwh_per_year']:.0f}</div><div class="k">kWh / year</div></div>
  <div class="stat"><div class="v">${totals['cost_per_year_usd']:.0f}</div><div class="k">est. cost / year @ ${totals['cost_per_kwh_usd']:.2f}/kWh</div></div>
</div>
{_category_breakdown_block(report)}
<div class="grid">{cards}
</div>
{_gemini_rejected_note(report)}
{_recommendations_block(report)}
<footer>Estimates use typical-draw priors per appliance class, not measured consumption. Captured with Meta Project Aria Gen 1.</footer>
</div></body></html>"""
    try:
        html_path.write_text(page, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write report HTML {html_path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the RoomScan HTML report from its JSON.")
    parser.add_argument("--json", default=str(ROOMSCAN_OUTPUT_DIR / ROOMSCAN_REPORT_JSON_NAME), help="Path to roomscan_report.json.")
    parser.add_argument("--out", default=None, help="Output HTML path (default: alongside the JSON).")
    args = parser.parse_args()

    json_path = Path(args.json).expanduser()
    try:
        report = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read report JSON {json_path}") from exc
    out_dir = json_path.parent
    html_path = Path(args.out).expanduser() if args.out else out_dir / ROOMSCAN_REPORT_HTML_NAME
    render_html(report, out_dir, html_path)
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
