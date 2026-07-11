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

from config import ROOMSCAN_OUTPUT_DIR, ROOMSCAN_REPORT_HTML_NAME, ROOMSCAN_REPORT_JSON_NAME
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
.noimg { height: 170px; display: flex; align-items: center; justify-content: center; color: #4a5265; background: #0b0e13; }
.recs { margin: 28px 0; }
.recs h2 { font-size: 1.2rem; margin-bottom: 12px; }
.recs ul { list-style: none; display: flex; flex-direction: column; gap: 8px; }
.recs li { background: #171c26; border: 1px solid #232a38; border-left: 3px solid #5dd0a0; border-radius: 8px; padding: 12px 16px; font-size: 0.92rem; color: #dbe1ea; }
footer { margin-top: 36px; color: #5a6375; font-size: 0.8rem; }
"""


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
    return f"""
  <div class="card">
    {img_block}
    <div class="body">
      <h2>{name} <span class="cnt">&times;{device['count']}</span></h2>
      <div class="row"><span>Active draw</span><b>{device['watts_active']:.0f} W</b></div>
      <div class="row"><span>Assumed use</span><b>{device['hours_per_day']:g} h/day</b></div>
      <div class="row"><span>Energy</span><b>{device['kwh_per_day']:.2f} kWh/day &middot; {device['kwh_per_year']:.0f} kWh/yr</b></div>
      <div class="row"><span>Est. cost</span><b>${device['cost_per_year_usd']:.2f} /yr</b></div>
    </div>
  </div>"""


def _recommendations_block(report: Dict[str, object]) -> str:
    items = report.get("recommendations", [])
    if not items:
        return ""
    lis = "".join(f"<li>{html.escape(str(s))}</li>" for s in items)
    return f'<div class="recs"><h2>Recommended Actions</h2><ul>{lis}</ul></div>'


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
<div class="grid">{cards}
</div>
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
