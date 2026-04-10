"""Generate a self-contained crime analytics dashboard HTML.

Phase 4 deliverable:
- Read 7 dashboard JSON files from output/dashboard_data/
- Read india_states.geojson
- Inject all data inline as JS variables
- Write dashboard/index.html (single-file HTML/CSS/JS)
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DASHBOARD_DATA_DIR = BASE_DIR / "output" / "dashboard_data"
DASHBOARD_DIR = BASE_DIR / "dashboard"
OUTPUT_HTML = DASHBOARD_DIR / "index.html"


REQUIRED_JSON_FILES = [
    "district_analysis.json",
    "women_safety.json",
    "forecasts.json",
    "crime_profiles.json",
    "clusters.json",
    "supplementary.json",
    "national_trends.json",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def build_key_findings(payload: Dict[str, Any]) -> List[str]:
    findings: List[str] = []

    national = payload["national_trends"]
    yearly = national.get("yearly_totals", [])
    if yearly:
        first = yearly[0]
        last = yearly[-1]
        ipc_change = safe_float(last.get("total_ipc")) - safe_float(first.get("total_ipc"))
        women_change = safe_float(last.get("total_women")) - safe_float(first.get("total_women"))
        ipc_pct = (ipc_change / safe_float(first.get("total_ipc"), 1.0)) * 100.0 if safe_float(first.get("total_ipc"), 0.0) else 0.0
        women_pct = (women_change / safe_float(first.get("total_women"), 1.0)) * 100.0 if safe_float(first.get("total_women"), 0.0) else 0.0
        findings.append(
            f"National IPC crimes changed by {ipc_pct:.1f}% between {first.get('year')} and {last.get('year')}."
        )
        findings.append(
            f"Crimes against women changed by {women_pct:.1f}% over the same period."
        )

    district = payload["district_analysis"]
    hotspots = district.get("hotspots_2014", [])
    if hotspots:
        top = hotspots[-1]
        findings.append(
            f"Highest district severity in 2014: {top.get('district')}, {top.get('state')} (score {safe_float(top.get('severity_score')):.2f})."
        )

    rising = district.get("rising_hotspots", [])
    if rising:
        fast = rising[-1]
        findings.append(
            f"Fastest-rising hotspot (2010-2014 avg YoY): {fast.get('district')}, {fast.get('state')} at {safe_float(fast.get('avg_yoy_growth_2010_2014')):.2f}%."
        )

    clusters = payload["clusters"]
    summaries = clusters.get("cluster_summaries", [])
    if summaries:
        biggest = max(summaries, key=lambda x: safe_float(x.get("state_count")))
        most_violent = max(summaries, key=lambda x: safe_float(x.get("avg_murder")))
        findings.append(
            f"Largest cluster: '{biggest.get('cluster_label')}' with {int(safe_float(biggest.get('state_count')))} states."
        )
        findings.append(
            f"Most violent cluster by average murders: '{most_violent.get('cluster_label')}'."
        )

    women = payload["women_safety"]
    safety = women.get("state_safety_index", [])
    if safety:
        by_state: Dict[str, List[float]] = {}
        for row in safety:
            st = row.get("state")
            by_state.setdefault(st, []).append(safe_float(row.get("women_safety_index")))
        if by_state:
            avg_safety = {k: mean(v) for k, v in by_state.items() if v}
            if avg_safety:
                safest = max(avg_safety.items(), key=lambda kv: kv[1])
                riskiest = min(avg_safety.items(), key=lambda kv: kv[1])
                findings.append(f"Safest state by average Women Safety Index: {safest[0]} ({safest[1]:.2f}).")
                findings.append(f"Highest women-safety risk state by index: {riskiest[0]} ({riskiest[1]:.2f}).")

    return findings[:8]


DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>India Crime Intelligence Dashboard</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>
  <style>
    :root {
      --bg: #f7f7f3;
      --surface: #ffffff;
      --surface-alt: #f1f3ec;
      --ink: #20231f;
      --muted: #66706a;
      --border: #d9ddd3;
      --accent: #c66a1f;
      --accent-soft: #ffe2c4;
      --danger: #b22222;
      --safe: #2f8f4e;
      --shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
      --radius: 12px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 10%, #fff4dc 0%, transparent 25%),
        radial-gradient(circle at 90% 0%, #ebf9ff 0%, transparent 30%),
        var(--bg);
    }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 1200;
      background: rgba(255, 255, 255, 0.92);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
      box-shadow: 0 6px 14px rgba(0,0,0,0.04);
    }

    .brand-row {
      max-width: 1500px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 12px 18px 8px;
    }

    .title {
      font-size: 20px;
      margin: 0;
      letter-spacing: 0.2px;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
    }

    .tabs {
      max-width: 1500px;
      margin: 0 auto;
      padding: 0 12px 12px;
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 8px;
    }

    .tab-btn {
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--ink);
      border-radius: 10px;
      padding: 10px 8px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s ease;
      font-size: 12px;
    }

    .tab-btn.active {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: #5a2d00;
      transform: translateY(-1px);
    }

    main {
      max-width: 1500px;
      margin: 16px auto 26px;
      padding: 0 12px;
    }

    .tab-panel {
      display: none;
      animation: fadein 0.22s ease;
    }

    .tab-panel.active { display: block; }

    @keyframes fadein {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 14px;
    }

    .grid-2 { display: grid; grid-template-columns: 1.3fr 1fr; gap: 12px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      align-items: center;
      margin-bottom: 10px;
    }

    .controls label {
      font-size: 12px;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    select, input[type="range"] {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      padding: 6px 8px;
      font-size: 12px;
      color: var(--ink);
    }

    select[multiple] {
      min-height: 82px;
      padding: 8px;
    }

    .map {
      width: 100%;
      height: 480px;
      border-radius: 10px;
      border: 1px solid var(--border);
      overflow: hidden;
    }

    .map.small { height: 420px; }

    .chart-wrap {
      position: relative;
      min-height: 280px;
      height: 320px;
    }

    .chart-wrap.tall { min-height: 420px; height: 460px; }

    .chart-wrap canvas {
      width: 100% !important;
      height: 100% !important;
    }

    .legend-title {
      margin: 0 0 6px;
      font-weight: 600;
      font-size: 13px;
    }

    .metric-list {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 10px;
    }

    .metric {
      background: var(--surface-alt);
      border-radius: 10px;
      padding: 10px;
      border: 1px solid var(--border);
    }

    .metric .k { font-size: 11px; color: var(--muted); }
    .metric .v { font-size: 16px; font-weight: 700; margin-top: 2px; }

    .hotspot-box {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .hotspot-list {
      margin: 0;
      padding-left: 18px;
      font-size: 12px;
      line-height: 1.45;
    }

    .state-line {
      font-size: 12px;
      color: var(--muted);
      margin-top: 6px;
    }

    .funnel {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }

    .funnel-row {
      display: grid;
      grid-template-columns: 140px 1fr 58px;
      gap: 10px;
      align-items: center;
    }

    .funnel-bar {
      height: 14px;
      border-radius: 999px;
      background: linear-gradient(90deg, #ffcf99, #d96f22);
    }

    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--surface-alt);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 11px;
      margin-right: 6px;
    }

    .gauge {
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #f0f2ee;
      height: 22px;
      overflow: hidden;
    }

    .gauge-fill {
      height: 100%;
      background: linear-gradient(90deg, #d94a3a, #f39a3d, #4fa764);
      width: 0;
      transition: width 0.3s ease;
    }

    .clusters {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }

    .cluster-card {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      background: #fff;
    }

    .cluster-card h4 {
      margin: 0 0 6px;
      font-size: 14px;
    }

    .insight-list {
      margin: 0;
      padding-left: 18px;
      line-height: 1.5;
      font-size: 13px;
    }

    .no-data {
      height: 100%;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 13px;
      border: 1px dashed var(--border);
      border-radius: 8px;
      background: #fafbf9;
    }

    #global-loader {
      position: fixed;
      inset: 0;
      background: rgba(255, 255, 255, 0.76);
      z-index: 2000;
      display: grid;
      place-items: center;
      color: #2b2f2c;
      font-weight: 600;
    }

    .loader-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 18px 0 0 #e7a364, -18px 0 0 #f1d0af;
      animation: pulse 1s infinite ease-in-out;
      margin: 0 auto 10px;
    }

    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.24); }
    }

    .leaflet-control.legend-control {
      background: rgba(255,255,255,0.95);
      padding: 8px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 11px;
      line-height: 1.35;
      min-width: 165px;
    }

    .legend-row {
      display: grid;
      grid-template-columns: 14px 1fr;
      gap: 6px;
      align-items: center;
      margin-top: 4px;
    }

    .sw {
      width: 14px;
      height: 14px;
      border: 1px solid rgba(0,0,0,0.15);
    }

    @media (max-width: 1100px) {
      .tabs { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
      .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
      .map, .map.small { height: 360px; }
      .clusters { grid-template-columns: 1fr; }
      .metric-list { grid-template-columns: 1fr 1fr; }
    }

    @media (max-width: 720px) {
      .tabs { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .metric-list { grid-template-columns: 1fr; }
      .hotspot-box { grid-template-columns: 1fr; }
      .funnel-row { grid-template-columns: 110px 1fr 52px; }
    }
  </style>
</head>
<body>
  <div id="global-loader">
    <div>
      <div class="loader-dot"></div>
      Rendering dashboard...
    </div>
  </div>

  <header class="topbar">
    <div class="brand-row">
      <div>
        <h1 class="title">India Crime Intelligence Platform</h1>
        <p class="subtitle">2001-2014 analytics dashboard with forecasting and cluster insights</p>
      </div>
      <div id="as-of" class="badge"></div>
    </div>
    <nav class="tabs">
      <button class="tab-btn active" data-tab="tab1">1. India Overview</button>
      <button class="tab-btn" data-tab="tab2">2. District Deep-Dive</button>
      <button class="tab-btn" data-tab="tab3">3. Women Safety</button>
      <button class="tab-btn" data-tab="tab4">4. Trends & Forecasting</button>
      <button class="tab-btn" data-tab="tab5">5. Crime Profiles</button>
      <button class="tab-btn" data-tab="tab6">6. Clusters & Findings</button>
    </nav>
  </header>

  <main>
    <section id="tab1" class="tab-panel active">
      <div class="card">
        <div class="controls">
          <label>Year <input id="tab1-year" type="range" min="2001" max="2014" step="1" value="2014" /></label>
          <span id="tab1-year-label" class="badge"></span>
          <label>Metric
            <select id="tab1-metric">
              <option value="total_ipc">Total IPC</option>
              <option value="violent">Violent Crimes</option>
              <option value="property">Property Crimes</option>
              <option value="women">Women Crimes</option>
              <option value="severity">Severity Score</option>
            </select>
          </label>
        </div>
        <div id="tab1-map" class="map"></div>
        <div id="tab1-state-card" class="card" style="margin-top:10px;"></div>
      </div>
    </section>

    <section id="tab2" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <div class="controls">
            <label>State <select id="tab2-state"></select></label>
            <label>Year <select id="tab2-year"></select></label>
          </div>
          <div class="chart-wrap tall"><canvas id="tab2-bar"></canvas></div>
          <div class="state-line" id="tab2-selection"></div>
        </div>
        <div class="card">
          <h3 style="margin-top:0">Hotspot Cards</h3>
          <div class="hotspot-box">
            <div>
              <p class="legend-title">Top 5 Dangerous (2014)</p>
              <ol id="tab2-hot-danger" class="hotspot-list"></ol>
            </div>
            <div>
              <p class="legend-title">Top 5 Fastest Rising</p>
              <ol id="tab2-hot-rising" class="hotspot-list"></ol>
            </div>
          </div>
          <hr style="border:none;border-top:1px solid var(--border);margin:14px 0" />
          <h3 style="margin:0 0 8px">District Detail</h3>
          <div class="grid-2" style="grid-template-columns:1fr 1fr;">
            <div class="chart-wrap"><canvas id="tab2-donut"></canvas></div>
            <div class="chart-wrap"><canvas id="tab2-spark"></canvas></div>
          </div>
          <div id="tab2-district-meta" class="state-line"></div>
        </div>
      </div>
    </section>

    <section id="tab3" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <div class="controls">
            <label>Year <select id="tab3-year"></select></label>
          </div>
          <div id="tab3-map" class="map small"></div>
        </div>
        <div class="card">
          <div class="controls">
            <label>State <select id="tab3-state"></select></label>
          </div>
          <h3 style="margin:2px 0 4px">Justice Funnel</h3>
          <div id="tab3-funnel" class="funnel"></div>
          <div id="tab3-dropoff" class="state-line"></div>
        </div>
      </div>
      <div class="grid-2" style="margin-top:12px;">
        <div class="card">
          <h3 style="margin-top:0">Women Crime Type Breakdown by State</h3>
          <div class="chart-wrap tall"><canvas id="tab3-stack"></canvas></div>
        </div>
        <div class="card">
          <h3 style="margin-top:0">National Trend Lines</h3>
          <div class="chart-wrap tall"><canvas id="tab3-trend"></canvas></div>
        </div>
      </div>
    </section>

    <section id="tab4" class="tab-panel">
      <div class="card">
        <div class="controls">
          <label>States (1-3)
            <select id="tab4-states" multiple></select>
          </label>
          <label>Metric
            <select id="tab4-metric">
              <option value="total_ipc">Total IPC</option>
              <option value="total_women">Women Crimes</option>
            </select>
          </label>
          <label><input id="tab4-national" type="checkbox" checked /> Show national average</label>
        </div>
        <div class="chart-wrap tall"><canvas id="tab4-lines"></canvas></div>
        <p id="tab4-model-info" class="state-line"></p>
      </div>
    </section>

    <section id="tab5" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <div class="controls">
            <label>Radar State A <select id="tab5-radar-a"></select></label>
            <label>Radar State B <select id="tab5-radar-b"></select></label>
          </div>
          <div class="chart-wrap"><canvas id="tab5-radar"></canvas></div>
        </div>
        <div class="card">
          <div class="controls">
            <label>Profile State <select id="tab5-state"></select></label>
          </div>
          <div class="grid-2" style="grid-template-columns:1fr 1fr;">
            <div>
              <h4 style="margin:0 0 4px">Kidnapping Motives</h4>
              <div class="chart-wrap"><canvas id="tab5-kidnap"></canvas></div>
            </div>
            <div>
              <h4 style="margin:0 0 4px">Crime Geography</h4>
              <div class="chart-wrap"><canvas id="tab5-geo"></canvas></div>
            </div>
          </div>
        </div>
      </div>
      <div class="grid-2" style="margin-top:12px;">
        <div class="card">
          <h3 style="margin-top:0">Murder Demographics (Male/Female x Age Brackets)</h3>
          <div class="chart-wrap"><canvas id="tab5-murder"></canvas></div>
        </div>
        <div class="card">
          <h3 style="margin-top:0">Property Recovery</h3>
          <div id="tab5-recovery-text" class="state-line"></div>
          <div class="gauge"><div id="tab5-recovery-fill" class="gauge-fill"></div></div>
          <div class="metric-list" id="tab5-metrics"></div>
        </div>
      </div>
    </section>

    <section id="tab6" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <h3 style="margin-top:0">Cluster Scatter Plot</h3>
          <div class="chart-wrap"><canvas id="tab6-scatter"></canvas></div>
        </div>
        <div class="card">
          <h3 style="margin-top:0">Cluster Cards</h3>
          <div id="tab6-clusters" class="clusters"></div>
        </div>
      </div>
      <div class="card" style="margin-top:12px;">
        <h3 style="margin-top:0">Key Findings</h3>
        <ol id="tab6-findings" class="insight-list"></ol>
      </div>
    </section>
  </main>

  <script>
    window.DASHBOARD_DATA = __DASHBOARD_DATA__;
  </script>

  <script>
    const data = window.DASHBOARD_DATA;
    const fmt = new Intl.NumberFormat("en-IN");

    const palette = {
      seq: ["#fff4b0", "#ffc46b", "#f1863c", "#c6451a", "#7f1d13"],
      womenSafety: ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf", "#2166ac", "#1a9850"],
      profile: { violent: "#c3382b", property: "#2c6fa8", women: "#7f3a97" },
      states: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    };

    const charts = {};
    const years = Array.from({ length: 14 }, (_, i) => 2001 + i);

    const aliasToCanonical = {
      "ORISSA": "ODISHA",
      "UTTARANCHAL": "UTTARAKHAND",
      "ANDAMAN AND NICOBAR": "ANDAMAN & NICOBAR ISLANDS",
      "DADRA AND NAGAR HAVELI": "DADRA & NAGAR HAVELI",
      "DAMAN AND DIU": "DAMAN & DIU",
      "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
      "DELHI UT": "DELHI"
    };

    function canon(s) {
      if (!s) return "";
      let up = String(s).trim().toUpperCase();
      if (aliasToCanonical[up]) up = aliasToCanonical[up];
      return up;
    }

    function prettify(s) {
      return String(s || "").toLowerCase().replace(/\b\w/g, (m) => m.toUpperCase());
    }

    function getSelectedValues(sel) {
      return Array.from(sel.selectedOptions).map((o) => o.value);
    }

    function createLegendControl(mapRef, title = "Legend") {
      const legend = L.control({ position: "bottomright" });
      legend.onAdd = function () {
        const div = L.DomUtil.create("div", "leaflet-control legend-control");
        div.innerHTML = `<strong>${title}</strong><div id="${mapRef._container.id}-legend-body"></div>`;
        return div;
      };
      legend.addTo(mapRef);
      return legend;
    }

    function renderLegend(containerId, bins, colors) {
      const body = document.getElementById(containerId);
      if (!body) return;
      body.innerHTML = bins
        .map((b, i) => {
          const lo = Number.isFinite(b[0]) ? b[0] : 0;
          const hi = Number.isFinite(b[1]) ? b[1] : 0;
          return `<div class="legend-row"><span class="sw" style="background:${colors[i]}"></span><span>${fmt.format(Math.round(lo))} - ${fmt.format(Math.round(hi))}</span></div>`;
        })
        .join("");
    }

    function quantileBins(values, steps) {
      const arr = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
      if (!arr.length) return [[0, 0]];
      const bins = [];
      for (let i = 0; i < steps; i++) {
        const loIdx = Math.floor((i * (arr.length - 1)) / steps);
        const hiIdx = Math.floor((((i + 1) * (arr.length - 1)) / steps));
        bins.push([arr[loIdx], arr[hiIdx]]);
      }
      return bins;
    }

    function colorFor(value, bins, colors) {
      if (!Number.isFinite(value)) return "#e8ebe5";
      for (let i = 0; i < bins.length; i++) {
        if (value <= bins[i][1]) return colors[i] || colors[colors.length - 1];
      }
      return colors[colors.length - 1];
    }

    function makeEmpty(elId, message = "No data available") {
      const root = document.getElementById(elId);
      if (!root) return;
      root.innerHTML = `<div class="no-data">${message}</div>`;
    }

    function initializeTabs() {
      const buttons = document.querySelectorAll(".tab-btn");
      const panels = document.querySelectorAll(".tab-panel");
      buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
          buttons.forEach((b) => b.classList.remove("active"));
          panels.forEach((p) => p.classList.remove("active"));
          btn.classList.add("active");
          const panel = document.getElementById(btn.dataset.tab);
          panel.classList.add("active");

          if (window.tab1Map) setTimeout(() => window.tab1Map.invalidateSize(), 50);
          if (window.tab3Map) setTimeout(() => window.tab3Map.invalidateSize(), 50);
        });
      });
    }

    function preprocess() {
      const geoFeatures = data.geojson.features || [];

      const geoToCanonical = {};
      const canonicalToGeo = {};

      const seed = [];
      [
        ...(data.clusters.state_assignments || []),
        ...(data.women_safety.state_safety_index || []),
        ...(data.crime_profiles.radar_dimensions || []),
        ...(data.women_safety.justice_pipeline || []),
      ].forEach((row) => {
        if (row.geojson_state && row.state) {
          geoToCanonical[row.geojson_state] = canon(row.state);
          canonicalToGeo[canon(row.state)] = row.geojson_state;
        }
      });

      geoFeatures.forEach((f) => {
        const g = f.properties?.NAME_1;
        if (!geoToCanonical[g]) {
          geoToCanonical[g] = canon(g);
          canonicalToGeo[canon(g)] = g;
        }
        seed.push(geoToCanonical[g]);
      });

      const overview = {};
      const ensure = (state, year) => {
        if (!overview[state]) overview[state] = {};
        if (!overview[state][year]) {
          overview[state][year] = {
            total_ipc: 0,
            women: 0,
            violent: 0,
            property: 0,
            severity: null,
          };
        }
        return overview[state][year];
      };

      (data.forecasts.time_series || []).forEach((r) => {
        if (r.type !== "actual") return;
        const st = canon(r.state);
        const yr = Number(r.year);
        const row = ensure(st, yr);
        if (r.metric === "total_ipc") row.total_ipc = Number(r.value) || 0;
        if (r.metric === "total_women") row.women = Number(r.value) || 0;
      });

      const yoyKey = new Map();
      (data.district_analysis.yoy_growth || []).forEach((r) => {
        const key = `${canon(r.state)}||${String(r.district || "").toUpperCase()}||${r.year}`;
        yoyKey.set(key, Number(r.total_ipc) || 0);
      });

      (data.district_analysis.crime_profiles || []).forEach((r) => {
        const st = canon(r.state);
        const yr = Number(r.year);
        const ipc = yoyKey.get(`${st}||${String(r.district || "").toUpperCase()}||${r.year}`) || 0;
        const row = ensure(st, yr);
        row.violent += ipc * ((Number(r.pct_violent) || 0) / 100);
        row.property += ipc * ((Number(r.pct_property) || 0) / 100);
      });

      const sevAgg = {};
      (data.district_analysis.severity_scores || []).forEach((r) => {
        const st = canon(r.state);
        const yr = Number(r.year);
        const key = `${st}||${yr}`;
        if (!sevAgg[key]) sevAgg[key] = { sum: 0, c: 0 };
        sevAgg[key].sum += Number(r.severity_score) || 0;
        sevAgg[key].c += 1;
      });
      Object.entries(sevAgg).forEach(([k, v]) => {
        const [st, yr] = k.split("||");
        const row = ensure(st, Number(yr));
        row.severity = v.c ? v.sum / v.c : null;
      });

      return { geoFeatures, geoToCanonical, canonicalToGeo, overview, states: Array.from(new Set(seed)).sort() };
    }

    function setupTab1(ctx) {
      const yearInput = document.getElementById("tab1-year");
      const yearLabel = document.getElementById("tab1-year-label");
      const metricSel = document.getElementById("tab1-metric");
      const card = document.getElementById("tab1-state-card");

      const map = L.map("tab1-map", { zoomControl: true }).setView([22.5, 79], 4.7);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 10 }).addTo(map);
      window.tab1Map = map;
      createLegendControl(map, "Value range");

      let layer;
      let selectedState = null;

      function valuesForYearMetric(year, metric) {
        return ctx.states
          .map((s) => ctx.overview[s]?.[year]?.[metric])
          .filter((v) => Number.isFinite(v));
      }

      function rankMap(year, metric) {
        const rows = ctx.states
          .map((s) => ({ s, v: ctx.overview[s]?.[year]?.[metric] }))
          .filter((x) => Number.isFinite(x.v))
          .sort((a, b) => b.v - a.v);
        const out = {};
        rows.forEach((r, i) => { out[r.s] = i + 1; });
        return out;
      }

      function renderCard(state, year, metric, rankLookup) {
        const item = ctx.overview[state]?.[year];
        if (!item) {
          card.innerHTML = `<div class="no-data">Click a state to view details.</div>`;
          return;
        }
        card.innerHTML = `
          <h3 style="margin:0 0 6px">${prettify(state)} (${year})</h3>
          <div class="metric-list">
            <div class="metric"><div class="k">Total IPC</div><div class="v">${fmt.format(Math.round(item.total_ipc || 0))}</div></div>
            <div class="metric"><div class="k">Violent</div><div class="v">${fmt.format(Math.round(item.violent || 0))}</div></div>
            <div class="metric"><div class="k">Property</div><div class="v">${fmt.format(Math.round(item.property || 0))}</div></div>
            <div class="metric"><div class="k">Women</div><div class="v">${fmt.format(Math.round(item.women || 0))}</div></div>
            <div class="metric"><div class="k">Severity</div><div class="v">${Number.isFinite(item.severity) ? item.severity.toFixed(2) : "NA"}</div></div>
            <div class="metric"><div class="k">National Rank (${metric})</div><div class="v">${rankLookup[state] || "NA"}</div></div>
          </div>`;
      }

      function draw() {
        const year = Number(yearInput.value);
        const metric = metricSel.value;
        yearLabel.textContent = `Year ${year}`;

        const vals = valuesForYearMetric(year, metric);
        const bins = quantileBins(vals, 5);
        renderLegend("tab1-map-legend-body", bins, palette.seq);
        const rankLookup = rankMap(year, metric);

        if (layer) map.removeLayer(layer);

        layer = L.geoJSON({ type: "FeatureCollection", features: ctx.geoFeatures }, {
          style: (feature) => {
            const geoName = feature.properties.NAME_1;
            const st = ctx.geoToCanonical[geoName] || canon(geoName);
            const val = ctx.overview[st]?.[year]?.[metric];
            return {
              color: "#36403b",
              weight: selectedState === st ? 2.2 : 0.8,
              fillColor: colorFor(Number(val), bins, palette.seq),
              fillOpacity: 0.85,
            };
          },
          onEachFeature: (feature, lyr) => {
            const geoName = feature.properties.NAME_1;
            const st = ctx.geoToCanonical[geoName] || canon(geoName);
            const v = ctx.overview[st]?.[year]?.[metric];
            lyr.bindTooltip(`${geoName}<br>${metric}: ${Number.isFinite(v) ? fmt.format(Math.round(v)) : "NA"}<br>Rank: ${rankLookup[st] || "NA"}`);
            lyr.on("click", () => {
              selectedState = st;
              draw();
              renderCard(st, year, metric, rankLookup);
            });
          }
        }).addTo(map);

        if (selectedState) renderCard(selectedState, year, metric, rankLookup);
        else renderCard(null, year, metric, rankLookup);
      }

      yearInput.addEventListener("input", draw);
      metricSel.addEventListener("change", draw);
      draw();
    }

    function setupTab2() {
      const stateSel = document.getElementById("tab2-state");
      const yearSel = document.getElementById("tab2-year");
      const selection = document.getElementById("tab2-selection");
      const meta = document.getElementById("tab2-district-meta");

      const severity = data.district_analysis.severity_scores || [];
      const profiles = data.district_analysis.crime_profiles || [];
      const yoy = data.district_analysis.yoy_growth || [];

      const states = Array.from(new Set(severity.map((r) => canon(r.state)))).sort();
      stateSel.innerHTML = states.map((s) => `<option value="${s}">${prettify(s)}</option>`).join("");
      yearSel.innerHTML = years.map((y) => `<option value="${y}" ${y === 2014 ? "selected" : ""}>${y}</option>`).join("");

      const profKey = new Map();
      profiles.forEach((r) => profKey.set(`${canon(r.state)}||${String(r.district).toUpperCase()}||${r.year}`, r));

      const trendByDistrict = {};
      yoy.forEach((r) => {
        const k = `${canon(r.state)}||${String(r.district).toUpperCase()}`;
        if (!trendByDistrict[k]) trendByDistrict[k] = [];
        trendByDistrict[k].push({ year: Number(r.year), total: Number(r.total_ipc) || 0 });
      });
      Object.values(trendByDistrict).forEach((arr) => arr.sort((a, b) => a.year - b.year));

      const hotspots = [...(data.district_analysis.hotspots_2014 || [])].sort((a, b) => (b.severity_score || 0) - (a.severity_score || 0)).slice(0, 5);
      const rising = [...(data.district_analysis.rising_hotspots || [])].sort((a, b) => (b.avg_yoy_growth_2010_2014 || 0) - (a.avg_yoy_growth_2010_2014 || 0)).slice(0, 5);

      document.getElementById("tab2-hot-danger").innerHTML = hotspots
        .map((r) => `<li>${prettify(r.district)} (${prettify(r.state)}) - ${Number(r.severity_score).toFixed(2)}</li>`)
        .join("");

      document.getElementById("tab2-hot-rising").innerHTML = rising
        .map((r) => `<li>${prettify(r.district)} (${prettify(r.state)}) - ${Number(r.avg_yoy_growth_2010_2014).toFixed(2)}%</li>`)
        .join("");

      let selectedDistrict = null;

      function renderDistrictDetail(state, district, year) {
        if (!district) {
          meta.textContent = "Click a district bar to view detail.";
          return;
        }
        const pk = `${state}||${String(district).toUpperCase()}||${year}`;
        const p = profKey.get(pk);
        if (!p) {
          meta.textContent = `${prettify(district)}: No data available for ${year}.`;
          return;
        }

        const donutData = [Number(p.pct_violent) || 0, Number(p.pct_property) || 0, Number(p.pct_women) || 0];
        if (charts.tab2Donut) charts.tab2Donut.destroy();
        charts.tab2Donut = new Chart(document.getElementById("tab2-donut"), {
          type: "doughnut",
          data: {
            labels: ["Violent", "Property", "Women"],
            datasets: [{ data: donutData, backgroundColor: [palette.profile.violent, palette.profile.property, palette.profile.women] }]
          },
          options: { plugins: { legend: { position: "bottom" } }, maintainAspectRatio: false }
        });

        const trend = trendByDistrict[`${state}||${String(district).toUpperCase()}`] || [];
        if (charts.tab2Spark) charts.tab2Spark.destroy();
        charts.tab2Spark = new Chart(document.getElementById("tab2-spark"), {
          type: "line",
          data: {
            labels: trend.map((x) => x.year),
            datasets: [{ label: "Total IPC", data: trend.map((x) => x.total), borderColor: "#c66a1f", fill: false, tension: 0.25, pointRadius: 1.5 }]
          },
          options: {
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { x: { grid: { display: false } }, y: { grid: { color: "#eceee9" } } }
          }
        });

        meta.textContent = `${prettify(district)}, ${prettify(state)} | Dominant: ${p.dominant_crime_type || "NA"}`;
      }

      function draw() {
        const state = stateSel.value;
        const year = Number(yearSel.value);
        selection.textContent = `${prettify(state)} | ${year}`;

        const rows = severity
          .filter((r) => canon(r.state) === state && Number(r.year) === year)
          .map((r) => ({ district: String(r.district), score: Number(r.severity_score) || 0 }))
          .sort((a, b) => b.score - a.score);

        if (charts.tab2Bar) charts.tab2Bar.destroy();
        charts.tab2Bar = new Chart(document.getElementById("tab2-bar"), {
          type: "bar",
          data: {
            labels: rows.map((r) => r.district),
            datasets: [{
              label: "Severity Score",
              data: rows.map((r) => r.score),
              backgroundColor: rows.map((r) => {
                const p = profKey.get(`${state}||${String(r.district).toUpperCase()}||${year}`);
                const d = p?.dominant_crime_type;
                if (d === "violent") return palette.profile.violent;
                if (d === "property") return palette.profile.property;
                if (d === "women") return palette.profile.women;
                return "#97a39a";
              })
            }]
          },
          options: {
            indexAxis: "y",
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { grid: { color: "#eceee9" } },
              y: { ticks: { autoSkip: false, font: { size: 10 } }, grid: { display: false } }
            },
            onClick: (_, elements) => {
              if (!elements.length) return;
              const idx = elements[0].index;
              selectedDistrict = rows[idx].district;
              renderDistrictDetail(state, selectedDistrict, year);
            }
          }
        });

        renderDistrictDetail(state, selectedDistrict || rows[0]?.district, year);
      }

      stateSel.addEventListener("change", draw);
      yearSel.addEventListener("change", draw);
      draw();
    }

    function setupTab3(ctx) {
      const yearSel = document.getElementById("tab3-year");
      const stateSel = document.getElementById("tab3-state");
      years.forEach((y) => {
        const o = document.createElement("option");
        o.value = y;
        o.textContent = y;
        if (y === 2014) o.selected = true;
        yearSel.appendChild(o);
      });

      const safetyRows = data.women_safety.state_safety_index || [];
      const justiceRows = data.women_safety.justice_pipeline || [];
      const crimeBreakdown = data.women_safety.crime_type_breakdown || [];
      const trend = data.women_safety.national_trend || [];

      const states = Array.from(new Set(justiceRows.map((r) => canon(r.state)))).sort();
      stateSel.innerHTML = states.map((s) => `<option value="${s}">${prettify(s)}</option>`).join("");

      const map = L.map("tab3-map", { zoomControl: true }).setView([22.5, 79], 4.7);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 10 }).addTo(map);
      window.tab3Map = map;
      createLegendControl(map, "Women Safety Index");

      let layer;
      function drawMap() {
        const year = Number(yearSel.value);
        const byState = {};
        safetyRows.filter((r) => Number(r.year) === year).forEach((r) => {
          byState[canon(r.state)] = Number(r.women_safety_index);
        });

        const vals = Object.values(byState);
        const bins = quantileBins(vals, 7);
        renderLegend("tab3-map-legend-body", bins, palette.womenSafety);

        if (layer) map.removeLayer(layer);
        layer = L.geoJSON({ type: "FeatureCollection", features: ctx.geoFeatures }, {
          style: (feature) => {
            const st = ctx.geoToCanonical[feature.properties.NAME_1] || canon(feature.properties.NAME_1);
            const v = byState[st];
            return {
              color: "#33413a",
              weight: 0.8,
              fillColor: colorFor(Number(v), bins, palette.womenSafety),
              fillOpacity: 0.82,
            };
          },
          onEachFeature: (feature, lyr) => {
            const st = ctx.geoToCanonical[feature.properties.NAME_1] || canon(feature.properties.NAME_1);
            const v = byState[st];
            lyr.bindTooltip(`${feature.properties.NAME_1}<br>Women Safety Index: ${Number.isFinite(v) ? v.toFixed(2) : "NA"}`);
          }
        }).addTo(map);
      }

      function drawFunnel() {
        const st = stateSel.value;
        const row = justiceRows.find((r) => canon(r.state) === st);
        const root = document.getElementById("tab3-funnel");
        const drop = document.getElementById("tab3-dropoff");
        if (!row) {
          root.innerHTML = `<div class="no-data">No data available</div>`;
          drop.textContent = "";
          return;
        }
        const reported = Number(row.avg_reported) || 0;
        const steps = [
          ["Reported", Number(row.avg_reported) || 0],
          ["Chargesheeted", Number(row.avg_chargesheeted) || 0],
          ["Trial", Number(row.avg_sent_for_trial) || 0],
          ["Convicted", Number(row.avg_convicted) || 0],
        ];
        root.innerHTML = steps
          .map(([name, val]) => {
            const pct = reported ? (val / reported) * 100 : 0;
            return `<div class="funnel-row"><div>${name}</div><div class="funnel-bar" style="width:${Math.max(2, pct)}%"></div><div>${pct.toFixed(1)}%</div></div>`;
          })
          .join("");

        const chargeDrop = 100 - ((Number(row.chargesheet_rate) || 0) * 100);
        const convictDrop = 100 - ((Number(row.conviction_rate) || 0) * 100);
        drop.textContent = `${prettify(st)}: Chargesheet drop-off ${chargeDrop.toFixed(1)}%, conviction drop-off ${convictDrop.toFixed(1)}%.`;
      }

      function drawStacked() {
        const rows = [...crimeBreakdown].sort((a, b) => canon(a.state).localeCompare(canon(b.state)));
        if (charts.tab3Stack) charts.tab3Stack.destroy();
        charts.tab3Stack = new Chart(document.getElementById("tab3-stack"), {
          type: "bar",
          data: {
            labels: rows.map((r) => prettify(r.state)),
            datasets: [
              { label: "Rape", data: rows.map((r) => r.rape_women), backgroundColor: "#9b1d20" },
              { label: "Dowry", data: rows.map((r) => r.dowry_deaths_women), backgroundColor: "#c74b50" },
              { label: "Kidnapping", data: rows.map((r) => r.kidnapping_women), backgroundColor: "#3c7db3" },
              { label: "Cruelty", data: rows.map((r) => r.domestic_cruelty), backgroundColor: "#8763a7" },
              { label: "Assault", data: rows.map((r) => r.assault_women), backgroundColor: "#e79f40" },
            ]
          },
          options: {
            maintainAspectRatio: false,
            scales: { x: { stacked: true, ticks: { maxRotation: 80, minRotation: 80, font: { size: 9 } } }, y: { stacked: true } },
            plugins: { legend: { position: "bottom" } }
          }
        });
      }

      function drawTrend() {
        if (charts.tab3Trend) charts.tab3Trend.destroy();
        charts.tab3Trend = new Chart(document.getElementById("tab3-trend"), {
          type: "line",
          data: {
            labels: trend.map((r) => r.year),
            datasets: [
              { label: "Total Women Crimes", data: trend.map((r) => r.total_women), borderColor: "#c3382b", tension: 0.25 },
              { label: "Rape", data: trend.map((r) => r.rape_women), borderColor: "#9547a4", tension: 0.25 },
              { label: "Kidnapping", data: trend.map((r) => r.kidnapping_women), borderColor: "#3076ad", tension: 0.25 },
              { label: "Cruelty", data: trend.map((r) => r.domestic_cruelty), borderColor: "#eb8b2f", tension: 0.25 },
            ]
          },
          options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } }
        });
      }

      yearSel.addEventListener("change", drawMap);
      stateSel.addEventListener("change", drawFunnel);
      drawMap();
      drawFunnel();
      drawStacked();
      drawTrend();
    }

    function setupTab4() {
      const stateSel = document.getElementById("tab4-states");
      const metricSel = document.getElementById("tab4-metric");
      const nationalChk = document.getElementById("tab4-national");
      const info = document.getElementById("tab4-model-info");

      const ts = data.forecasts.time_series || [];
      const meta = data.forecasts.model_metadata || [];
      const ntrend = data.national_trends.yearly_totals || [];

      const states = Array.from(new Set(ts.map((r) => canon(r.state)))).sort();
      stateSel.innerHTML = states.map((s, i) => `<option value="${s}" ${i < 2 ? "selected" : ""}>${prettify(s)}</option>`).join("");

      function draw() {
        const selected = getSelectedValues(stateSel).slice(0, 3);
        const metric = metricSel.value;
        const labels = Array.from({ length: 20 }, (_, i) => 2001 + i);

        const datasets = [];
        selected.forEach((state, idx) => {
          const color = palette.states[idx % palette.states.length];
          const actual = labels.map((year) => {
            const row = ts.find((r) => canon(r.state) === state && r.metric === metric && r.type === "actual" && Number(r.year) === year);
            return row ? Number(row.value) : null;
          });
          const pred = labels.map((year) => {
            const row = ts.find((r) => canon(r.state) === state && r.metric === metric && r.type === "predicted" && Number(r.year) === year);
            return row ? Number(row.value) : null;
          });
          datasets.push({ label: `${prettify(state)} (Actual)`, data: actual, borderColor: color, pointRadius: 1.5, tension: 0.25 });
          datasets.push({ label: `${prettify(state)} (Predicted)`, data: pred, borderColor: color, borderDash: [7, 5], pointRadius: 0, tension: 0.25 });
        });

        if (nationalChk.checked) {
          const field = metric === "total_ipc" ? "total_ipc" : "total_women";
          const nat = labels.map((year) => {
            const row = ntrend.find((r) => Number(r.year) === year);
            return row ? Number(row[field]) : null;
          });
          datasets.push({ label: "National Average", data: nat, borderColor: "#333", borderWidth: 2, pointRadius: 0, tension: 0.2 });
        }

        if (charts.tab4Lines) charts.tab4Lines.destroy();
        charts.tab4Lines = new Chart(document.getElementById("tab4-lines"), {
          type: "line",
          data: { labels, datasets },
          options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } }
        });

        const infoText = selected
          .map((st) => {
            const m = meta.find((r) => canon(r.state) === st && r.metric === metric);
            if (!m) return `${prettify(st)}: model metadata unavailable`;
            return `${prettify(st)}: ${m.best_model}, CV MAE ${Number(m.cv_mae).toFixed(2)}`;
          })
          .join(" | ");
        info.textContent = infoText || "Select at least one state.";
      }

      stateSel.addEventListener("change", draw);
      metricSel.addEventListener("change", draw);
      nationalChk.addEventListener("change", draw);
      draw();
    }

    function setupTab5() {
      const radarA = document.getElementById("tab5-radar-a");
      const radarB = document.getElementById("tab5-radar-b");
      const stateSel = document.getElementById("tab5-state");

      const radar = data.crime_profiles.radar_dimensions || [];
      const kidnap = data.supplementary.kidnapping_motives.state_level || [];
      const murder = data.supplementary.murder_demographics.state_level || [];
      const geo = data.supplementary.crime_geography.state_level || [];
      const recovery = data.supplementary.property_recovery.state_level || [];

      const states = Array.from(new Set(radar.map((r) => canon(r.state)))).sort();
      const options = states.map((s, i) => `<option value="${s}" ${i === 0 ? "selected" : ""}>${prettify(s)}</option>`).join("");
      radarA.innerHTML = options;
      radarB.innerHTML = states.map((s, i) => `<option value="${s}" ${i === 1 ? "selected" : ""}>${prettify(s)}</option>`).join("");
      stateSel.innerHTML = options;

      const radarAxes = ["Violent", "Property", "Women", "Public Order", "White Collar"];

      function rowBy(arr, st) {
        return arr.find((r) => canon(r.state) === st);
      }

      function drawRadar() {
        const a = rowBy(radar, radarA.value);
        const b = rowBy(radar, radarB.value);
        if (charts.tab5Radar) charts.tab5Radar.destroy();
        charts.tab5Radar = new Chart(document.getElementById("tab5-radar"), {
          type: "radar",
          data: {
            labels: radarAxes,
            datasets: [
              { label: prettify(radarA.value), data: radarAxes.map((k) => Number(a?.[k]) || 0), borderColor: "#c3382b", backgroundColor: "rgba(195,56,43,0.12)" },
              { label: prettify(radarB.value), data: radarAxes.map((k) => Number(b?.[k]) || 0), borderColor: "#2c6fa8", backgroundColor: "rgba(44,111,168,0.12)" },
            ]
          },
          options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { r: { beginAtZero: true } } }
        });
      }

      function drawStateDetail() {
        const st = stateSel.value;

        const k = rowBy(kidnap, st);
        const kLabels = ["Marriage", "Ransom", "Prostitution", "Illicit", "Unlawful", "Other"];
        const kVals = [
          Number(k?.kidnap_cases_marriage) || 0,
          Number(k?.kidnap_cases_ransom) || 0,
          Number(k?.kidnap_cases_prostitution) || 0,
          Number(k?.kidnap_cases_illicit_intercourse) || 0,
          Number(k?.kidnap_cases_unlawful_activity) || 0,
          Number(k?.kidnap_cases_other_purposes) || 0,
        ];
        if (charts.tab5Kidnap) charts.tab5Kidnap.destroy();
        charts.tab5Kidnap = new Chart(document.getElementById("tab5-kidnap"), {
          type: "doughnut",
          data: { labels: kLabels, datasets: [{ data: kVals, backgroundColor: ["#8a1c1f", "#be4928", "#db7a2f", "#4f7ca7", "#5f8f56", "#a1a9b3"] }] },
          options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } }
        });

        const g = rowBy(geo, st);
        const gLabels = ["Residential", "Highway", "Railway", "Bank", "Commercial", "Other"];
        const gVals = [
          Number(g?.residential) || 0,
          Number(g?.highway) || 0,
          Number(g?.railway) || 0,
          Number(g?.bank) || 0,
          Number(g?.commercial) || 0,
          Number(g?.other_places) || 0,
        ];
        if (charts.tab5Geo) charts.tab5Geo.destroy();
        charts.tab5Geo = new Chart(document.getElementById("tab5-geo"), {
          type: "doughnut",
          data: { labels: gLabels, datasets: [{ data: gVals, backgroundColor: ["#376da2", "#d4672a", "#8f5ea5", "#c83c3c", "#548f58", "#a4adb7"] }] },
          options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } }
        });

        const m = rowBy(murder, st);
        const femalePct = Number(m?.female_pct) || 0;
        const maleShare = 100 / (100 + femalePct);
        const femaleShare = femalePct / (100 + femalePct);
        const ageKeys = [
          ["Up to 10", "murder_age_upto_10"],
          ["10-15", "murder_age_10_15"],
          ["15-18", "murder_age_15_18"],
          ["18-30", "murder_age_18_30"],
          ["30-50", "murder_age_30_50"],
          ["50+", "murder_age_above_50"],
        ];
        if (charts.tab5Murder) charts.tab5Murder.destroy();
        charts.tab5Murder = new Chart(document.getElementById("tab5-murder"), {
          type: "bar",
          data: {
            labels: ["Male", "Female"],
            datasets: ageKeys.map(([label, key], idx) => {
              const base = Number(m?.[key]) || 0;
              return {
                label,
                data: [base * maleShare, base * femaleShare],
                backgroundColor: palette.states[idx % palette.states.length],
              };
            })
          },
          options: {
            maintainAspectRatio: false,
            scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
            plugins: { legend: { position: "bottom" } }
          }
        });

        const r = rowBy(recovery, st);
        const rate = (Number(r?.avg_recovery_rate) || 0) * 100;
        document.getElementById("tab5-recovery-text").textContent = `${prettify(st)} recovery rate: ${rate.toFixed(1)}%`;
        document.getElementById("tab5-recovery-fill").style.width = `${Math.max(0, Math.min(100, rate))}%`;
        document.getElementById("tab5-metrics").innerHTML = `
          <div class="metric"><div class="k">Total Stolen</div><div class="v">${fmt.format(Math.round(Number(r?.total_stolen) || 0))}</div></div>
          <div class="metric"><div class="k">Total Recovered</div><div class="v">${fmt.format(Math.round(Number(r?.total_recovered) || 0))}</div></div>
          <div class="metric"><div class="k">Recovery %</div><div class="v">${rate.toFixed(1)}%</div></div>
        `;
      }

      radarA.addEventListener("change", drawRadar);
      radarB.addEventListener("change", drawRadar);
      stateSel.addEventListener("change", drawStateDetail);
      drawRadar();
      drawStateDetail();
    }

    function setupTab6() {
      const ts = data.forecasts.time_series || [];
      const assign = data.clusters.state_assignments || [];
      const summaries = data.clusters.cluster_summaries || [];
      const findings = data.key_findings || [];

      const actualIPC = ts.filter((r) => r.metric === "total_ipc" && r.type === "actual");
      const byState = {};
      actualIPC.forEach((r) => {
        const st = canon(r.state);
        if (!byState[st]) byState[st] = [];
        byState[st].push({ year: Number(r.year), value: Number(r.value) || 0 });
      });
      Object.values(byState).forEach((arr) => arr.sort((a, b) => a.year - b.year));

      const assignByState = {};
      assign.forEach((r) => { assignByState[canon(r.state)] = r; });

      const clusterMap = {};
      Object.entries(byState).forEach(([st, arr]) => {
        if (!arr.length) return;
        const avg = arr.reduce((a, b) => a + b.value, 0) / arr.length;
        let yoyTotal = 0;
        let yoyCount = 0;
        for (let i = 1; i < arr.length; i++) {
          if (arr[i - 1].value > 0) {
            yoyTotal += ((arr[i].value - arr[i - 1].value) / arr[i - 1].value) * 100;
            yoyCount += 1;
          }
        }
        const yoy = yoyCount ? yoyTotal / yoyCount : 0;
        const total = arr.reduce((a, b) => a + b.value, 0);
        const meta = assignByState[st];
        const cid = meta?.cluster_id ?? -1;
        if (!clusterMap[cid]) clusterMap[cid] = { label: meta?.cluster_label || "Unclustered", points: [] };
        clusterMap[cid].points.push({ x: avg, y: yoy, r: Math.max(4, Math.sqrt(total) / 140), state: prettify(st), total });
      });

      const datasets = Object.entries(clusterMap).map(([cid, obj], i) => ({
        label: obj.label,
        data: obj.points,
        backgroundColor: palette.states[i % palette.states.length],
      }));

      if (charts.tab6Scatter) charts.tab6Scatter.destroy();
      charts.tab6Scatter = new Chart(document.getElementById("tab6-scatter"), {
        type: "bubble",
        data: { datasets },
        options: {
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "bottom" },
            tooltip: {
              callbacks: {
                label: (ctx) => `${ctx.raw.state}: avg ${fmt.format(Math.round(ctx.raw.x))}, YoY ${ctx.raw.y.toFixed(2)}%, total ${fmt.format(Math.round(ctx.raw.total))}`
              }
            }
          },
          scales: {
            x: { title: { display: true, text: "Average Crime Rate (IPC)" } },
            y: { title: { display: true, text: "Average YoY Growth %" } },
          }
        }
      });

      const membersByCluster = {};
      assign.forEach((r) => {
        const id = r.cluster_id;
        if (!membersByCluster[id]) membersByCluster[id] = [];
        membersByCluster[id].push(prettify(r.state));
      });

      document.getElementById("tab6-clusters").innerHTML = summaries.map((s) => {
        const members = (membersByCluster[s.cluster_id] || []).slice(0, 8).join(", ");
        return `
          <div class="cluster-card">
            <h4>${s.cluster_label}</h4>
            <div class="state-line">States: ${s.state_count}</div>
            <div class="state-line">Avg IPC: ${fmt.format(Math.round(Number(s.avg_ipc_crimes) || 0))}</div>
            <div class="state-line">Avg Women Crime: ${fmt.format(Math.round(Number(s.avg_crimes_women) || 0))}</div>
            <div class="state-line">Key: recovery ${(Number(s.avg_recovery_rate) * 100).toFixed(1)}%, firearms ${Number(s.avg_firearms_murder).toFixed(1)}</div>
            <div class="state-line">Members: ${members || "NA"}</div>
          </div>`;
      }).join("");

      document.getElementById("tab6-findings").innerHTML = findings.map((f) => `<li>${f}</li>`).join("");
    }

    function run() {
      initializeTabs();
      document.getElementById("as-of").textContent = `Generated ${new Date().toLocaleString()}`;

      const ctx = preprocess();
      setupTab1(ctx);
      setupTab2();
      setupTab3(ctx);
      setupTab4();
      setupTab5();
      setupTab6();

      setTimeout(() => {
        const loader = document.getElementById("global-loader");
        if (loader) loader.style.display = "none";
      }, 300);
    }

    run();
  </script>
</body>
</html>
"""


def build_dashboard_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    for name in REQUIRED_JSON_FILES:
        path = DASHBOARD_DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Required dashboard input missing: {path}")
        payload[path.stem] = load_json(path)

    geojson_path = DATA_DIR / "india_states.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
    payload["geojson"] = load_json(geojson_path)

    payload["key_findings"] = build_key_findings(payload)
    return payload


def render_html(payload: Dict[str, Any]) -> str:
    data_js = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return DASHBOARD_TEMPLATE.replace("__DASHBOARD_DATA__", data_js)


def main() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    payload = build_dashboard_payload()
    html = render_html(payload)

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Dashboard generated: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
