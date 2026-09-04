#!/usr/bin/env python3
"""W&B-style interactive metrics dashboard for residual-RL TensorBoard logs.

Serves a self-contained single-page app (no external JS/CDN) that reads the
event files on every request, so refreshing shows fresh data from an
in-progress run. Works even when the full tensorboard binary is broken by
plugin conflicts: only tensorboard.backend.event_processing is used.

Features: per-run toggles, metric search, visible chart titles, hover
tooltips, EMA smoothing slider, auto-refresh interval selection.

Usage:
    python3 training_dashboard.py --logdir runs/residual-rl --port 8765
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def collect_scalars(logdir: Path) -> dict[str, dict[str, list[float]]]:
    """Return {tag: {run_label: [step0, value0, step1, value1, ...]}}."""
    runs: dict[str, Path] = {}
    for event_file in sorted(logdir.rglob("events.out.tfevents.*")):
        parts = event_file.relative_to(logdir).parts
        run_label = parts[0] if len(parts) > 1 else str(event_file.parent)
        runs.setdefault(run_label, event_file)

    all_tags: dict[str, dict[str, list[float]]] = {}
    for run_label, event_file in runs.items():
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:  # noqa: BLE001 - keep serving other runs
            print(f"[dashboard] skipping {event_file}: {exc}", flush=True)
            continue
        for tag in accumulator.Tags().get("scalars", []):
            flat: list[float] = []
            last_step = None
            for record in accumulator.Scalars(tag):
                if last_step is not None and record.step <= last_step:
                    continue  # keep strictly increasing steps
                flat.extend([float(record.step), float(record.value)])
                last_step = record.step
            all_tags.setdefault(tag, {})[run_label] = flat

    return all_tags


def render_page() -> str:
    return r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Residual PPO — training metrics</title>
<style>
  :root {
    --bg: #0f1117; --panel: #161a23; --panel2: #1c2130; --border: #2a3040;
    --text: #d5dbe8; --muted: #8b94a7; --accent: #4f8ef7; --accent2: #9b6ff2;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); margin: 0;
         font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; align-items: center; gap: 16px; padding: 12px 20px;
           background: var(--panel); border-bottom: 1px solid var(--border);
           position: sticky; top: 0; z-index: 10; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  header .sub { color: var(--muted); font-size: 12px; }
  #logdir { color: var(--muted); font-size: 12px; margin-left: auto; }
  .layout { display: flex; min-height: calc(100vh - 49px); }
  aside { width: 260px; flex-shrink: 0; background: var(--panel);
          border-right: 1px solid var(--border); padding: 14px; overflow-y: auto; }
  aside h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
             color: var(--muted); margin: 14px 0 8px; }
  aside h2:first-child { margin-top: 0; }
  .run-row { display: flex; align-items: center; gap: 8px; padding: 5px 6px;
             border-radius: 6px; cursor: pointer; font-size: 13px; user-select: none; }
  .run-row:hover { background: var(--panel2); }
  .swatch { width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }
  .run-meta { color: var(--muted); font-size: 11px; margin-left: auto; }
  input[type="checkbox"] { accent-color: var(--accent); }
  .btnrow { display: flex; gap: 6px; }
  .btn { background: var(--panel2); color: var(--text); border: 1px solid var(--border);
         border-radius: 6px; padding: 4px 8px; font-size: 12px; cursor: pointer; flex: 1; }
  .btn:hover { border-color: var(--accent); }
  select, input[type="search"], input[type="range"] { width: 100%; }
  select, input[type="search"] { background: var(--panel2); color: var(--text);
         border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font-size: 13px; }
  label.small { font-size: 12px; color: var(--muted); display: block; margin-top: 4px; }
  main { flex: 1; padding: 16px 20px; overflow-x: hidden; }
  .group-title { font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
                 color: var(--muted); margin: 18px 0 8px; border-bottom: 1px solid var(--border);
                 padding-bottom: 4px; }
  .chart-card { background: var(--panel); border: 1px solid var(--border);
                border-radius: 10px; margin-bottom: 14px; overflow: hidden; }
  .chart-head { display: flex; align-items: baseline; gap: 12px; padding: 10px 14px 6px; }
  .chart-title { font-size: 13.5px; font-weight: 600; color: var(--text);
                 font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  .chart-stats { color: var(--muted); font-size: 11.5px; margin-left: auto; }
  .chart-body { position: relative; padding: 0 6px 8px; }
  canvas { display: block; width: 100%; height: 220px; cursor: crosshair; }
  #tooltip { position: fixed; pointer-events: none; background: #0b0d12ee; border: 1px solid var(--border);
             border-radius: 8px; padding: 8px 10px; font-size: 12px; display: none; z-index: 50;
             max-width: 340px; }
  #tooltip .tt-step { color: var(--muted); margin-bottom: 4px; font-family: ui-monospace, monospace; }
  #tooltip .tt-row { display: flex; align-items: center; gap: 6px; padding: 1px 0;
                     font-family: ui-monospace, monospace; }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
  .badge { background: var(--panel2); border: 1px solid var(--border); border-radius: 999px;
           padding: 1px 8px; font-size: 11px; color: var(--muted); }
</style></head>
<body>
<header>
  <h1>Residual PPO — training metrics</h1>
  <span class="sub" id="status">loading…</span>
  <span id="logdir"></span>
</header>
<div class="layout">
  <aside>
    <h2>Runs</h2>
    <div class="btnrow" style="margin-bottom:8px">
      <button class="btn" id="all">All</button>
      <button class="btn" id="none">None</button>
      <button class="btn" id="latest">Latest</button>
    </div>
    <div id="runs"></div>

    <h2>Metrics</h2>
    <div class="btnrow" style="margin-bottom:8px">
      <button class="btn" id="mall">All</button>
      <button class="btn" id="mnone">None</button>
    </div>
    <input type="search" id="filter" placeholder="filter metric name…" autocomplete="off">
    <div id="metrics"></div>
    <label class="small">Group:
      <select id="groupby">
        <option value="auto">auto (train/rollout/time)</option>
        <option value="none">flat list</option>
      </select>
    </label>

    <h2>Display</h2>
    <label class="small">Smoothing (EMA α): <span id="smoothval">0</span></label>
    <input type="range" id="smooth" min="0" max="95" value="0">
    <label class="small">Auto-refresh:
      <select id="refresh">
        <option value="0">off</option>
        <option value="2" selected>2 s</option>
        <option value="5">5 s</option>
        <option value="10">10 s</option>
        <option value="30">30 s</option>
      </select>
    </label>
    <button class="btn" id="refreshnow" style="margin-top:8px">Refresh now</button>
  </aside>
  <main><div id="charts"></div></main>
</div>
<div id="tooltip"></div>

<script>
"use strict";
const PALETTE = ["#4f8ef7","#3fb950","#d29922","#bc8cff","#f778ba","#ff7b72",
                 "#39c5cf","#e3b341","#7ee787","#db61a2","#a5d6ff","#ffa657"];
let DATA = null;          // {logdir, runs:[{name,tags:{tag:[s,v,...]}}]}
let enabledRuns = new Set();
let lastRunKey = "";      // membership key to avoid needless list rebuilds
let enabledTags = new Set();  // which metric charts are shown (user-selectable)
let seenTags = new Set();     // tags present in prior fetches (new ones default ON)
let filterText = "";
let smoothAlpha = 0;      // 0 = off; slider maps to alpha = 1 - v/100
let refreshTimer = null;

function $(id){ return document.getElementById(id); }

async function fetchData(){
  const res = await fetch("/api/data", {cache: "no-store"});
  DATA = await res.json();
  $("logdir").textContent = "logdir: " + DATA.logdir;
  if (enabledRuns.size === 0) enabledRuns = new Set(DATA.runs.map(r=>r.name));
  // Track all tags; brand-new tags default to visible.
  const allTags = new Set();
  DATA.runs.forEach(r => Object.keys(r.tags).forEach(t=>allTags.add(t)));
  let tagsChanged = false;
  allTags.forEach(t => { if (!seenTags.has(t)) { enabledTags.add(t); seenTags.add(t); tagsChanged = true; } });
  const runKey = DATA.runs.map(r=>r.name).join("|");
  if (runKey !== lastRunKey) { lastRunKey = runKey; renderRunList(); }
  if (tagsChanged) renderMetricList();
  renderCharts();
  const total = DATA.runs.reduce((a,r)=>a+r.points,0);
  $("status").textContent = DATA.runs.length + " runs · " + total + " points · " +
    new Date().toLocaleTimeString();
}

function renderRunList(){
  const box = $("runs"); box.innerHTML = "";
  DATA.runs.forEach((r, i) => {
    const row = document.createElement("label");
    row.className = "run-row";
    row.innerHTML = '<input type="checkbox" ' + (enabledRuns.has(r.name)?"checked":"") + '>' +
      '<span class="swatch" style="background:' + PALETTE[i % PALETTE.length] + '"></span>' +
      '<span>' + r.name + '</span><span class="run-meta">' + r.points + '</span>';
    row.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) enabledRuns.add(r.name); else enabledRuns.delete(r.name);
      renderCharts();
    });
    box.appendChild(row);
  });
}

function smoothSeries(flat, alpha){
  if (!alpha || flat.length < 4) return flat;
  const out = [flat[0], flat[1]];
  for (let i = 2; i < flat.length; i += 2) {
    out.push(flat[i]);
    out.push(alpha * flat[i-1] + (1 - alpha) * flat[i+1]);
  }
  return out;
}

function renderMetricList(){
  const box = $("metrics"); box.innerHTML = "";
  const f = filterText.toLowerCase();
  const tags = [...seenTags].filter(t => !f || t.toLowerCase().includes(f)).sort();
  if (!tags.length) { box.innerHTML = '<div class="empty" style="padding:8px;font-size:12px">no metrics</div>'; return; }
  tags.forEach(t => {
    const row = document.createElement("label");
    row.className = "run-row";
    row.style.fontSize = "12px";
    row.innerHTML = '<input type="checkbox" ' + (enabledTags.has(t)?"checked":"") + '>' +
      '<span style="font-family:ui-monospace,monospace">' + t + '</span>';
    row.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) enabledTags.add(t); else enabledTags.delete(t);
      renderCharts();
    });
    box.appendChild(row);
  });
}

function visibleTags(){
  const tags = new Set();
  DATA.runs.forEach(r => { if (enabledRuns.has(r.name)) Object.keys(r.tags).forEach(t=>tags.add(t)); });
  const f = filterText.toLowerCase();
  return [...tags].filter(t => enabledTags.has(t) && (!f || t.toLowerCase().includes(f))).sort();
}

function groupOf(tag){
  const mode = $("groupby").value;
  if (mode === "none") return "";
  const prefix = tag.split("/")[0];
  return ["train","rollout","time"].includes(prefix) ? prefix : "other";
}

const chartRegistry = [];   // {canvas, tag, series:[{name,color,flat}], minS,maxS,minV,maxV}

function renderCharts(){
  const main = $("charts"); main.innerHTML = ""; chartRegistry.length = 0;
  const tags = visibleTags();
  if (!tags.length) {
    const anyData = [...seenTags].some(t => DATA.runs.some(r => enabledRuns.has(r.name) && r.tags[t]));
    main.innerHTML = '<div class="empty">' + (anyData ? "All metrics are hidden — tick some in the Metrics panel." : "No metrics to show — enable a run or clear the filter.") + '</div>';
    return;
  }

  const groups = {};
  tags.forEach(t => { (groups[groupOf(t)] = groups[groupOf(t)] || []).push(t); });
  const order = ["train","rollout","time","other",""];
  for (const g of order) {
    if (!groups[g]) continue;
    if (g) { const h = document.createElement("div"); h.className="group-title"; h.textContent=g; main.appendChild(h); }
    for (const tag of groups[g]) main.appendChild(makeChart(tag));
  }
  // All cards are now in the DOM. Wait one frame so layout/widths are real,
  // then draw every chart. (Drawing before insertion gives width 0.)
  requestAnimationFrame(() => {
    for (const reg of chartRegistry) {
      if (reg.canvas.parentElement && reg.canvas.parentElement.style.display !== "none") drawChart(reg);
    }
  });
}

function makeChart(tag){
  const card = document.createElement("div"); card.className = "chart-card";
  const head = document.createElement("div"); head.className = "chart-head";
  const title = document.createElement("span"); title.className = "chart-title"; title.textContent = tag;
  const stats = document.createElement("span"); stats.className = "chart-stats";
  head.appendChild(title); head.appendChild(stats);
  const body = document.createElement("div"); body.className = "chart-body";
  const canvas = document.createElement("canvas");
  card.appendChild(head); card.appendChild(body); body.appendChild(canvas);

  const series = [];
  let minS=Infinity, maxS=-Infinity, minV=Infinity, maxV=-Infinity;
  DATA.runs.forEach((r, i) => {
    if (!enabledRuns.has(r.name)) return;
    const flat = r.tags[tag]; if (!flat || flat.length < 2) return;
    const s = smoothSeries(flat, smoothAlpha);
    series.push({name: r.name, color: PALETTE[i % PALETTE.length], flat: s});
    for (let k=0;k<s.length;k+=2){
      if (s[k]<minS) minS=s[k]; if (s[k]>maxS) maxS=s[k];
      if (s[k+1]<minV) minV=s[k+1]; if (s[k+1]>maxV) maxV=s[k+1];
    }
  });
  stats.textContent = series.length + " run" + (series.length===1?"":"s") +
    (series.length ? " · step " + Math.round(minS) + "–" + Math.round(maxS) : "");
  if (!series.length) { card.style.display="none"; return card; }
  if (maxV - minV < 1e-12){ minV -= .5; maxV += .5; }

  const reg = {canvas, tag, series, minS, maxS, minV, maxV};
  chartRegistry.push(reg);
  canvas.addEventListener("mousemove", (e) => onHover(e, reg));
  canvas.addEventListener("mouseleave", () => $("tooltip").style.display="none");
  // Do NOT draw here: the card is still off-DOM, so getBoundingClientRect()
  // would report width 0 and lines would be invisible. renderCharts() draws
  // everything once after all cards are inserted into <main>.
  return card;
}

function setupCanvas(canvas){
  const dpr = window.devicePixelRatio || 1;
  let rect = canvas.getBoundingClientRect();
  // Fallback if layout hasn't produced a real width yet (e.g. hidden/just-inserted).
  if (!rect.width) rect = {width: Math.max(320, window.innerWidth - 300), height: 220};
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, w: rect.width, h: rect.height};
}

function niceTicks(min, max, count){
  if (!isFinite(min) || !isFinite(max)) return [];
  if (min === max) return [min];
  const span = max - min;
  const rawStep = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  let step;
  if (norm < 1.5) step = mag;
  else if (norm < 3.5) step = 2 * mag;
  else if (norm < 7.5) step = 5 * mag;
  else step = 10 * mag;
  // round to enough decimals for the step so small values aren't zeroed out
  const decimals = Math.max(0, -Math.floor(Math.log10(step)) + 1);
  const start = Math.ceil(min / step) * step;
  const n = Math.floor((max - start) / step) + 1;
  const ticks = [];
  for (let i = 0; i < n; i++) ticks.push(+(start + i * step).toFixed(decimals));
  return ticks;
}

function drawChart(reg){
  const {ctx,w,h} = setupCanvas(reg.canvas);
  const padL=64, padR=12, padT=10, padB=28;
  const pw=w-padL-padR, ph=h-padT-padB;
  ctx.clearRect(0,0,w,h);
  const X = s => padL + (s-reg.minS)/((reg.maxS-reg.minS)||1)*pw;
  const Y = v => padT + (1-(v-reg.minV)/((reg.maxV-reg.minV)||1))*ph;
  // grid + axes (round-number ticks at their true data positions)
  ctx.font="11px ui-monospace, monospace"; ctx.fillStyle="#8b94a7"; ctx.strokeStyle="#232936"; ctx.lineWidth=1;
  for (const v of niceTicks(reg.minV, reg.maxV, 5)){
    const y = Y(v); if (y < padT-1 || y > h-padB+1) continue;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(w-padR,y); ctx.stroke();
    ctx.textAlign="right"; ctx.fillText(fmtNum(v), padL-8, y+4);
  }
  for (const s of niceTicks(reg.minS, reg.maxS, 6)){
    const x = X(s); if (x < padL-1 || x > w-padR+1) continue;
    ctx.beginPath(); ctx.moveTo(x,padT); ctx.lineTo(x,h-padB); ctx.stroke();
    ctx.textAlign="center"; ctx.fillText(String(s), x, h-padB+16);
  }
  for (const sr of reg.series){
    ctx.beginPath(); ctx.strokeStyle=sr.color; ctx.lineWidth=1.8;
    let started=false;
    for (let k=0;k<sr.flat.length;k+=2){
      const x=X(sr.flat[k]), y=Y(sr.flat[k+1]);
      if(!started){ctx.moveTo(x,y);started=true;} else ctx.lineTo(x,y);
    }
    ctx.stroke();
  }
  reg._X = X; reg._Y = Y; reg._padL=padL; reg._pw=pw;
}

function fmtNum(v){
  if (!isFinite(v)) return "";
  const a=Math.abs(v);
  if (a>=1000) return v.toFixed(0);
  if (a>=10) return v.toFixed(2);
  if (a>=0.01) return v.toFixed(4);
  return v.toExponential(2);
}

function onHover(e, reg){
  const rect = reg.canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  if (mx < reg._padL || mx > rect.width - 12) { $("tooltip").style.display="none"; return; }
  const step = reg.minS + (mx-reg._padL)/reg._pw*(reg.maxS-reg.minS);
  // nearest point per series
  const rows = [];
  for (const sr of reg.series){
    let best=-1, bd=Infinity;
    for (let k=0;k<sr.flat.length;k+=2){
      const d=Math.abs(sr.flat[k]-step); if(d<bd){bd=d;best=k;}
    }
    if (best>=0) rows.push({name:sr.name,color:sr.color,step:sr.flat[best],value:sr.flat[best+1]});
  }
  rows.sort((a,b)=>b.value-a.value);
  const tt=$("tooltip");
  tt.innerHTML = '<div class="tt-step">step ≈ ' + Math.round(step) + '</div>' +
    rows.map(r=>'<div class="tt-row"><span class="swatch" style="background:'+r.color+'"></span>'+
      r.name + ': <b>' + fmtNum(r.value) + '</b> <span style="color:#8b94a7">(s ' + Math.round(r.step) + ')</span></div>').join("");
  tt.style.display="block";
  const pad=14;
  let left=e.clientX+pad, top=e.clientY+pad;
  if (left + tt.offsetWidth > window.innerWidth-8) left = e.clientX - tt.offsetWidth - pad;
  if (top + tt.offsetHeight > window.innerHeight-8) top = e.clientY - tt.offsetHeight - pad;
  tt.style.left=left+"px"; tt.style.top=top+"px";
}

function setRefresh(sec){
  if (refreshTimer){ clearInterval(refreshTimer); refreshTimer=null; }
  if (sec>0) refreshTimer=setInterval(fetchData, sec*1000);
}

$("all").onclick = () => { enabledRuns=new Set(DATA.runs.map(r=>r.name)); renderRunList(); renderCharts(); };
$("none").onclick = () => { enabledRuns.clear(); renderRunList(); renderCharts(); };
$("latest").onclick = () => { enabledRuns=new Set([DATA.runs[DATA.runs.length-1].name]); renderRunList(); renderCharts(); };
$("filter").addEventListener("input", e => { filterText=e.target.value; renderMetricList(); renderCharts(); });
$("mall").onclick = () => { seenTags.forEach(t=>enabledTags.add(t)); renderMetricList(); renderCharts(); };
$("mnone").onclick = () => { enabledTags.clear(); renderMetricList(); renderCharts(); };
$("groupby").addEventListener("change", renderCharts);
$("smooth").addEventListener("input", e => {
  const v=+e.target.value; $("smoothval").textContent=v;
  smoothAlpha = v===0 ? 0 : 1 - v/100;
  renderCharts();
});
$("refresh").addEventListener("change", e => setRefresh(+e.target.value));
$("refreshnow").onclick = fetchData;
window.addEventListener("resize", () => chartRegistry.forEach(drawChart));

fetchData().catch(err => { $("status").textContent="failed to load: "+err; });
setRefresh(2);
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, default=Path("runs/residual-rl"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/data":
                scalars = collect_scalars(args.logdir)
                run_names: list[str] = []
                for tag_points in scalars.values():
                    for name in tag_points:
                        if name not in run_names:
                            run_names.append(name)
                runs: list[dict] = []
                for name in sorted(run_names):
                    tags = {tag: pts[name] for tag, pts in scalars.items() if name in pts}
                    runs.append({"name": name, "tags": tags, "points": sum(len(v)//2 for v in tags.values())})
                payload = json.dumps({"logdir": str(args.logdir), "runs": runs}).encode("utf-8")
                self._send(200, payload, "application/json")
            else:
                self._send(200, render_page().encode("utf-8"), "text/html; charset=utf-8")

        def log_message(self, format: str, *args) -> None:  # quiet access log
            return

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[dashboard] serving {args.logdir} on port {args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
