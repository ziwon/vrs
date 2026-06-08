/*
 * VRS Console — sample dashboard logic.
 *
 * Renders VRS cascade output (alerts.jsonl-shaped records) in a VSS-styled shell.
 * When served by docker-compose it reads the FastAPI backend. The embedded sample
 * data remains a file:// and backend-unavailable fallback.
 */
"use strict";

const CONFIG = window.VRS_CONFIG || {};
const API_BASE = CONFIG.apiBaseUrl ?? "";
const DEFAULT_RUN = CONFIG.defaultRun ?? "fixture_multi";
const POLL_MS = Number(CONFIG.pollMs ?? 3000);

/* ───────────────────────── Icons (Tabler-style, 24×24, currentColor) ───────────────────────── */
const ICON = {
  alert: '<path d="M12 9v4"/><path d="M10.24 3.957l-8.422 14.06a1.989 1.989 0 0 0 1.7 2.983h16.845a1.989 1.989 0 0 0 1.7 -2.983l-8.423 -14.06a1.989 1.989 0 0 0 -3.4 0z"/><path d="M12 16h.01"/>',
  cascade: '<path d="M12 3l-8 4.5l8 4.5l8 -4.5l-8 -4.5"/><path d="M4 12l8 4.5l8 -4.5"/><path d="M4 16.5l8 4.5l8 -4.5"/>',
  video: '<path d="M15 10l4.553 -2.276a1 1 0 0 1 1.447 .894v6.764a1 1 0 0 1 -1.447 .894l-4.553 -2.276v-4z"/><rect x="3" y="6" width="12" height="12" rx="2"/>',
  policy: '<path d="M9 5h-2a2 2 0 0 0 -2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2 -2v-12a2 2 0 0 0 -2 -2h-2"/><path d="M9 3m0 2a2 2 0 0 1 2 -2h2a2 2 0 0 1 2 2v0a2 2 0 0 1 -2 2h-2a2 2 0 0 1 -2 -2z"/><path d="M9 12l.01 0"/><path d="M13 12l2 0"/><path d="M9 16l.01 0"/><path d="M13 16l2 0"/>',
  sun: '<path d="M12 12m-4 0a4 4 0 1 0 8 0a4 4 0 1 0 -8 0"/><path d="M3 12h1m8 -9v1m8 8h1m-9 8v1m-6.4 -15.4l.7 .7m12.1 -.7l-.7 .7m0 11.4l.7 .7m-12.1 -.7l-.7 .7"/>',
  moon: '<path d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 0 0 7.92 12.446a9 9 0 1 1 -8.313 -12.454z"/>',
  refresh: '<path d="M20 11a8.1 8.1 0 0 0 -15.5 -2m-.5 -4v4h4"/><path d="M4 13a8.1 8.1 0 0 0 15.5 2m.5 4v-4h-4"/>',
  x: '<path d="M18 6l-12 12"/><path d="M6 6l12 12"/>',
  pin: '<path d="M9 11a3 3 0 1 0 6 0a3 3 0 0 0 -6 0"/><path d="M17.657 16.657l-4.243 4.243a2 2 0 0 1 -2.827 0l-4.244 -4.243a8 8 0 1 1 11.314 0z"/>',
  cpu: '<path d="M5 5m0 1a1 1 0 0 1 1 -1h12a1 1 0 0 1 1 1v12a1 1 0 0 1 -1 1h-12a1 1 0 0 1 -1 -1z"/><path d="M9 9h6v6h-6z"/><path d="M3 10h2"/><path d="M3 14h2"/><path d="M10 3v2"/><path d="M14 3v2"/><path d="M21 10h-2"/><path d="M21 14h-2"/><path d="M10 21v-2"/><path d="M14 21v-2"/>',
  bolt: '<path d="M13 3l0 7l6 0l-8 11l0 -7l-6 0l8 -11"/>',
  shield: '<path d="M11.46 20.846a12 12 0 0 1 -7.96 -14.846a12 12 0 0 0 8.5 -3a12 12 0 0 0 8.5 3a12 12 0 0 1 -.09 7.06"/><path d="M15 19l2 2l4 -4"/>',
  camera: '<path d="M5 7h1a2 2 0 0 0 2 -2a1 1 0 0 1 1 -1h6a1 1 0 0 1 1 1a2 2 0 0 0 2 2h1a2 2 0 0 1 2 2v9a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2v-9a2 2 0 0 1 2 -2"/><path d="M9 13a3 3 0 1 0 6 0a3 3 0 0 0 -6 0"/>',
  chevron: '<path d="M9 6l6 6l-6 6"/>',
  activity: '<path d="M3 12h4l3 8l4 -16l3 8h4"/>',
};
const icon = (name, size = 18, cls = "") =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="${cls}">${ICON[name] || ""}</svg>`;

/* ───────────────────────── Watch policy (configs/policies/safety.yaml) ───────────────────────── */
const DEFAULT_POLICY = [
  { name: "fire", severity: "critical", min_score: 0.3, min_persist_frames: 2,
    detector: ["fire", "open flame", "burning object"],
    verifier: "Open flames or active fire indoors that pose a safety risk" },
  { name: "smoke", severity: "high", min_score: 0.25, min_persist_frames: 2,
    detector: ["smoke", "smoke cloud", "billowing smoke"],
    verifier: "Thick smoke filling a space, indicating fire or fault" },
  { name: "falldown", severity: "high", min_score: 0.35, min_persist_frames: 3,
    detector: ["person lying on the ground", "fallen person", "collapsed person"],
    verifier: "A person has collapsed and is lying motionless on the floor" },
  { name: "weapon", severity: "critical", min_score: 0.4, min_persist_frames: 1,
    detector: ["handgun", "knife", "rifle"],
    verifier: "A person is visibly holding or brandishing a weapon" },
];
let POLICY = DEFAULT_POLICY.slice();
let policyByName = Object.fromEntries(POLICY.map((p) => [p.name, p]));

function setPolicy(entries) {
  if (!Array.isArray(entries) || !entries.length) return;
  POLICY = entries.map((p) => ({
    name: String(p.name || ""),
    severity: String(p.severity || "info"),
    min_score: Number(p.min_score ?? 0),
    min_persist_frames: Number(p.min_persist_frames ?? 1),
    detector: Array.isArray(p.detector) ? p.detector.map((d) => String(d)) : [],
    verifier: String(p.verifier || ""),
  })).filter((p) => p.name);
  policyByName = Object.fromEntries(POLICY.map((p) => [p.name, p]));
}

const STREAMS = [
  { id: "cam-01", name: "Lobby", location: "Bldg A · Floor 1", status: "online", fps: 12 },
  { id: "cam-02", name: "Warehouse", location: "Bldg C · Bay 3", status: "online", fps: 12 },
  { id: "cam-03", name: "Parking", location: "Outdoor · West", status: "online", fps: 10 },
  { id: "cam-04", name: "Stairwell", location: "Bldg A · Core", status: "degraded", fps: 8 },
];

/* ───────────────────────── Sample alerts (VerifiedAlert.to_json() shape) ───────────────────────── */
const FW = 1280, FH = 720; // reference frame size for xyxy
function bboxToXyxy(b) {
  return [b[0] * FW, b[1] * FH, (b[0] + b[2]) * FW, (b[1] + b[3]) * FH].map((v) => Math.round(v));
}
// Build one record from a compact spec.
function mkAlert(spec) {
  const cls = policyByName[spec.class_name];
  const bbox = spec.bbox;
  const det = {
    score: spec.score,
    xyxy: bboxToXyxy(bbox),
    raw_label: spec.raw_label || cls.detector[0],
    track_id: spec.track_id ?? null,
  };
  return {
    stream_id: spec.stream_id,
    ts: spec.ts,
    class_name: spec.class_name,
    severity: cls.severity,
    start_pts_s: spec.peak - 1.2,
    peak_pts_s: spec.peak,
    peak_frame_index: Math.round(spec.peak * 12),
    track_id: spec.track_id ?? null,
    peak_detections: [det],
    num_keyframes: spec.keyframes ?? 5,
    true_alert: spec.true_alert,
    confidence: spec.confidence,
    false_negative_class: spec.fn ?? null,
    rationale: spec.rationale,
    bbox_xywh_norm: bbox,
    trajectory_xy_norm: spec.traj || [],
    verifier_raw: JSON.stringify(
      {
        true_alert: spec.true_alert,
        confidence: spec.confidence,
        false_negative_class: spec.fn ?? null,
        bbox_xywh_norm: bbox,
        trajectory_xy_norm: spec.traj || [],
        rationale: spec.rationale,
      },
      null,
      2,
    ),
    thumbnail_path: `thumbnails/${spec.stream_id}_${spec.class_name}_${spec.peak_frame_index || Math.round(spec.peak * 12)}.jpg`,
  };
}

const T0 = Date.now();
const ago = (min) => new Date(T0 - min * 60000).toISOString();
const SAMPLE_ALERTS = [
  { stream_id: "cam-02", ts: ago(1.5), class_name: "fire", peak: 832.4, score: 0.71, true_alert: true, confidence: 0.97, track_id: 41, bbox: [0.52, 0.46, 0.22, 0.3], rationale: "Sustained open flame on pallet stacking; intensity rising across keyframes." },
  { stream_id: "cam-02", ts: ago(3.2), class_name: "smoke", peak: 818.9, score: 0.63, true_alert: true, confidence: 0.66, track_id: 44, bbox: [0.33, 0.18, 0.4, 0.34], fn: "fire", rationale: "Dense smoke column; a flame base is visible that the detector did not box." },
  { stream_id: "cam-04", ts: ago(4.7), class_name: "falldown", peak: 511.2, score: 0.58, true_alert: true, confidence: 0.85, track_id: 7, bbox: [0.28, 0.55, 0.34, 0.22], traj: [[0.62, 0.4], [0.55, 0.5], [0.45, 0.62], [0.4, 0.66]], rationale: "Person descends the stairwell, then stays prone and motionless past the persistence window." },
  { stream_id: "cam-01", ts: ago(6.1), class_name: "weapon", peak: 274.0, score: 0.46, true_alert: false, confidence: 0.21, track_id: 18, bbox: [0.61, 0.42, 0.1, 0.26], rationale: "Elongated object resolves to a closed umbrella, not a weapon." },
  { stream_id: "cam-03", ts: ago(8.4), class_name: "falldown", peak: 9913.5, score: 0.41, true_alert: false, confidence: 0.18, track_id: 23, bbox: [0.44, 0.6, 0.2, 0.16], rationale: "Subject crouches to tie a shoelace and stands back up within two seconds." },
  { stream_id: "cam-02", ts: ago(11.0), class_name: "fire", peak: 612.7, score: 0.69, true_alert: true, confidence: 0.93, track_id: 41, bbox: [0.5, 0.48, 0.2, 0.28], rationale: "Active fire at the same bay; consistent with the earlier track." },
  { stream_id: "cam-03", ts: ago(13.9), class_name: "weapon", peak: 444.1, score: 0.52, true_alert: true, confidence: 0.69, track_id: 31, bbox: [0.4, 0.46, 0.14, 0.2], rationale: "Individual draws and points a handgun toward the lot entrance." },
  { stream_id: "cam-01", ts: ago(17.3), class_name: "smoke", peak: 188.2, score: 0.34, true_alert: true, confidence: 0.74, track_id: 12, bbox: [0.2, 0.22, 0.3, 0.26], rationale: "Light haze accumulating near the ceiling diffuser; consistent with early smolder." },
  { stream_id: "cam-04", ts: ago(21.8), class_name: "falldown", peak: 402.6, score: 0.55, true_alert: true, confidence: 0.81, track_id: 5, bbox: [0.34, 0.58, 0.3, 0.2], rationale: "Elderly person slips on the landing and does not get up." },
  { stream_id: "cam-01", ts: ago(26.5), class_name: "fire", peak: 95.4, score: 0.33, true_alert: false, confidence: 0.29, track_id: null, bbox: [0.7, 0.3, 0.16, 0.18], rationale: "Warm-colored region is direct sunlight glare on a glass partition, not flame." },
  { stream_id: "cam-02", ts: ago(31.2), class_name: "smoke", peak: 770.1, score: 0.6, true_alert: true, confidence: 0.78, track_id: 44, bbox: [0.36, 0.2, 0.36, 0.3], rationale: "Billowing grey smoke spreading laterally across the bay." },
  { stream_id: "cam-03", ts: ago(38.0), class_name: "falldown", peak: 251.9, score: 0.49, true_alert: true, confidence: 0.77, track_id: 27, bbox: [0.5, 0.62, 0.26, 0.18], rationale: "Cyclist falls near the ramp and remains down across the clip." },
].map(mkAlert);

/* ───────────────────────── Palette helpers ───────────────────────── */
const SEVERITY = {
  critical: { hex: "#ef4444", dot: "bg-red-500", chip: "border-red-500 text-red-400", chipLight: "bg-red-100 text-red-700 border-red-300" },
  high: { hex: "#f59e0b", dot: "bg-amber-500", chip: "border-amber-500 text-amber-400", chipLight: "bg-amber-100 text-amber-700 border-amber-300" },
  medium: { hex: "#eab308", dot: "bg-yellow-500", chip: "border-yellow-500 text-yellow-400", chipLight: "bg-yellow-100 text-yellow-700 border-yellow-300" },
  low: { hex: "#3b82f6", dot: "bg-blue-500", chip: "border-blue-500 text-blue-400", chipLight: "bg-blue-100 text-blue-700 border-blue-300" },
  info: { hex: "#6b7280", dot: "bg-gray-500", chip: "border-gray-500 text-gray-400", chipLight: "bg-gray-100 text-gray-700 border-gray-300" },
};
const sev = (s) => SEVERITY[s] || SEVERITY.info;

const fmtTime = (iso) => {
  if (!iso) return "recorded";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "recorded";
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
};
const fmtPts = (s) => {
  const m = Math.floor(s / 60), r = (s % 60).toFixed(1);
  return `${m}:${r.padStart(4, "0")}`;
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

async function apiJSON(path) {
  const res = await fetch(apiUrl(path), { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} returned ${res.status}`);
  return res.json();
}

function normalizeAlert(raw) {
  const fallback = policyByName[raw.class_name] || POLICY[0];
  return {
    ...raw,
    ts: raw.ts || raw.created_at || raw.written_at || "",
    stream_id: String(raw.stream_id || "cam-01"),
    class_name: String(raw.class_name || fallback.name),
    severity: String(raw.severity || fallback.severity),
    true_alert: Boolean(raw.true_alert),
    confidence: Number(raw.confidence ?? 0),
    peak_pts_s: Number(raw.peak_pts_s ?? raw.start_pts_s ?? 0),
    peak_frame_index: Number(raw.peak_frame_index ?? 0),
    peak_detections: Array.isArray(raw.peak_detections) && raw.peak_detections.length
      ? raw.peak_detections
      : [{ score: Number(raw.confidence ?? 0), xyxy: [], raw_label: raw.class_name, track_id: raw.track_id ?? null }],
    num_keyframes: Number(raw.num_keyframes ?? 0),
    rationale: String(raw.rationale || "No verifier rationale recorded."),
    bbox_xywh_norm: Array.isArray(raw.bbox_xywh_norm) ? raw.bbox_xywh_norm : [0.2, 0.2, 0.35, 0.35],
    trajectory_xy_norm: Array.isArray(raw.trajectory_xy_norm) ? raw.trajectory_xy_norm : [],
    verifier_raw: String(raw.verifier_raw || JSON.stringify(raw, null, 2)),
    thumbnail_path: String(raw.thumbnail_path || ""),
    thumbnail_url: String(raw.thumbnail_url || ""),
    false_negative_class: raw.false_negative_class == null ? null : String(raw.false_negative_class),
  };
}

function mergeByAlertId(current, incoming) {
  const seen = new Map(current.map((a) => [a._alert_id || `${a.stream_id}:${a._line}:${a.ts}`, a]));
  incoming.forEach((a) => seen.set(a._alert_id || `${a.stream_id}:${a._line}:${a.ts}`, a));
  return [...seen.values()].sort((a, b) => {
    const ta = Date.parse(a.ts || "") || Number(a.peak_pts_s || 0);
    const tb = Date.parse(b.ts || "") || Number(b.peak_pts_s || 0);
    return tb - ta;
  });
}

function thumbMarkup(a, w = 168, h = 94) {
  if (a.thumbnail_url) {
    return `<img src="${esc(apiUrl(a.thumbnail_url))}" width="${w}" height="${h}" alt="${esc(a.class_name)} thumbnail" class="rounded-md block object-cover bg-gray-900" style="width:${w}px;height:${h}px">`;
  }
  return thumbSVG(a, w, h);
}

function runSourceLabel(runName) {
  if (!runName) return "none";
  return runName === "live" ? "inference" : "fixture/demo";
}

function runSelect() {
  if (state.backend !== "ok") return "";
  const options = state.runs.length
    ? state.runs.map((run) => `<option value="${esc(run.name)}" ${state.selectedRun === run.name ? "selected" : ""}>${esc(run.name)} · ${runSourceLabel(run.name)}</option>`).join("")
    : `<option value="">No runs</option>`;
  return `<label class="flex items-center gap-1.5 text-sm">
    <span class="text-gray-500 dark:text-gray-400">Run</span>
    <select data-run-select class="bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-nv-green">${options}</select>
  </label>`;
}

/* Synthetic CCTV thumbnail: dark frame + detector bbox + optional trajectory. */
function thumbSVG(a, w = 168, h = 94) {
  const s = sev(a.severity), b = a.bbox_xywh_norm;
  const x = b[0] * w, y = b[1] * h, bw = b[2] * w, bh = b[3] * h, tick = Math.min(bw, bh) * 0.28;
  const dimmed = !a.true_alert;
  const traj = (a.trajectory_xy_norm || []).map((p) => `${(p[0] * w).toFixed(1)},${(p[1] * h).toFixed(1)}`).join(" ");
  const gid = `g${Math.random().toString(36).slice(2, 8)}`;
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg" class="rounded-md block">
    <defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#1f2733"/><stop offset="1" stop-color="#0b0f16"/></linearGradient></defs>
    <rect width="${w}" height="${h}" fill="url(#${gid})"/>
    ${Array.from({ length: 6 }, (_, i) => `<line x1="0" y1="${(i + 1) * h / 7}" x2="${w}" y2="${(i + 1) * h / 7}" stroke="#ffffff" stroke-opacity="0.04"/>`).join("")}
    ${traj ? `<polyline points="${traj}" fill="none" stroke="${s.hex}" stroke-opacity="0.7" stroke-width="1.5" stroke-dasharray="3 3"/>` : ""}
    <g opacity="${dimmed ? 0.55 : 1}">
      <rect x="${x}" y="${y}" width="${bw}" height="${bh}" fill="${s.hex}" fill-opacity="0.12" stroke="${s.hex}" stroke-width="1.6"/>
      <path d="M${x} ${y + tick} L${x} ${y} L${x + tick} ${y}" stroke="#fff" stroke-width="1.4" fill="none"/>
      <path d="M${x + bw - tick} ${y + bh} L${x + bw} ${y + bh} L${x + bw} ${y + bh - tick}" stroke="#fff" stroke-width="1.4" fill="none"/>
    </g>
    <rect x="${x}" y="${Math.max(0, y - 11)}" width="${Math.max(34, String(a.class_name).length * 6.2 + 16)}" height="11" fill="${s.hex}"/>
    <text x="${x + 3}" y="${Math.max(8, y - 2.5)}" font-family="monospace" font-size="7.5" fill="#0b0f16" font-weight="bold">${esc(a.class_name)} ${a.peak_detections[0].score.toFixed(2)}</text>
    <text x="4" y="10" font-family="monospace" font-size="7" fill="#9fb84f">${esc(a.stream_id)}</text>
    <circle cx="${w - 18}" cy="8" r="2.2" fill="#ef4444"/>
    <text x="${w - 13}" y="10.5" font-family="monospace" font-size="7" fill="#cbd5e1">REC</text>
    <text x="4" y="${h - 4}" font-family="monospace" font-size="7" fill="#9aa6b2">${fmtPts(a.peak_pts_s)}</text>
  </svg>`;
}

const verdictChip = (a) =>
  a.true_alert
    ? `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold border border-emerald-500 text-emerald-400 dark:bg-transparent bg-emerald-100 dark:text-emerald-400">${icon("shield", 12)} CONFIRMED</span>`
    : `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold border border-gray-400 text-gray-400">${icon("x", 12)} SUPPRESSED</span>`;

const sevChip = (s) => {
  const c = sev(s);
  return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold border ${c.chip} dark:bg-transparent ${c.chipLight}"><span class="w-1.5 h-1.5 rounded-full ${c.dot}"></span>${esc(s)}</span>`;
};

const confBar = (a) => {
  const pct = Math.round(a.confidence * 100);
  const color = a.true_alert ? "#76b900" : "#9ca3af";
  return `<div class="flex items-center gap-2 min-w-[92px]">
    <div class="flex-1 h-1.5 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
      <div class="h-full rounded-full" style="width:${pct}%;background:${color}"></div>
    </div>
    <span class="text-xs tabular-nums text-gray-600 dark:text-gray-300 w-8 text-right">${pct}%</span>
  </div>`;
};

/* ───────────────────────── State ───────────────────────── */
const TABS = [
  { id: "alerts", label: "Live Alerts", icon: "alert" },
  { id: "cascade", label: "Cascade", icon: "cascade" },
  { id: "streams", label: "Streams", icon: "video" },
  { id: "policy", label: "Watch Policy", icon: "policy" },
];
const state = {
  tab: "alerts",
  theme: localStorage.getItem("vrs-theme") || "dark",
  alerts: SAMPLE_ALERTS.slice(),
  runs: [],
  selectedRun: DEFAULT_RUN,
  streams: STREAMS.slice(),
  backend: "sample",
  rtsp: null,
  jsonlErrors: [],
  tailCursor: "",
  autoSelectedFallback: true,
  filters: { severity: "all", class: "all", verdict: "all", stream: "all" },
  autorefresh: true,
};

/* ───────────────────────── Renderers ───────────────────────── */
const $ = (id) => document.getElementById(id);

function renderNav() {
  $("tab-nav").innerHTML = TABS.map((t) => {
    const active = state.tab === t.id;
    const cls = active
      ? "active bg-gray-300 dark:bg-gray-600 text-gray-900 dark:text-white border-gray-400 dark:border-gray-800 shadow-lg"
      : "text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 hover:shadow-md";
    const count = t.id === "alerts" ? `<span class="ml-auto text-xs font-semibold px-1.5 py-0.5 rounded bg-nv-green/15 text-nv-green">${state.alerts.length}</span>` : "";
    return `<button data-tab="${t.id}" class="vss-tab ${cls} w-full flex items-center px-3 py-2 text-[14px] font-medium rounded-md">
      <span class="mr-3 flex-shrink-0">${icon(t.icon, 18)}</span>
      <span class="text-left leading-tight">${t.label}</span>${count}
    </button>`;
  }).join("");
  $("tab-nav").querySelectorAll("[data-tab]").forEach((b) =>
    b.addEventListener("click", () => { state.tab = b.dataset.tab; render(); }));
}

function renderRuntimePanel() {
  const rows = [
    ["Runs root", state.runsRoot || "runs/", "cascade"],
    ["Backend", state.backend === "ok" ? "FastAPI connected" : "sample fallback", "shield"],
    ["RTSP sample", state.rtsp?.url || "not configured", "video"],
    ["Selected run", `${state.selectedRun || "none"} · ${runSourceLabel(state.selectedRun)}`, "activity"],
  ];
  $("runtime-panel").innerHTML =
    rows.map(([k, v, ic]) =>
      `<div class="flex items-start gap-2">
        <span class="text-nv-green mt-0.5">${icon(ic, 14)}</span>
        <div class="min-w-0">
          <div class="text-xs text-gray-400 dark:text-gray-500">${k}</div>
          <div class="text-[13px] text-gray-700 dark:text-gray-200 truncate">${esc(v)}</div>
        </div>
      </div>`).join("") +
    `<div class="text-[11px] text-gray-400 dark:text-gray-500 pt-1 leading-snug">Docker Compose starts RTSP sample streaming, FastAPI, and this dashboard together.</div>`;
}

/* ── Live Alerts ── */
function filteredAlerts() {
  const f = state.filters;
  return state.alerts.filter((a) =>
    (f.severity === "all" || a.severity === f.severity) &&
    (f.class === "all" || a.class_name === f.class) &&
    (f.stream === "all" || a.stream_id === f.stream) &&
    (f.verdict === "all" || (f.verdict === "confirmed") === a.true_alert));
}

function statCard(label, value, sub, accent) {
  return `<div class="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 py-3">
    <div class="text-xs uppercase tracking-wider text-gray-400 dark:text-gray-500">${label}</div>
    <div class="text-2xl font-bold mt-1" style="${accent ? `color:${accent}` : ""}">${value}</div>
    <div class="text-xs text-gray-400 dark:text-gray-500 mt-0.5">${sub}</div>
  </div>`;
}

function selectFilter(key, label, options) {
  const opts = options.map((o) =>
    `<option value="${esc(o.v)}" ${state.filters[key] === o.v ? "selected" : ""}>${esc(o.t)}</option>`).join("");
  return `<label class="flex items-center gap-1.5 text-sm">
    <span class="text-gray-500 dark:text-gray-400">${esc(label)}</span>
    <select data-filter="${key}" class="bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-nv-green">${opts}</select>
  </label>`;
}

function renderAlerts() {
  const all = state.alerts, list = filteredAlerts();
  const confirmed = all.filter((a) => a.true_alert).length;
  const suppressed = all.length - confirmed;
  const flip = all.length ? Math.round((suppressed / all.length) * 100) : 0;
  const avg = confirmed ? all.filter((a) => a.true_alert).reduce((s, a) => s + a.confidence, 0) / confirmed : 0;
  const online = state.streams.filter((s) => s.status === "online").length;

  const rows = list.map((a, i) => `
    <tr class="border-b border-gray-100 dark:border-gray-700/60 hover:bg-gray-50 dark:hover:bg-gray-700/40 cursor-pointer" data-alert="${all.indexOf(a)}">
      <td class="py-2 pl-4 pr-2">${thumbMarkup(a, 132, 74)}</td>
      <td class="px-2 align-middle">
        <div class="flex items-center gap-2">${icon("camera", 14, "text-gray-400")}<span class="font-medium">${esc(a.stream_id)}</span></div>
        <div class="text-xs text-gray-400 dark:text-gray-500">${fmtTime(a.ts)} · t=${fmtPts(a.peak_pts_s)}</div>
      </td>
      <td class="px-2 align-middle">${sevChip(a.severity)}<div class="text-sm mt-1 font-medium">${esc(a.class_name)}</div></td>
      <td class="px-2 align-middle">${verdictChip(a)}</td>
      <td class="px-2 align-middle">${confBar(a)}</td>
      <td class="px-2 align-middle text-sm text-gray-500 dark:text-gray-400">${a.track_id == null ? "—" : "#" + a.track_id}</td>
      <td class="px-2 pr-4 align-middle max-w-[280px]">
        <div class="text-sm text-gray-700 dark:text-gray-200 line-clamp-2">${esc(a.rationale)}</div>
        ${a.false_negative_class ? `<div class="text-xs mt-1 inline-flex items-center gap-1 text-amber-500">${icon("alert", 12)} verifier flagged missed: ${esc(a.false_negative_class)}</div>` : ""}
      </td>
    </tr>`).join("");

  $("view").innerHTML = `
    <div class="p-6 space-y-5">
      <div class="grid grid-cols-2 base:grid-cols-3 lg:grid-cols-6 gap-3">
        ${statCard("Candidates", all.length, "promoted by event-state", null)}
        ${statCard("Confirmed", confirmed, "VLM true_alert", "#76b900")}
        ${statCard("Suppressed", suppressed, "false alarms cut", "#9ca3af")}
        ${statCard("Flip rate", flip + "%", "detector→verifier", "#f59e0b")}
        ${statCard("Avg conf", avg.toFixed(2), "on confirmed", null)}
        ${statCard("Streams", online + "/" + state.streams.length, "online", null)}
      </div>

      <div class="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden">
        <div class="flex flex-wrap items-center gap-4 px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          ${runSelect()}
          ${selectFilter("severity", "Severity", [{ v: "all", t: "All" }, { v: "critical", t: "Critical" }, { v: "high", t: "High" }, { v: "medium", t: "Medium" }, { v: "low", t: "Low" }])}
          ${selectFilter("class", "Class", [{ v: "all", t: "All" }, ...POLICY.map((p) => ({ v: p.name, t: p.name }))])}
          ${selectFilter("verdict", "Verdict", [{ v: "all", t: "All" }, { v: "confirmed", t: "Confirmed" }, { v: "suppressed", t: "Suppressed" }])}
          ${selectFilter("stream", "Stream", [{ v: "all", t: "All" }, ...state.streams.map((s) => ({ v: s.id, t: s.id }))])}
          <span class="ml-auto text-sm text-gray-400 dark:text-gray-500">${list.length} of ${all.length}</span>
        </div>
        <div class="overflow-x-auto nv-scroll">
          <table class="w-full text-left border-collapse">
            <thead class="text-xs uppercase tracking-wider text-gray-400 dark:text-gray-500 bg-gray-50 dark:bg-gray-750/40">
              <tr>
                <th class="py-2 pl-4 pr-2 font-medium">Keyframe</th>
                <th class="px-2 font-medium">Source</th>
                <th class="px-2 font-medium">Event</th>
                <th class="px-2 font-medium">VLM verdict</th>
                <th class="px-2 font-medium">Confidence</th>
                <th class="px-2 font-medium">Track</th>
                <th class="px-2 pr-4 font-medium">Rationale</th>
              </tr>
            </thead>
            <tbody>${rows || `<tr><td colspan="7" class="py-10 text-center text-gray-400">No alerts match the current filters.</td></tr>`}</tbody>
          </table>
        </div>
      </div>
    </div>`;

  $("view").querySelectorAll("[data-filter]").forEach((s) =>
    s.addEventListener("change", () => { state.filters[s.dataset.filter] = s.value; renderAlerts(); }));
  $("view").querySelectorAll("[data-run-select]").forEach((s) =>
    s.addEventListener("change", () => {
      state.selectedRun = s.value;
      state.autoSelectedFallback = false;
      state.alerts = [];
      state.tailCursor = "";
      refreshFromApi(true).catch(() => renderAlerts());
    }));
  $("view").querySelectorAll("[data-alert]").forEach((r) =>
    r.addEventListener("click", () => openDrawer(all[+r.dataset.alert])));
}

/* ── Cascade ── */
function renderCascade() {
  const stage = (ic, title, sub, lines, accent) => `
    <div class="flex-1 min-w-[180px] rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
      <div class="flex items-center gap-2 mb-2"><span style="color:${accent}">${icon(ic, 20)}</span>
        <h3 class="font-semibold">${title}</h3></div>
      <div class="text-xs text-gray-400 dark:text-gray-500 mb-3">${sub}</div>
      <ul class="space-y-1 text-sm text-gray-600 dark:text-gray-300">
        ${lines.map((l) => `<li class="flex gap-2"><span class="text-nv-green">›</span><span>${l}</span></li>`).join("")}
      </ul>
    </div>`;
  const arrow = `<div class="flex items-center justify-center px-1 text-gray-400 shrink-0">${icon("chevron", 22)}</div>`;

  $("view").innerHTML = `
    <div class="p-6 space-y-6">
      <div>
        <h2 class="text-lg font-bold">Two-stage cascade</h2>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-2xl">
          A cheap open-vocabulary detector runs on every frame; only persisted candidates pay for the VLM.
          Text prompts come from the watch policy, so adding an event class is a YAML edit — no retraining.
        </p>
      </div>
      <div class="flex flex-col lg:flex-row gap-2 items-stretch">
        ${stage("video", "Decode", "ingest", ["OpenCV / NVDEC reader", "BGR uint8 @ target FPS", "BoundedQueue · drop-oldest"], "#38bdf8")}
        ${arrow}
        ${stage("bolt", "YOLOE — fast path", "every frame", ["Open-vocab detector (FP16)", "Prompts → boxes by event name", "Zero per-frame text cost"], "#76b900")}
        ${arrow}
        ${stage("activity", "Event-state", "persistence + cooldown", ["min_persist_frames gate", "Cooldown per (class, track)", "Keyframes around the peak"], "#a855f7")}
        ${arrow}
        ${stage("cpu", "VLM verifier — slow path", "candidates only", ["Strict-JSON verdict", "true_alert · confidence · bbox", "Cosmos baseline · pluggable"], "#f59e0b")}
        ${arrow}
        ${stage("shield", "Sink", "outputs", ["alerts.jsonl record", "Event thumbnails", "Optional annotated.mp4"], "#10b981")}
      </div>

      <div class="grid base:grid-cols-2 gap-4">
        <div class="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
          <h3 class="font-semibold mb-2 flex items-center gap-2">${icon("cpu", 16, "text-nv-green")} Verifier contract</h3>
          <p class="text-sm text-gray-500 dark:text-gray-400 mb-3">Every candidate returns one strict-JSON object, parsed by a balanced-brace scanner (XGrammar-constrained when available).</p>
          <pre class="text-xs bg-gray-50 dark:bg-gray-900 rounded-md p-3 overflow-x-auto nv-scroll text-gray-700 dark:text-gray-300">${esc(`{
  "true_alert": true,
  "confidence": 0.0-1.0,
  "false_negative_class": null | "<policy class>",
  "bbox_xywh_norm": [x, y, w, h],
  "trajectory_xy_norm": [[x, y], ...],
  "rationale": "one short sentence"
}`)}</pre>
        </div>
        <div class="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
          <h3 class="font-semibold mb-2 flex items-center gap-2">${icon("cascade", 16, "text-nv-green")} Multi-stream</h3>
          <p class="text-sm text-gray-500 dark:text-gray-400 mb-3">One shared YOLOE + one shared verifier fan out across N cameras; only the detector thread touches the YOLOE CUDA model, only the verifier thread touches the VLM.</p>
          <div class="space-y-2 font-mono text-xs text-gray-600 dark:text-gray-300">
            <div>RTSP[i] → DecoderThread[i] → frame_q</div>
            <div class="pl-8">→ DetectorWorker (batched YOLOE)</div>
            <div class="pl-16">→ VerifierWorker (shared VLM)</div>
            <div class="pl-24">→ SinkWorker[i] (jsonl + thumbs)</div>
          </div>
        </div>
      </div>
    </div>`;
}

/* ── Streams ── */
function renderStreams() {
  const cards = state.streams.map((s) => {
    const recent = state.alerts.filter((a) => a.stream_id === s.id);
    const last = recent[0];
    const online = s.status === "online";
    const dotCls = online ? "bg-nv-green" : "bg-amber-500";
    const streamUrl = s.rtsp_url || state.rtsp?.url || "";
    return `<div class="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden">
      <div class="relative">
        ${last ? thumbMarkup(last, 640, 240).replace('style="width:640px;height:240px"', 'style="width:100%;height:240px"') : `<div class="w-full aspect-video bg-gray-900 flex items-center justify-center text-gray-600">${icon("video", 32)}</div>`}
        <div class="absolute top-2 left-2 flex items-center gap-1.5 px-2 py-1 rounded bg-black/55 text-xs text-white">
          <span class="live-dot w-2 h-2 rounded-full ${dotCls}"></span>${esc(String(s.status || "").toUpperCase())} · ${Number(s.fps || 0)} fps
        </div>
      </div>
      <div class="p-3">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2"><span class="text-gray-400">${icon("camera", 16)}</span>
            <span class="font-semibold">${esc(s.name)}</span>
            <span class="text-xs text-gray-400 dark:text-gray-500">${esc(s.id)}</span></div>
          <span class="text-xs px-1.5 py-0.5 rounded bg-nv-green/15 text-nv-green font-semibold">${recent.length} alerts</span>
        </div>
        <div class="flex items-center gap-1 text-xs text-gray-400 dark:text-gray-500 mt-1">${icon("pin", 12)} ${esc(s.location)}</div>
        ${streamUrl ? `<div class="mt-1 text-xs text-gray-400 dark:text-gray-500 font-mono break-all">RTSP ${esc(streamUrl)}</div>` : ""}
        ${last ? `<div class="mt-2 flex items-center gap-2 text-sm">${sevChip(last.severity)}<span class="text-gray-500 dark:text-gray-400 truncate">${esc(last.rationale)}</span></div>` : ""}
      </div>
    </div>`;
  }).join("");
  $("view").innerHTML = `<div class="p-6">
    <h2 class="text-lg font-bold mb-1">Streams</h2>
    <p class="text-sm text-gray-500 dark:text-gray-400 mb-4">Per-camera latest keyframe. Docker Compose also publishes a synthetic RTSP sample stream for local plumbing checks.</p>
    <div class="grid base:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 gap-4">${cards}</div>
  </div>`;
}

/* ── Policy ── */
function renderPolicy() {
  const cards = POLICY.map((p) => {
    const c = sev(p.severity);
    const count = state.alerts.filter((a) => a.class_name === p.name).length;
    return `<div class="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
      <div class="flex items-center justify-between mb-2">
        <div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full ${c.dot}"></span>
          <h3 class="font-bold text-md">${esc(p.name)}</h3>${sevChip(p.severity)}</div>
        <span class="text-xs text-gray-400 dark:text-gray-500">${count} alerts</span>
      </div>
      <div class="text-sm text-gray-600 dark:text-gray-300 mb-3 italic">"${esc(p.verifier)}"</div>
      <div class="flex flex-wrap gap-1.5 mb-3">
        ${p.detector.map((d) => `<span class="text-xs px-2 py-0.5 rounded-full border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300">${esc(d)}</span>`).join("")}
      </div>
      <div class="grid grid-cols-2 gap-2 text-xs">
        <div class="rounded bg-gray-50 dark:bg-gray-900 px-2 py-1"><span class="text-gray-400">min_score</span> <span class="font-mono font-semibold">${p.min_score.toFixed(2)}</span></div>
        <div class="rounded bg-gray-50 dark:bg-gray-900 px-2 py-1"><span class="text-gray-400">min_persist</span> <span class="font-mono font-semibold">${p.min_persist_frames}f</span></div>
      </div>
    </div>`;
  }).join("");
  $("view").innerHTML = `<div class="p-6">
    <div class="flex items-start justify-between flex-wrap gap-2 mb-4">
      <div>
        <h2 class="text-lg font-bold">Watch policy</h2>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-2xl">Operator-facing event list — the only file you edit to add a class. Detector prompts feed YOLOE; the verifier sentence defines the event for the VLM.</p>
      </div>
      <code class="text-xs px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">configs/policies/safety.yaml</code>
    </div>
    <div class="grid base:grid-cols-2 xl:grid-cols-2 gap-4 max-w-5xl">${cards}</div>
  </div>`;
}

/* ── Detail drawer ── */
function openDrawer(a) {
  const d = a.peak_detections[0];
  $("drawer").innerHTML = `
    <div class="p-5">
      <div class="flex items-start justify-between mb-4">
        <div>
          <div class="flex items-center gap-2">${sevChip(a.severity)}${verdictChip(a)}</div>
          <h2 class="text-xl font-bold mt-2">${esc(a.class_name)} · ${esc(a.stream_id)}</h2>
          <div class="text-sm text-gray-400 dark:text-gray-500">${new Date(a.ts).toLocaleString()} · frame ${a.peak_frame_index} · t=${fmtPts(a.peak_pts_s)}</div>
        </div>
        <button id="drawer-close" class="p-1.5 text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">${icon("x", 20)}</button>
      </div>
      <div class="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 mb-4">${thumbMarkup(a, 440, 248).replace('style="width:440px;height:248px"', 'style="width:100%;height:248px"')}</div>
      <div class="grid grid-cols-2 gap-3 mb-4 text-sm">
        <div class="rounded bg-gray-50 dark:bg-gray-900 px-3 py-2"><div class="text-xs text-gray-400">Confidence</div><div class="font-semibold">${(a.confidence * 100).toFixed(0)}%</div></div>
        <div class="rounded bg-gray-50 dark:bg-gray-900 px-3 py-2"><div class="text-xs text-gray-400">Detector score</div><div class="font-semibold">${Number(d.score ?? 0).toFixed(2)}</div></div>
        <div class="rounded bg-gray-50 dark:bg-gray-900 px-3 py-2"><div class="text-xs text-gray-400">Track id</div><div class="font-semibold">${a.track_id == null ? "untracked" : "#" + a.track_id}</div></div>
        <div class="rounded bg-gray-50 dark:bg-gray-900 px-3 py-2"><div class="text-xs text-gray-400">Keyframes</div><div class="font-semibold">${a.num_keyframes}</div></div>
      </div>
      ${a.false_negative_class ? `<div class="mb-4 flex items-center gap-2 text-sm text-amber-500 border border-amber-500/40 rounded px-3 py-2">${icon("alert", 16)} Verifier flagged a missed event the detector did not box: <b>${esc(a.false_negative_class)}</b></div>` : ""}
      <div class="mb-4">
        <div class="text-xs uppercase tracking-wider text-gray-400 mb-1">Rationale</div>
        <p class="text-sm text-gray-700 dark:text-gray-200">${esc(a.rationale)}</p>
      </div>
      <div class="mb-2">
        <div class="text-xs uppercase tracking-wider text-gray-400 mb-1">Raw verifier output</div>
        <pre class="text-xs bg-gray-50 dark:bg-gray-900 rounded-md p-3 overflow-x-auto nv-scroll text-gray-700 dark:text-gray-300">${esc(a.verifier_raw)}</pre>
      </div>
      <div class="text-xs text-gray-400 dark:text-gray-500 font-mono break-all">${esc(a.thumbnail_path)}</div>
    </div>`;
  $("drawer").style.transform = "translateX(0)";
  $("drawer-backdrop").classList.remove("hidden");
  $("drawer-close").addEventListener("click", closeDrawer);
}
function closeDrawer() {
  $("drawer").style.transform = "translateX(100%)";
  $("drawer-backdrop").classList.add("hidden");
}

/* ───────────────────────── Theme + chrome ───────────────────────── */
function applyTheme() {
  document.documentElement.classList.toggle("dark", state.theme === "dark");
  $("theme-toggle").innerHTML = icon(state.theme === "dark" ? "sun" : "moon", 22);
}
function toggleTheme() {
  state.theme = state.theme === "dark" ? "light" : "dark";
  localStorage.setItem("vrs-theme", state.theme);
  applyTheme();
}

function render() {
  renderNav();
  ({ alerts: renderAlerts, cascade: renderCascade, streams: renderStreams, policy: renderPolicy }[state.tab] || renderAlerts)();
}

/* Simulate a live feed: occasionally synthesize a new candidate. */
function maybePushAlert() {
  if (!state.autorefresh) return;
  if (state.backend === "ok") {
    syncBackendMetadata().then(() => refreshFromApi(false)).catch(() => {});
    return;
  }
  if (Math.random() > 0.55) return;
  const p = POLICY[Math.floor(Math.random() * POLICY.length)];
  const s = state.streams[Math.floor(Math.random() * state.streams.length)];
  const confirmed = Math.random() > 0.35;
  const a = mkAlert({
    stream_id: s.id, ts: new Date().toISOString(), class_name: p.name,
    peak: 100 + Math.random() * 800, score: +(p.min_score + Math.random() * 0.5).toFixed(2),
    true_alert: confirmed, confidence: +(confirmed ? 0.6 + Math.random() * 0.39 : Math.random() * 0.35).toFixed(2),
    track_id: Math.random() > 0.2 ? Math.floor(Math.random() * 60) : null,
    bbox: [0.2 + Math.random() * 0.4, 0.2 + Math.random() * 0.4, 0.15 + Math.random() * 0.2, 0.15 + Math.random() * 0.2],
    rationale: confirmed ? `${p.name} confirmed on ${s.name.toLowerCase()} feed; persisted past the gate.` : `Candidate ${p.name} on ${s.name.toLowerCase()} dismissed by the verifier.`,
  });
  state.alerts.unshift(a);
  if (state.alerts.length > 40) state.alerts.pop();
  renderNav();
  if (state.tab === "alerts") renderAlerts();
  else if (state.tab === "streams") renderStreams();
}

async function refreshFromApi(reset) {
  if (!state.selectedRun) return;
  const params = new URLSearchParams({ limit: "500" });
  if (reset) params.set("mode", "latest");
  else if (state.tailCursor) params.set("cursor", state.tailCursor);
  const body = await apiJSON(`/api/runs/${encodeURIComponent(state.selectedRun)}/tail?${params}`);
  const incoming = body.alerts.map(normalizeAlert);
  state.jsonlErrors = body.errors || [];
  state.alerts = reset ? incoming : mergeByAlertId(state.alerts, incoming);
  state.tailCursor = body.next_cursor || state.tailCursor || "";
  state.backend = "ok";
  renderRuntimePanel();
  render();
}

async function syncBackendMetadata() {
  const health = await apiJSON("/api/health");
  const runsBody = await apiJSON("/api/runs");
  const streamsBody = await apiJSON("/api/streams");
  const previousRun = state.selectedRun;
  state.backend = "ok";
  state.runsRoot = health.runs_root;
  state.runs = runsBody.runs || [];
  state.rtsp = streamsBody.rtsp_sample || null;
  state.streams = streamsBody.streams?.length ? streamsBody.streams : STREAMS.slice();
  const hasSelected = state.runs.some((r) => r.name === state.selectedRun);
  const hasDefault = state.runs.some((r) => r.name === DEFAULT_RUN);
  if (hasDefault && state.autoSelectedFallback && state.selectedRun !== DEFAULT_RUN) {
    state.selectedRun = DEFAULT_RUN;
    state.alerts = [];
    state.tailCursor = "";
  } else if (!hasSelected) {
    state.selectedRun = hasDefault ? DEFAULT_RUN : state.runs[0]?.name || "";
    state.autoSelectedFallback = true;
    if (state.selectedRun !== previousRun) {
      state.alerts = [];
      state.tailCursor = "";
    }
  }
}

async function loadBackend() {
  try {
    await syncBackendMetadata();
    try {
      const policyBody = await apiJSON("/api/policy");
      setPolicy(policyBody.watch);
    } catch (err) {
      // Keep embedded defaults when the backend has no configured policy.
    }
    await refreshFromApi(true);
  } catch (err) {
    state.backend = "sample";
    state.streams = STREAMS.slice();
    renderRuntimePanel();
    render();
  }
}

/* ───────────────────────── Boot ───────────────────────── */
function tickClock() {
  $("header-clock").textContent = new Date().toLocaleTimeString([], { hour12: false });
}
function init() {
  // Deep-link support: ?tab=cascade&theme=light
  const params = new URLSearchParams(location.search);
  if (TABS.some((t) => t.id === params.get("tab"))) state.tab = params.get("tab");
  if (["dark", "light"].includes(params.get("theme"))) state.theme = params.get("theme");
  applyTheme();
  renderRuntimePanel();
  render();
  $("theme-toggle").addEventListener("click", toggleTheme);
  $("drawer-backdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  $("autorefresh-toggle").addEventListener("click", () => {
    state.autorefresh = !state.autorefresh;
    $("autorefresh-state").textContent = state.autorefresh ? "ON · 3s" : "OFF";
    $("autorefresh-state").className = state.autorefresh ? "font-semibold text-nv-green" : "font-semibold text-gray-400";
  });
  tickClock();
  setInterval(tickClock, 1000);
  setInterval(maybePushAlert, POLL_MS);
  loadBackend();
}
document.addEventListener("DOMContentLoaded", init);
