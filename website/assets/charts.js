/* Small SVG chart + player components shared by the results page and the demo. */
(function () {
  const NS = "http://www.w3.org/2000/svg";

  const CLASS_INFO = {
    accident:            { color: "#ff5a6a", name: "Accident" },
    near_miss:           { color: "#ff9f43", name: "Near miss" },
    red_light:           { color: "#ee5253", name: "Red-light running" },
    wrong_way:           { color: "#f368e0", name: "Wrong-way driving" },
    illegal_u_turn:      { color: "#a55eea", name: "Illegal U-turn" },
    stopped_vehicle:     { color: "#feca57", name: "Stopped vehicle" },
    jaywalking:          { color: "#1dd1a1", name: "Pedestrian on roadway" },
    failure_to_yield:    { color: "#10ac84", name: "Not yielding to pedestrian" },
    illegal_turn:        { color: "#5f27cd", name: "Illegal turn" },
    solid_line_crossing: { color: "#48dbfb", name: "Solid line crossing" },
    stop_line:           { color: "#ff6b6b", name: "Stop-line violation" },
    congestion:          { color: "#8395a7", name: "Congestion" },
    road_obstacle:       { color: "#c8d6e5", name: "Obstacle on road" },
    fire_smoke:          { color: "#ff7f00", name: "Fire or smoke" },
  };
  const CAT_COLORS = { vehicle: "#36c2ff", bike: "#ffc850", person: "#78ff78", obstacle: "#ff5050" };

  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function html(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  const fmt = (s) => { s = Math.max(0, s); const m = Math.floor(s / 60); return `${m}:${(s - 60 * m).toFixed(1).padStart(4, "0")}`; };

  /** Event timeline: one row per class, click a bar to seek. */
  function timelineChart(host, events, duration, onSeek) {
    const classes = [...new Set(events.map((e) => e[2]))];
    const rowH = 22, left = 150, W = 900, H = Math.max(1, classes.length) * rowH + 26;
    const box = html("div", "chart");
    box.appendChild(html("h4", null, "Event timeline"));
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
    const x = (t) => left + (t / duration) * (W - left - 10);
    const axis = el("g", { class: "axis" }, svg);
    const ticks = niceTicks(duration, 8);
    for (const t of ticks) {
      el("line", { x1: x(t), x2: x(t), y1: 0, y2: H - 18 }, axis);
      el("text", { x: x(t), y: H - 4, "text-anchor": "middle" }, axis).textContent = fmt(t);
    }
    if (!classes.length) el("text", { x: W / 2, y: 16, "text-anchor": "middle", fill: "#93a1b8" }, svg).textContent = "no events detected";
    classes.forEach((c, i) => {
      const info = CLASS_INFO[c] || { color: "#ccc", name: c };
      el("text", { x: 4, y: i * rowH + 15, fill: info.color, "font-size": 12 }, svg).textContent = info.name;
      for (const ev of events.filter((e) => e[2] === c)) {
        const r = el("rect", { class: "ev", x: x(ev[0]), y: i * rowH + 3, width: Math.max(3, x(ev[1]) - x(ev[0])), height: rowH - 6, rx: 3, fill: info.color }, svg);
        el("title", {}, r).textContent = `${info.name}: ${fmt(ev[0])} – ${fmt(ev[1])}`;
        r.addEventListener("click", () => onSeek && onSeek(ev[0]));
      }
    });
    const cursor = el("line", { class: "cursor", x1: left, x2: left, y1: 0, y2: H - 18 }, svg);
    svg.addEventListener("click", (e) => {
      if (e.target.classList.contains("ev")) return;
      const p = svgPoint(svg, e);
      if (p.x >= left && onSeek) onSeek(((p.x - left) / (W - left - 10)) * duration);
    });
    host.appendChild(box);
    return { setTime: (t) => { cursor.setAttribute("x1", x(t)); cursor.setAttribute("x2", x(t)); } };
  }

  /** Line chart over time (risk curve). */
  function lineChart(host, points, { title = "", duration, yMax = 1, threshold, color = "#36c2ff", onSeek, events = [] } = {}) {
    const W = 900, H = 150, left = 34, bottom = 20;
    const box = html("div", "chart");
    box.appendChild(html("h4", null, title));
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
    const dur = duration || (points.length ? points[points.length - 1][0] : 1);
    const x = (t) => left + (t / dur) * (W - left - 10);
    const y = (v) => 6 + (1 - v / yMax) * (H - bottom - 6);
    for (const ev of events.filter((e) => e[2] === "accident")) {
      el("rect", { x: x(ev[0] - 5), y: 6, width: x(ev[0]) - x(ev[0] - 5), height: H - bottom - 6, fill: "rgba(255,90,106,.15)" }, svg);
    }
    const axis = el("g", { class: "axis" }, svg);
    for (const v of [0, yMax / 2, yMax]) {
      el("line", { x1: left, x2: W - 10, y1: y(v), y2: y(v) }, axis);
      el("text", { x: left - 6, y: y(v) + 4, "text-anchor": "end" }, axis).textContent = v.toFixed(1);
    }
    for (const t of niceTicks(dur, 8)) el("text", { x: x(t), y: H - 4, "text-anchor": "middle" }, axis).textContent = fmt(t);
    if (threshold !== undefined) el("line", { class: "thr", x1: left, x2: W - 10, y1: y(threshold), y2: y(threshold) }, svg);
    if (points.length) {
      const d = points.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
      el("path", { d: d + `L${x(points[points.length - 1][0])},${y(0)}L${x(points[0][0])},${y(0)}Z`, fill: color, opacity: 0.15 }, svg);
      el("path", { d, stroke: color, "stroke-width": 1.8, fill: "none" }, svg);
    } else {
      el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: "#93a1b8" }, svg).textContent = "no data";
    }
    const cursor = el("line", { class: "cursor", x1: left, x2: left, y1: 6, y2: H - bottom }, svg);
    svg.addEventListener("click", (e) => { const p = svgPoint(svg, e); if (p.x >= left && onSeek) onSeek(((p.x - left) / (W - left - 10)) * dur); });
    host.appendChild(box);
    return { setTime: (t) => { cursor.setAttribute("x1", x(t)); cursor.setAttribute("x2", x(t)); } };
  }

  /** Multi-series line chart (object counts per category over time). */
  function seriesChart(host, series, { title = "", xLabel = (i) => i, colors = CAT_COLORS } = {}) {
    const W = 900, H = 200, left = 34, bottom = 20;
    const box = html("div", "chart");
    box.appendChild(html("h4", null, title));
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
    const keys = Object.keys(series).filter((k) => series[k].some((v) => v > 0));
    const n = Math.max(...keys.map((k) => series[k].length), 1);
    const yMax = Math.max(1, ...keys.flatMap((k) => series[k]));
    const x = (i) => left + (i / Math.max(n - 1, 1)) * (W - left - 10);
    const y = (v) => 6 + (1 - v / yMax) * (H - bottom - 6);
    const axis = el("g", { class: "axis" }, svg);
    for (const v of [0, yMax / 2, yMax]) {
      el("line", { x1: left, x2: W - 10, y1: y(v), y2: y(v) }, axis);
      el("text", { x: left - 6, y: y(v) + 4, "text-anchor": "end" }, axis).textContent = Math.round(v);
    }
    for (const i of niceTicks(n - 1, 8)) el("text", { x: x(i), y: H - 4, "text-anchor": "middle" }, axis).textContent = xLabel(i);
    for (const k of keys) {
      const d = series[k].map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
      el("path", { d, stroke: colors[k] || "#ccc", "stroke-width": 1.6, fill: "none" }, svg);
    }
    const legend = html("div", "chips");
    for (const k of keys) { const c = html("span", "chip"); c.innerHTML = `<span class="swatch" style="background:${colors[k] || "#ccc"}"></span>${k}`; legend.appendChild(c); }
    box.appendChild(legend);
    host.appendChild(box);
  }

  /** Horizontal bar chart: items = [{label, value, color}]. */
  function barChart(host, items, { title = "", unit = "" } = {}) {
    const box = html("div", "chart");
    box.appendChild(html("h4", null, title));
    const W = 600, rowH = 26, left = 190, H = Math.max(1, items.length) * rowH + 6;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
    const vMax = Math.max(1e-9, ...items.map((i) => i.value));
    items.forEach((it, i) => {
      el("text", { x: left - 8, y: i * rowH + 18, "text-anchor": "end", fill: "#c7d1e2", "font-size": 13 }, svg).textContent = it.label;
      const w = ((W - left - 60) * it.value) / vMax;
      el("rect", { x: left, y: i * rowH + 5, width: Math.max(2, w), height: rowH - 10, rx: 3, fill: it.color || "#36c2ff" }, svg);
      el("text", { x: left + w + 6, y: i * rowH + 18, fill: "#93a1b8", "font-size": 12 }, svg).textContent = `${+it.value.toFixed(2)}${unit}`;
    });
    if (!items.length) el("text", { x: W / 2, y: 16, "text-anchor": "middle", fill: "#93a1b8" }, svg).textContent = "no data";
    host.appendChild(box);
  }

  function histogram(host, values, { title = "", bins = 20, color = "#7cf0c8", xMax } = {}) {
    const max = xMax || Math.max(1e-9, ...values);
    const counts = new Array(bins).fill(0);
    for (const v of values) counts[Math.min(bins - 1, Math.floor((v / max) * bins))]++;
    const W = 600, H = 160, left = 30, bottom = 20;
    const box = html("div", "chart");
    box.appendChild(html("h4", null, title));
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
    const cMax = Math.max(1, ...counts), bw = (W - left - 10) / bins;
    counts.forEach((c, i) => el("rect", { x: left + i * bw + 1, y: 6 + (1 - c / cMax) * (H - bottom - 6), width: bw - 2, height: (c / cMax) * (H - bottom - 6), fill: color, rx: 2 }, svg));
    const axis = el("g", { class: "axis" }, svg);
    for (const k of [0, bins / 2, bins]) el("text", { x: left + k * bw, y: H - 4, "text-anchor": "middle" }, axis).textContent = ((k / bins) * max).toFixed(1);
    host.appendChild(box);
  }

  function niceTicks(max, n) {
    if (max <= 0) return [0];
    const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const out = [];
    for (let t = 0; t <= max + 1e-9; t += step) out.push(t);
    return out;
  }
  function svgPoint(svg, e) {
    const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    return p.matrixTransform(svg.getScreenCTM().inverse());
  }

  /** Box overlay drawn on a canvas above a <video>, synced to currentTime. */
  function attachOverlay(video, canvas, result) {
    const tracks = result.tracks.map((tr) => ({ ...tr, t0: tr.t[0], t1: tr.t[tr.t.length - 1] }));
    const events = result.events;
    const W = result.meta.width, H = result.meta.height;
    const ctx = canvas.getContext("2d");
    function draw() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(rect.width * dpr)) { canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr); }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const sx = canvas.width / W, sy = canvas.height / H, t = video.currentTime;
      const active = events.filter((e) => e[0] <= t && t <= e[1]);
      const hot = new Set(active.flatMap((e) => e[4] || []));
      ctx.lineWidth = 2 * dpr; ctx.font = `${11 * dpr}px system-ui`;
      for (const tr of tracks) {
        if (t < tr.t0 - 0.2 || t > tr.t1 + 0.2) continue;
        let i = bisect(tr.t, t);
        if (Math.abs(tr.t[i] - t) > 0.3) continue;
        const [x1, y1, x2, y2] = interpBox(tr, t, i);
        const col = hot.has(tr.id) ? "#ff5a6a" : CAT_COLORS[tr.cat] || "#fff";
        ctx.strokeStyle = col; ctx.lineWidth = (hot.has(tr.id) ? 3.5 : 2) * dpr;
        ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
        ctx.fillStyle = col; ctx.fillText(`${tr.name} ${tr.id}`, x1 * sx, y1 * sy - 3 * dpr);
      }
      active.forEach((e, k) => {
        const info = CLASS_INFO[e[2]] || { color: "#ccc", name: e[2] };
        const label = `${info.name.toUpperCase()}  ${fmt(e[0])}–${fmt(e[1])}`;
        ctx.font = `600 ${13 * dpr}px system-ui`;
        const w = ctx.measureText(label).width + 16 * dpr;
        ctx.fillStyle = info.color; ctx.fillRect(10 * dpr, (10 + 28 * k) * dpr, w, 24 * dpr);
        ctx.fillStyle = "#081018"; ctx.fillText(label, 18 * dpr, (27 + 28 * k) * dpr);
      });
      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
  }
  function bisect(arr, t) {
    let lo = 0, hi = arr.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < t) lo = m + 1; else hi = m; }
    return lo;
  }
  function interpBox(tr, t, i) {
    const j = tr.t[i] > t && i > 0 ? i - 1 : i;
    const k = Math.min(j + 1, tr.t.length - 1);
    if (k === j) return tr.box[j];
    const a = (t - tr.t[j]) / (tr.t[k] - tr.t[j] || 1);
    return tr.box[j].map((v, n) => v + (tr.box[k][n] - v) * Math.max(0, Math.min(1, a)));
  }

  /**
   * Full player: video (+ optional live overlay), event list, timeline, risk curve.
   * opts.videoUrl, opts.result, opts.overlay (draw boxes client-side)
   */
  function createPlayer(host, { videoUrl, result, overlay = false }) {
    host.innerHTML = "";
    const duration = result.meta.duration;
    const wrap = html("div", "player");
    const stage = html("div", "stage");
    const video = html("video");
    video.controls = true; video.playsInline = true; video.preload = "metadata"; video.muted = true; video.src = videoUrl;
    stage.appendChild(video);
    if (overlay) { const c = html("canvas"); stage.appendChild(c); attachOverlay(video, c, result); }
    const side = html("div", "side");
    side.appendChild(html("h4", null, `${result.events.length} event${result.events.length === 1 ? "" : "s"}`));
    const items = [];
    const seek = (t) => { video.currentTime = Math.max(0, t - 0.5); video.play().catch(() => {}); };
    [...result.events].sort((a, b) => a[0] - b[0]).forEach((ev) => {
      const info = CLASS_INFO[ev[2]] || { color: "#ccc", name: ev[2] };
      const it = html("div", "ev-item");
      it.innerHTML = `<span class="swatch" style="background:${info.color}"></span><span>${info.name}</span><span class="time">${fmt(ev[0])}–${fmt(ev[1])}</span>`;
      it.addEventListener("click", () => seek(ev[0]));
      side.appendChild(it); items.push([ev, it]);
    });
    if (!result.events.length) side.appendChild(html("p", "small", "No events detected in this clip."));
    wrap.append(stage, side);
    host.appendChild(wrap);
    const tl = timelineChart(host, result.events, duration, seek);
    const rc = lineChart(host, result.risk || [], { title: "Accident risk — P(accident starts within 5 s)", duration, threshold: 0.5, color: "#ff9f43", onSeek: seek, events: result.events });
    video.addEventListener("timeupdate", () => {
      const t = video.currentTime;
      tl.setTime(t); rc.setTime(t);
      for (const [ev, it] of items) it.classList.toggle("active", ev[0] <= t && t <= ev[1]);
    });
    return video;
  }

  window.Charts = { CLASS_INFO, CAT_COLORS, timelineChart, lineChart, seriesChart, barChart, histogram, createPlayer, fmt, html };
})();
