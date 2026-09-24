/* Page logic: fills sections from website/data/*.json and drives the live demo. */
(function () {
  const { CLASS_INFO, CAT_COLORS, html, fmt } = window.Charts;
  const API = (window.APP_CONFIG && window.APP_CONFIG.apiBase) || "";

  const RULES = [
    ["accident", "Boxes touch at the same ground depth after fast closing, then hard braking / swerve / fall / both stay stopped", "first contact → all involved stop or leave", "rule"],
    ["near_miss", "Predicted time-to-collision < 1.2 s with miss distance < 1 body, plus hard braking or swerve, no contact", "evasive action → road users clear", "rule"],
    ["red_light", "Vehicle front crosses a stop line in its approach direction while the signal-head ROI reads red", "crosses line → leaves intersection", "pixel"],
    ["wrong_way", "Moving ≥ 1.5 s against the lane direction (scene lanes or the learned flow field)", "enters opposing flow → back / leaves", "rule"],
    ["illegal_u_turn", "Heading reverses ≥ 150° within 20 s outside zones where U-turns are allowed", "starts turning → completes turn", "rule"],
    ["stopped_vehicle", "Stationary ≥ 10 s on the carriageway, not in a learned queue zone and not released with its queue", "stops → moves / removed", "rule"],
    ["jaywalking", "Pedestrian's feet inside the carriageway, outside every crossing, for ≥ 1 s", "steps onto road → leaves road", "rule"],
    ["failure_to_yield", "Vehicle drives through a crossing while a pedestrian is on it", "enters crossing → leaves", "rule"],
    ["illegal_turn", "Entry zone → exit zone pair listed as prohibited in the scene config", "starts turning → completes turn", "rule"],
    ["solid_line_crossing", "Bottom corners of the box cross a solid-line polyline and stay across", "wheel crosses → fully in new lane", "rule"],
    ["stop_line", "Vehicle stops with its front just past the stop line while the signal is red", "stops → signal turns green", "pixel"],
    ["congestion", "≥ 4 vehicles in a direction, ≥ 75 % of them crawling, for ≥ 30 s", "queue stops → queue clears", "rule"],
    ["road_obstacle", "Detected animal / loose object on the road, or a static foreground blob nobody detected", "appears → removed", "pixel"],
    ["fire_smoke", "Flickering saturated orange pixels (minus static lamps), or a grey, low-texture, growing blob", "first smoke → clears", "pixel"],
  ];

  const TEAM = [
    { name: "Member 1", role: "Team lead · CV pipeline", did: ["Detection & tracking", "Event rules", "Evaluation"], links: {} },
    { name: "Member 2", role: "Data & modelling", did: ["Dev-set annotation", "EDA", "Accident / near-miss tuning"], links: {} },
    { name: "Member 3", role: "Web & demo", did: ["Website", "Live demo backend", "Visualisations"], links: {} },
  ];

  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-cache" });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }

  // ------------------------------------------------------------------ static sections
  function renderRules() {
    const tbody = document.getElementById("rules-table");
    const tagName = { rule: "trajectory rule", pixel: "rule + pixel cue", learned: "learned" };
    for (const [id, signal, span, kind] of RULES) {
      const info = CLASS_INFO[id];
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><span class="swatch" style="background:${info.color}"></span><b>${info.name}</b><br><code>${id}</code></td>
        <td>${signal}</td><td>${span}</td><td><span class="tag ${kind}">${tagName[kind]}</span></td>`;
      tbody.appendChild(tr);
    }
  }

  function renderTeam() {
    const host = document.getElementById("team-cards");
    for (const m of TEAM) {
      const card = html("article", "card member");
      const initials = m.name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
      const links = Object.entries(m.links).map(([k, v]) => `<a href="${v}" target="_blank" rel="noopener">${k}</a>`).join("");
      card.innerHTML = `<div class="avatar">${initials}</div><h3>${m.name}</h3><div class="role">${m.role}</div>
        <ul>${m.did.map((d) => `<li>${d}</li>`).join("")}</ul>
        <div class="socials">${links || '<span class="small">GitHub · LinkedIn · portfolio — coming soon</span>'}</div>`;
      host.appendChild(card);
    }
  }

  function kpi(label, value) {
    const d = html("div", "kpi");
    d.innerHTML = `<b>${value}</b><span>${label}</span>`;
    return d;
  }

  // ------------------------------------------------------------------ results
  async function loadResults() {
    let index;
    try { index = await getJSON("data/index.json"); } catch { index = []; }
    const kpis = document.getElementById("kpis");
    const hours = index.reduce((s, v) => s + v.duration, 0) / 3600;
    const nEvents = index.reduce((s, v) => s + v.n_events, 0);
    kpis.append(kpi("event classes", 14), kpi("sample videos analysed", index.length || "—"),
      kpi("hours of footage", index.length ? hours.toFixed(1) : "—"), kpi("events found", index.length ? nEvents : "—"),
      kpi("paid APIs used", 0));
    if (!index.length) return [];

    const host = document.getElementById("results-content");
    host.innerHTML = "";
    const tabs = html("div", "tabs");
    const view = html("div");
    host.append(tabs, view);
    const results = {};
    const show = async (v, btn) => {
      tabs.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
      view.innerHTML = '<p class="small">loading…</p>';
      results[v.stem] = results[v.stem] || await getJSON(`data/results/${v.stem}.json`);
      const res = results[v.stem];
      Charts.createPlayer(view, { videoUrl: `data/videos/${v.stem}.mp4`, result: res, overlay: false });
      const note = html("p", "small", `${v.video} · ${fmt(res.meta.duration)} · ${res.meta.width}×${res.meta.height} @ ${res.meta.fps} fps · ` +
        `${res.stats.n_tracks} tracks · analysed in ${res.stats.seconds}s (stride ${res.stats.stride})`);
      view.appendChild(note);
    };
    index.forEach((v, i) => {
      const b = html("button", "tab", v.video.replace(/\.mp4$/, ""));
      b.addEventListener("click", () => show(v, b));
      tabs.appendChild(b);
      if (i === 0) show(v, b);
    });
    const all = await Promise.all(index.map((v) => getJSON(`data/results/${v.stem}.json`).catch(() => null)));
    return all.filter(Boolean);
  }

  // ------------------------------------------------------------------ dashboard
  function renderDashboard(results) {
    if (!results.length) return;
    const host = document.getElementById("dashboard-content");
    host.innerHTML = "";
    const hours = results.reduce((s, r) => s + r.meta.duration, 0) / 3600;
    const byClass = {};
    for (const r of results) for (const e of r.events) byClass[e[2]] = (byClass[e[2]] || 0) + 1;
    const grid = html("div", "cards two");
    host.appendChild(grid);
    const a = html("div"), b = html("div");
    grid.append(a, b);
    Charts.barChart(a, Object.entries(byClass).sort((x, y) => y[1] - x[1]).map(([c, n]) => ({ label: CLASS_INFO[c]?.name || c, value: n, color: CLASS_INFO[c]?.color })), { title: "Events by class (all samples)" });
    Charts.barChart(b, Object.entries(byClass).sort((x, y) => y[1] - x[1]).map(([c, n]) => ({ label: CLASS_INFO[c]?.name || c, value: n / Math.max(hours, 1e-9), color: CLASS_INFO[c]?.color })), { title: "Events per hour of footage", unit: "/h" });
    const rows = results.map((r) => {
      const counts = {};
      for (const e of r.events) counts[e[2]] = (counts[e[2]] || 0) + 1;
      const risk = r.risk.length ? Math.max(...r.risk.map((p) => p[1])) : 0;
      return `<tr><td>${r.video}</td><td>${fmt(r.meta.duration)}</td><td>${r.events.length}</td>
        <td>${Object.entries(counts).map(([c, n]) => `<span class="tag rule" style="color:${CLASS_INFO[c]?.color}">${CLASS_INFO[c]?.name || c} × ${n}</span>`).join(" ") || "—"}</td>
        <td>${risk.toFixed(2)}</td><td>${r.stats.seconds}s</td></tr>`;
    }).join("");
    const t = html("div", "table-wrap mt");
    t.innerHTML = `<table><thead><tr><th>Video</th><th>Length</th><th>Events</th><th>Breakdown</th><th>Peak risk</th><th>Runtime</th></tr></thead><tbody>${rows}</tbody></table>`;
    host.appendChild(t);
  }

  // ------------------------------------------------------------------ EDA
  async function loadEDA() {
    let eda;
    try { eda = await getJSON("data/eda/eda.json"); } catch { return; }
    if (!eda.videos || !eda.videos.length) return;
    const host = document.getElementById("eda-content");
    host.innerHTML = "";
    const rows = eda.videos.map(({ props: p }) => `<tr><td>${p.video}</td><td>${p.width}×${p.height}</td><td>${p.fps}</td>
      <td>${fmt(p.duration)}</td><td>${p.codec}</td><td>${p.bitrate_mbps} Mb/s</td><td>${p.lighting}</td></tr>`).join("");
    const table = html("div", "table-wrap");
    table.innerHTML = `<table><thead><tr><th>Video</th><th>Resolution</th><th>FPS</th><th>Duration</th><th>Codec</th><th>Bitrate</th><th>Lighting</th></tr></thead><tbody>${rows}</tbody></table>`;
    host.appendChild(table);

    const totals = { vehicle: 0, bike: 0, person: 0, obstacle: 0 };
    for (const v of eda.videos) for (const k in v.n_tracks || {}) totals[k] += v.n_tracks[k];
    const grid = html("div", "cards two mt");
    host.appendChild(grid);
    const c1 = html("div"), c2 = html("div");
    grid.append(c1, c2);
    Charts.barChart(c1, Object.entries(totals).map(([k, v]) => ({ label: k, value: v, color: CAT_COLORS[k] })), { title: "Tracked road users across all samples" });
    Charts.histogram(c2, eda.videos.flatMap((v) => v.speeds || []), { title: "Vehicle speed distribution (body units / s)", xMax: 6 });

    if (eda.flow_field) {
      const f = html("div", "card mt");
      f.innerHTML = `<h3>Learned lane directions</h3><img loading="lazy" src="${eda.flow_field}" alt="Flow field arrows">
        <p class="cap">Dominant travel direction per grid cell, learned from every vehicle trajectory in the samples (colour = direction). Cells with mixed directions are left empty; the wrong-way rule only fires where the direction is clear.</p>`;
      host.appendChild(f);
    }

    const tabs = html("div", "tabs mt");
    const view = html("div");
    host.append(tabs, view);
    eda.videos.forEach((v, i) => {
      const stem = v.props.video.replace(/\.mp4$/, "");
      const b = html("button", "tab", stem);
      const show = () => {
        tabs.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
        view.innerHTML = "";
        if (v.counts_per_minute) Charts.seriesChart(view, v.counts_per_minute, { title: "Road users in view per second (mean per minute)", xLabel: (i) => `${Math.round(i)}m` });
        Charts.seriesChart(view, { brightness: v.props.brightness.mean, contrast: v.props.brightness.std },
          { title: "Lighting: mean brightness and contrast over time", xLabel: (i) => fmt(v.props.brightness.t[Math.round(i)] || 0), colors: { brightness: "#feca57", contrast: "#8395a7" } });
        const imgs = html("div", "grid-imgs mt");
        for (const [key, cap] of [["vehicles_heatmap", "Where vehicles drive (ground-point density)"], ["pedestrian_heatmap", "Where pedestrians walk"], ["trajectories", "Vehicle trajectories, coloured by direction"]]) {
          const card = html("div", "card");
          card.innerHTML = `<img loading="lazy" src="data/eda/${stem}_${key}.jpg" alt="${cap}"><p class="cap">${cap}</p>`;
          imgs.appendChild(card);
        }
        view.appendChild(imgs);
      };
      b.addEventListener("click", show);
      tabs.appendChild(b);
      if (i === 0) show();
    });
  }

  // ------------------------------------------------------------------ demo
  function setupDemo() {
    const form = document.getElementById("upload-form");
    const input = document.getElementById("file");
    const drop = document.getElementById("drop");
    const btn = document.getElementById("upload-btn");
    const status = document.getElementById("demo-status");
    const bar = document.getElementById("demo-bar");
    const msg = document.getElementById("demo-msg");
    const out = document.getElementById("demo-result");
    const offline = document.getElementById("demo-offline");

    fetch(`${API}/api/health`).then((r) => { if (!r.ok) throw 0; }).catch(() => { offline.hidden = false; });

    const pick = (f) => {
      if (!f) return;
      if (!/\.mp4$/i.test(f.name)) { document.getElementById("drop-text").textContent = "Please choose an .mp4 file"; return; }
      if (f.size > 200 * 2 ** 20) { document.getElementById("drop-text").textContent = "File is larger than 200 MB"; return; }
      document.getElementById("drop-text").textContent = `${f.name} · ${(f.size / 2 ** 20).toFixed(1)} MB`;
      btn.disabled = false;
      form._file = f;
    };
    input.addEventListener("change", () => pick(input.files[0]));
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));

    const setProgress = (p, text) => { bar.style.width = `${Math.round(p * 100)}%`; msg.textContent = text; };

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      if (!form._file) return;
      btn.disabled = true; status.hidden = false; out.innerHTML = "";
      const fd = new FormData();
      fd.append("file", form._file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API}/api/jobs`);
      xhr.upload.onprogress = (ev) => ev.lengthComputable && setProgress(0.1 * ev.loaded / ev.total, `Uploading… ${Math.round(100 * ev.loaded / ev.total)}%`);
      xhr.onerror = () => { setProgress(0, "Upload failed — is the demo backend running?"); btn.disabled = false; };
      xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch {}
        if (xhr.status !== 200) { setProgress(0, body.detail || `Upload failed (${xhr.status})`); btn.disabled = false; return; }
        poll(body.id);
      };
      xhr.send(fd);
    });

    async function poll(id) {
      const t0 = performance.now();
      for (;;) {
        let s;
        try { s = await getJSON(`${API}/api/jobs/${id}`); } catch (err) { setProgress(0, `Lost contact with the server: ${err.message}`); break; }
        const elapsed = ((performance.now() - t0) / 1000).toFixed(0);
        if (s.status === "done") { setProgress(1, `Done in ${s.result.stats.total_seconds}s — ${s.result.events.length} events.`); showResult(id, s.result); break; }
        if (s.status === "error") { setProgress(0, s.message); break; }
        setProgress(0.1 + 0.9 * s.progress, `${s.status === "queued" ? "Waiting in queue" : s.message}… ${Math.round(100 * s.progress)}% · ${elapsed}s`);
        await new Promise((r) => setTimeout(r, 1000));
      }
      btn.disabled = false;
    }

    function showResult(id, result) {
      out.innerHTML = "";
      Charts.createPlayer(out, { videoUrl: `${API}/api/jobs/${id}/video`, result, overlay: true });
      const dl = html("button", "btn mt", "Download events JSON");
      dl.addEventListener("click", () => {
        const blob = new Blob([JSON.stringify({ events: result.events.map((e) => e.slice(0, 3)), risk: result.risk }, null, 1)], { type: "application/json" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob); a.download = `${result.video.replace(/\.mp4$/i, "")}_events.json`; a.click();
      });
      out.appendChild(dl);
    }
  }

  renderRules();
  renderTeam();
  setupDemo();
  loadEDA();
  loadResults().then(renderDashboard);
})();
