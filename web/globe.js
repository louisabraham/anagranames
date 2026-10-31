// The globe of the anagrams: the countries of Natural Earth (globe.json, made by web/globe_prep.py) on a sphere,
// drawn as the screen of an oscilloscope. It turns slowly on its own (not for a visitor who asks for less motion), turns
// with the mouse or a finger (on a touch screen, a drag across; a drag down scrolls the page), shows the numbers of the
// country under the pointer (a dot, a line, a box), and chooses a country with a click.
// Orthographic projection: a point (lon, lat) of the earth is a unit vector; the view turns it so that the center of
// the view (this.lon, this.lat) faces the screen; z > 0: the point is on the side of the screen.
const DEG = Math.PI / 180;
const CYAN = "0, 255, 229";
const SPIN = 4; // degrees per second, on its own
const lessMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const clampLat = (lat) => Math.max(-70, Math.min(70, lat));

/** The unit vectors of points given in tenths of a degree (lon, lat, lon, lat...). */
function vectors(flat) {
  const n = flat.length / 2, v = { x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n) };
  for (let i = 0; i < n; i++) {
    const lon = (flat[2 * i] / 10) * DEG, lat = (flat[2 * i + 1] / 10) * DEG;
    v.x[i] = Math.cos(lat) * Math.cos(lon);
    v.y[i] = Math.cos(lat) * Math.sin(lon);
    v.z[i] = Math.sin(lat);
  }
  return v;
}

/** A ring of the map: its points (degrees, for the test of a point in it), their unit vectors, and their place on the
 * screen at the last frame. */
function ring(flat) {
  const n = flat.length / 2, lon = new Float32Array(n), lat = new Float32Array(n);
  for (let i = 0; i < n; i++) [lon[i], lat[i]] = [flat[2 * i] / 10, flat[2 * i + 1] / 10];
  return { lon, lat, box: [Math.min(...lon), Math.min(...lat), Math.max(...lon), Math.max(...lat)], v: vectors(flat),
    sx: new Float32Array(n), sy: new Float32Array(n), sz: new Float32Array(n) };
}

/** The lines of longitude and latitude, every 15 degrees. */
function graticule() {
  const lines = [];
  for (let lon = -180; lon < 180; lon += 15) lines.push(Array.from({ length: 61 }, (_, i) => [lon * 10, (-90 + 3 * i) * 10]).flat());
  for (let lat = -75; lat <= 75; lat += 15) lines.push(Array.from({ length: 121 }, (_, i) => [(-180 + 3 * i) * 10, lat * 10]).flat());
  return lines.map(ring);
}

export class Globe {
  constructor(box, data, { pick, info }) {
    Object.assign(this, { box, pick, info, lon: 10, lat: 22, values: new Map(), hover: null, chosen: null });
    this.canvas = box.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.tip = box.querySelector(".globe-tip");
    this.countries = data.countries.map((c) => ({ code: c.c, name: c.n, rings: c.p.map(ring),
      label: { lon: c.x, lat: c.y, v: vectors([c.x * 10, c.y * 10]) } }));
    this.byCode = new Map(this.countries.filter((c) => c.code).map((c) => [c.code, c]));
    this.lines = graticule();
    this.raf = 0;
    this.last = 0;
    this.visible = true;
    box.globe = this; // for the tests
    new ResizeObserver(() => this.resize()).observe(box);
    new IntersectionObserver(([e]) => {
      this.visible = e.isIntersecting;
      if (this.visible) this.wake();
    }).observe(box);
    this.listen();
    this.resize();
  }

  // ---------- data and choice ----------
  /** values: code -> { real, synthetic }; mode: "real" (the countries lit by their real people, blips) or
   * "synthetic" (lit by the synthetic people expected). */
  setData(values, mode = "real") {
    this.values = values;
    this.mode = mode;
    this.maxReal = Math.max(1, ...[...values.values()].map((v) => v.real));
    this.maxSynthetic = Math.max(1e-9, ...[...values.values()].map((v) => v.synthetic));
    this.wake();
  }

  /** The chosen country (null: the world): the globe turns to it. */
  select(code) {
    if (code === this.chosen) return;
    this.chosen = code;
    const c = code && this.byCode.get(code);
    if (c) this.turnTo(c.label.lon, c.label.lat);
    this.box.dataset.chosen = code ?? "";
    this.wake();
  }

  turnTo(lon, lat) {
    const to = { lon, lat: clampLat(lat) }, d = ((to.lon - this.lon + 540) % 360) - 180;
    if (lessMotion()) [this.lon, this.lat, this.tween] = [to.lon, to.lat, null];
    else this.tween = { t0: performance.now(), lon: this.lon, lat: this.lat, dLon: d, dLat: to.lat - this.lat, ms: 900 };
    this.spin = 0;
  }

  /** The place of a country on the screen (its label), relative to the box; null on the other side. For the tests. */
  where(code) {
    const c = this.byCode.get(code);
    if (!c) return null;
    const [x, y, z] = this.toScreen(c.label.v, 0);
    return z > 0 ? { x, y } : null;
  }

  // ---------- projection ----------
  view() {
    const lon = this.lon * DEG, lat = this.lat * DEG;
    this.m = { cl: Math.cos(lon), sl: Math.sin(lon), cp: Math.cos(lat), sp: Math.sin(lat) };
  }

  /** Point i of the vectors v on the screen: [x, y, z] (z > 0: this side). */
  toScreen(v, i) {
    const { cl, sl, cp, sp } = this.m, a = cl * v.x[i] + sl * v.y[i];
    const x = -sl * v.x[i] + cl * v.y[i], y = cp * v.z[i] - sp * a, z = sp * v.z[i] + cp * a;
    return [this.cx + this.R * x, this.cy - this.R * y, z];
  }

  /** The point of the earth under a point of the box, in degrees; null off the globe. */
  toEarth(px, py) {
    const x = (px - this.cx) / this.R, y = (this.cy - py) / this.R, d = x * x + y * y;
    if (d > 1) return null;
    const z = Math.sqrt(1 - d), { cl, sl, cp, sp } = this.m;
    const a = cp * z - sp * y, ez = sp * z + cp * y; // back from the view: a = cl ex + sl ey
    const ex = cl * a - sl * x, ey = sl * a + cl * x;
    return { lon: Math.atan2(ey, ex) / DEG, lat: Math.asin(Math.max(-1, Math.min(1, ez))) / DEG };
  }

  /** The country at a point of the earth (a test of the point in its rings). */
  countryAt(p) {
    if (!p) return null;
    for (const c of this.countries) {
      for (const r of c.rings) {
        const [x0, y0, x1, y1] = r.box;
        if (p.lon < x0 || p.lon > x1 || p.lat < y0 || p.lat > y1) continue;
        let inside = false;
        for (let i = 0, j = r.lon.length - 1; i < r.lon.length; j = i++) {
          if ((r.lat[i] > p.lat) !== (r.lat[j] > p.lat)
            && p.lon < ((r.lon[j] - r.lon[i]) * (p.lat - r.lat[i])) / (r.lat[j] - r.lat[i]) + r.lon[i]) inside = !inside;
        }
        if (inside) return c;
      }
    }
    return null;
  }

  // ---------- drawing ----------
  resize() {
    const w = this.box.clientWidth, h = this.box.clientHeight, dpr = devicePixelRatio || 1;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    Object.assign(this, { w, h, cx: w / 2, cy: h / 2, R: Math.max(40, Math.min(w, h) / 2 - 18) });
    this.wake();
  }

  /** The path of a ring on the screen, cut at the edge of the globe: fill (closed, along the edge where it is cut)
   * or a line. False: nothing on this side. */
  trace(r, fill) {
    const { ctx, cx, cy, R } = this, n = r.v.x.length;
    let any = false;
    for (let i = 0; i < n; i++) {
      [r.sx[i], r.sy[i], r.sz[i]] = this.toScreen(r.v, i);
      any ||= r.sz[i] > 0;
    }
    if (!any) return false;
    const edge = (i, j) => { // the point where the segment i-j crosses the edge, and its angle
      const t = r.sz[i] / (r.sz[i] - r.sz[j]);
      const x = r.sx[i] + t * (r.sx[j] - r.sx[i]) - cx, y = r.sy[i] + t * (r.sy[j] - r.sy[i]) - cy;
      const a = Math.atan2(y, x);
      return [cx + R * Math.cos(a), cy + R * Math.sin(a), a];
    };
    if (!fill) { // an open line: the parts on this side
      let pen = r.sz[0] > 0;
      if (pen) ctx.moveTo(r.sx[0], r.sy[0]);
      for (let j = 1; j < n; j++) {
        const i = j - 1;
        if (r.sz[i] > 0 && r.sz[j] > 0) ctx.lineTo(r.sx[j], r.sy[j]);
        else if (r.sz[i] > 0) ctx.lineTo(...edge(i, j).slice(0, 2));
        else if (r.sz[j] > 0) {
          ctx.moveTo(...edge(i, j).slice(0, 2));
          ctx.lineTo(r.sx[j], r.sy[j]);
        }
      }
      return true;
    }
    if (r.sz.every((z) => z > 0)) {
      ctx.moveTo(r.sx[0], r.sy[0]);
      for (let i = 1; i < n; i++) ctx.lineTo(r.sx[i], r.sy[i]);
      ctx.closePath();
      return true;
    }
    // a closed ring cut by the edge: from each point where it comes to this side to the point where it leaves, then
    // along the edge of the globe (the short way) to the next point where it comes back
    let start = 0; // a point of this side after a point of the other side
    while (!(r.sz[start] > 0 && r.sz[(start - 1 + n) % n] <= 0)) start++;
    let out = null, first = null;
    const along = (from, to) => ctx.arc(cx, cy, R, from, to, ((to - from + 3 * Math.PI) % (2 * Math.PI)) - Math.PI < 0);
    for (let k = 0; k < n; k++) {
      const j = (start + k) % n, i = (j - 1 + n) % n;
      if (r.sz[i] <= 0 && r.sz[j] > 0) {
        const [x, y, a] = edge(i, j);
        if (out === null) ctx.moveTo(x, y);
        else along(out, a);
        first ??= a;
        ctx.lineTo(r.sx[j], r.sy[j]);
      } else if (r.sz[i] > 0 && r.sz[j] > 0) ctx.lineTo(r.sx[j], r.sy[j]);
      else if (r.sz[i] > 0) {
        const [x, y, a] = edge(i, j);
        ctx.lineTo(x, y);
        out = a;
      }
    }
    if (out !== null) along(out, first);
    ctx.closePath();
    return true;
  }

  draw(now) {
    const { ctx, cx, cy, R, w, h } = this;
    this.view();
    ctx.clearRect(0, 0, w, h);
    // the sphere: dark, a halo
    const g = ctx.createRadialGradient(cx - R * 0.3, cy - R * 0.35, R * 0.1, cx, cy, R);
    g.addColorStop(0, "rgba(0, 60, 70, 0.55)");
    g.addColorStop(1, "rgba(0, 12, 18, 0.9)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.fill();
    for (const [width, alpha] of [[6, 0.08], [1.2, 0.85]]) {
      ctx.strokeStyle = `rgba(${CYAN}, ${alpha})`;
      ctx.lineWidth = width;
      ctx.stroke();
    }
    // the lines of longitude and latitude
    ctx.beginPath();
    for (const l of this.lines) this.trace(l, false);
    ctx.strokeStyle = `rgba(${CYAN}, 0.13)`;
    ctx.lineWidth = 0.7;
    ctx.stroke();
    // the countries: a fill for the names possible in it, an outline; the chosen and the pointed countries brighter
    for (const c of this.countries) {
      const v = this.values.get(c.code), chosen = c.code && c.code === this.chosen, pointed = c === this.hover;
      ctx.beginPath();
      let seen = false;
      for (const r of c.rings) seen = this.trace(r, true) || seen;
      if (!seen) continue;
      const x = !v ? 0 : this.mode === "real" ? v.real / this.maxReal : v.synthetic / this.maxSynthetic;
      const fill = chosen ? 0.62 : pointed ? 0.45 : x ? 0.06 + 0.34 * Math.sqrt(x) : 0;
      if (fill) {
        ctx.fillStyle = `rgba(${CYAN}, ${fill})`;
        ctx.fill();
      }
      ctx.lineWidth = chosen || pointed ? 1.6 : 0.8;
      ctx.strokeStyle = chosen || pointed ? "rgba(235, 255, 252, 0.95)" : `rgba(${CYAN}, ${x ? 0.75 : 0.3})`;
      ctx.stroke();
    }
    // the real people: blips at the places of the countries
    for (const [code, v] of this.mode === "real" ? this.values : []) {
      const c = this.byCode.get(code);
      if (!c || !v.real) continue;
      const [x, y, z] = this.toScreen(c.label.v, 0);
      if (z < 0.08) continue;
      const r = 1.8 + 5 * Math.sqrt(v.real / this.maxReal);
      ctx.fillStyle = `rgba(${CYAN}, 0.18)`;
      ctx.beginPath();
      ctx.arc(x, y, r * 2.4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = "rgba(235, 255, 252, 0.95)";
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fill();
    }
    // the chosen country: a ring that grows (still with less motion)
    const chosen = this.chosen && this.byCode.get(this.chosen);
    if (chosen) {
      const [x, y, z] = this.toScreen(chosen.label.v, 0);
      if (z > 0) {
        const t = lessMotion() ? 0.5 : ((now / 1600) % 1);
        ctx.strokeStyle = `rgba(255, 255, 255, ${0.9 * (1 - t)})`;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(x, y, 6 + 18 * t, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
    this.lead();
  }

  /** The line from the country under the pointer (else the chosen country) to its box: a dot, an oblique line, a
   * horizontal line, then the box of its numbers. */
  lead() {
    const c = this.hover ?? (this.chosen && this.byCode.get(this.chosen));
    const info = c && this.info(c.code, c.name);
    const at = c && this.toScreen(c.label.v, 0);
    if (!info || !at || at[2] <= 0) {
      this.tip.hidden = true;
      return;
    }
    const { ctx } = this, [x, y] = at, dir = x < this.cx ? -1 : 1, rise = Math.min(34, y - 24);
    const bend = [x + dir * 26, y - rise], end = [bend[0] + dir * 46, y - rise];
    ctx.strokeStyle = "rgba(235, 255, 252, 0.95)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(...bend);
    ctx.lineTo(...end);
    ctx.stroke();
    ctx.fillStyle = "rgba(235, 255, 252, 1)";
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
    const key = `${c.code}|${c.name}|${JSON.stringify(info)}`;
    if (this.tip.dataset.key !== key) {
      const title = document.createElement("b");
      title.textContent = info.title;
      const rows = info.rows.map(([label, value]) => {
        const row = document.createElement("span");
        row.className = "row";
        row.append(Object.assign(document.createElement("span"), { textContent: label }),
          Object.assign(document.createElement("span"), { textContent: value }));
        return row;
      });
      const hint = Object.assign(document.createElement("span"), { className: "hint", textContent: info.hint ?? "" });
      this.tip.replaceChildren(title, ...rows, ...(info.hint ? [hint] : []));
      this.tip.dataset.key = key;
    }
    this.tip.hidden = false;
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    const left = dir > 0 ? end[0] : end[0] - tw;
    this.tip.style.left = `${Math.max(4, Math.min(this.w - tw - 4, left))}px`;
    this.tip.style.top = `${Math.max(4, Math.min(this.h - th - 4, end[1] - th / 2))}px`;
  }

  // ---------- motion ----------
  wake() {
    if (!this.raf) this.raf = requestAnimationFrame((t) => this.frame(t));
  }

  frame(now) {
    this.raf = 0;
    if (!this.visible || !this.box.offsetParent) return; // not on the screen: no frames
    if (this.w !== this.box.clientWidth || this.h !== this.box.clientHeight) this.resize(); // it was hidden: no size
    const dt = Math.min(64, now - (this.last || now));
    this.last = now;
    let more = false;
    if (this.tween) {
      const tw = this.tween, t = Math.min(1, (now - tw.t0) / tw.ms), e = t < 0.5 ? 4 * t ** 3 : 1 - (2 - 2 * t) ** 3 / 2;
      [this.lon, this.lat] = [tw.lon + tw.dLon * e, tw.lat + tw.dLat * e];
      if (t >= 1) this.tween = null;
      more = true;
    } else if (!this.down && Math.abs(this.spin) > 0.002) { // the turn after a drag, slower and slower
      this.lon += this.spin * dt;
      this.spin *= Math.pow(0.995, dt);
      more = true;
    } else if (!this.down && !this.hover && !this.chosen && !lessMotion()) {
      this.lon += (SPIN * dt) / 1000;
      more = true;
    }
    this.draw(now);
    if (more || (this.chosen && !lessMotion())) this.wake();
    else this.last = 0;
  }

  // ---------- the mouse and the finger ----------
  listen() {
    const c = this.canvas, point = (e) => {
      const b = c.getBoundingClientRect();
      return [e.clientX - b.left, e.clientY - b.top];
    };
    const at = (e) => this.countryAt(this.toEarth(...point(e)));
    c.addEventListener("pointerdown", (e) => {
      if (e.button) return;
      const [x, y] = point(e);
      this.down = { x, y, lon: this.lon, lat: this.lat, moved: false, touch: e.pointerType === "touch", t: performance.now(), lx: x };
      this.spin = 0;
      this.tween = null;
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      const [x, y] = point(e), d = this.down;
      if (d) {
        if (!d.moved && Math.abs(x - d.x) + Math.abs(y - d.y) > 4) {
          d.moved = true;
          c.classList.add("dragging");
          this.hover = null;
        }
        if (d.moved) {
          const k = 1 / (this.R * DEG); // degrees per pixel: the earth follows the pointer at the center
          this.lon = d.lon - (x - d.x) * k;
          if (!d.touch) this.lat = clampLat(d.lat + (y - d.y) * k);
          const now = performance.now();
          if (now > d.t) this.velocity = (-(x - d.lx) * k) / (now - d.t);
          [d.t, d.lx] = [now, x];
          this.wake();
          return;
        }
      }
      const hit = at(e);
      if (hit !== this.hover) {
        this.hover = hit;
        this.box.dataset.hover = hit?.code ?? "";
        c.style.cursor = hit && this.values.has(hit.code) ? "pointer" : "";
        this.wake();
      }
    });
    const end = (e, click) => {
      const d = this.down;
      this.down = null;
      c.classList.remove("dragging");
      if (!d) return;
      if (d.moved) this.spin = performance.now() - d.t < 80 ? this.velocity ?? 0 : 0;
      else if (click) {
        const hit = at(e);
        if (hit?.code && this.values.has(hit.code)) this.pick(hit.code);
        if (d.touch) this.hover = hit; // a finger: the box of the country too
      }
      this.wake();
    };
    c.addEventListener("pointerup", (e) => end(e, true));
    c.addEventListener("pointercancel", (e) => end(e, false)); // the browser scrolls the page
    c.addEventListener("pointerleave", () => {
      if (this.down) return;
      this.hover = null;
      this.box.dataset.hover = "";
      this.wake();
    });
  }
}
