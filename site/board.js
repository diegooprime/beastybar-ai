// The table: 24 cards as persistent elements that move between hands, the queue, the bar and the out pile.
// Used by the replays, the rule demos and the play page. Positions are tweened in JavaScript from one clock, so a
// move can be interrupted at any frame and a capture script can step time (window.__bbClock).
import {ANIMALS, POINTS, animalOf, ownerOf, decode} from "./engine.js";
import {GLYPHS} from "./glyphs.js";

const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const now = () => (window.__bbClock ? window.__bbClock() : performance.now());
const ease = t => 1 - Math.pow(1 - t, 4);

export function glyph(animal, cls = "gl") {
  const g = GLYPHS[animal];
  return `<svg class="${cls}" viewBox="0 0 ${g.w} ${g.h}" aria-hidden="true"><g transform="${g.t}"><path d="${g.d}"/></g></svg>`;
}

export function cardHTML(card) {
  const a = animalOf(card);
  return `<div class="face"><span class="str">${a + 1}</span><span class="pts">${"<i></i>".repeat(POINTS[a])}</span>` +
    `${glyph(a)}<span class="nm">${ANIMALS[a]}</span></div><div class="back"></div>`;
}

// A card outside a table (rule strip, logs, option lists).
export function chip(card, extra = "") {
  return `<span class="chip s${ownerOf(card)} ${extra}" title="${ANIMALS[animalOf(card)]}">${glyph(animalOf(card))}<b>${animalOf(card) + 1}</b></span>`;
}

// What a move did, from the positions before and after it.
export function describe(before, action, after) {
  const m = decode(action), card = before.toMove * 12 + m.animal;
  const entered = after.bar.slice(before.bar.length), left = after.out.slice(before.out.length);
  const checked = entered.length === 2;
  const bounced = checked ? left.slice(0, -1) : left;
  const mid = checked ? [...entered, ...after.queue, left[left.length - 1]] : after.queue;
  let how = "";
  if (m.animal === 1) how = "bounces " + ANIMALS[animalOf(before.queue[m.params[0]])];
  else if (m.animal === 2 && m.params.length) how = "hops " + m.params[0];
  else if (m.animal === 4) {
    const t = before.queue[m.params[0]];
    how = "copies " + ANIMALS[animalOf(t)] + (m.params.length === 2 && animalOf(t) === 1
      ? ", bounces " + ANIMALS[animalOf(before.queue[m.params[1]])] : m.params.length === 2 ? ", hops " + m.params[1] : "");
  }
  return {card, animal: m.animal, how, entered, bounced, last: checked ? left[left.length - 1] : null, mid, checked,
    gain: [after.scores[0] - before.scores[0], after.scores[1] - before.scores[1]]};
}

// A move the mover could make, in words (the card itself is shown beside it).
export function optionLabel(state, action) {
  const m = decode(action), q = state.queue, name = i => ANIMALS[animalOf(q[i])];
  if (m.animal === 1) return "bounce " + name(m.params[0]);
  if (m.animal === 2 && m.params.length) return "hop " + m.params[0];
  if (m.animal === 4) return "copy " + name(m.params[0]) + (m.params.length === 2
    ? (animalOf(q[m.params[0]]) === 1 ? ", bounce " + name(m.params[1]) : ", hop " + m.params[1]) : "");
  return ANIMALS[m.animal];
}

// The search's values for a set of options, as text. Early in a game they are points (score margin plus the queue);
// once it can see the end they carry +-1000 for a win or a loss, averaged over sampled worlds: shown as a win rate.
export function valueText(values) {
  if (values.length < 2) return values.map(() => "");   // a forced move was not searched
  const decided = values.some(v => Math.abs(v) >= 50);
  return values.map(v => decided ? Math.round(Math.max(0, Math.min(1, (v / 1000 + 1) / 2)) * 100) + "%" : (v > 0 ? "+" : "") + v.toFixed(1));
}

export class Stage {
  // el: a positioned container. reveal[p]: show player p's hand face up. bottom: the seat drawn at the bottom.
  constructor(el, {reveal = [true, true], bottom = 0, names = ["", ""], onCard = null, maxCard = 118} = {}) {
    Object.assign(this, {el, reveal, bottom, names, onCard, maxCard, cards: [], state: null, run: 0, speed: 1});
    el.classList.add("stage");
    el.innerHTML = `<div class="zone bar"><h4>Bar</h4><div class="sc"><b class="s${1 - bottom}">0</b><b class="s${bottom}">0</b></div></div>` +
      `<div class="zone out"><h4>Out</h4></div><div class="slots">${"<i></i>".repeat(5)}</div>` +
      `<div class="who top s${1 - bottom}"><b></b><span></span></div><div class="who bot s${bottom}"><b></b><span></span></div>` +
      `<div class="deck top"><i></i><span></span></div><div class="deck bot"><i></i><span></span></div>`;
    this.order = Array.from({length: 24}, () => Math.random());
    for (let c = 0; c < 24; c++) {
      const d = document.createElement("div");
      d.className = `card s${ownerOf(c)}`;
      d.dataset.c = c;
      d.innerHTML = cardHTML(c);
      d.addEventListener("click", () => this.onCard && this.onCard(c, d));
      el.appendChild(d);
      this.cards.push({el: d, cur: null, from: null, to: null, t0: 0, dur: 0});
    }
    this.q = s => el.querySelector(s);
    new ResizeObserver(() => { this.measure(); if (this.state) this.set(this.state); }).observe(el);
    this.measure();
    this.tick = this.tick.bind(this);
  }

  measure() {
    const W = this.el.clientWidth || 800, side = W < 520 ? Math.max(52, W * 0.16) : W * 0.19, mid = W - 2 * side;
    const gap = Math.max(4, mid * 0.018);
    const cw = Math.min(this.maxCard, (mid - 4 * gap - 2 * gap) / 5), ch = cw * 1.4, vgap = Math.max(14, ch * 0.24);
    const label = 22;
    const g = this.g = {W, side, mid, gap, cw, ch, vgap, label, y: [label, label + ch + vgap, label + 2 * (ch + vgap)],
      qx: side + (mid - 5 * cw - 4 * gap) / 2, ms: Math.max(0.36, Math.min(0.5, (side - 20) / (cw * 3.4)))};
    g.H = g.y[2] + ch + label;
    const s = this.el.style;
    s.height = g.H + "px";
    s.setProperty("--cw", cw + "px");
    s.setProperty("--side", side + "px");
    this.el.classList.toggle("small", cw < 62);
    const slots = this.q(".slots");
    slots.style.cssText = `left:${g.qx}px;top:${g.y[1]}px;gap:${gap}px`;
    this.q(".who.top").style.cssText = `left:${side}px;width:${mid}px;top:0`;
    this.q(".who.bot").style.cssText = `left:${side}px;width:${mid}px;top:${g.H - label + 4}px`;
  }

  // Where every card belongs in `state`; `queue` overrides the queue (for the middle of a move).
  targets(state, queue = state.queue, bar = state.bar, out = state.out, hands = state.hands) {
    const g = this.g, t = new Array(24).fill(null);
    queue.forEach((c, i) => { t[c] = {x: g.qx + i * (g.cw + g.gap), y: g.y[1], s: 1, o: 1, z: 20 + i}; });
    const mw = g.cw * g.ms, mh = g.ch * g.ms, mg = Math.max(3, g.gap * 0.5);
    const cols = Math.max(2, Math.floor((g.side - 16 + mg) / (mw + mg)));
    const pile = (list, x0, top) => list.forEach((c, i) => {
      t[c] = {x: x0 + (i % cols) * (mw + mg), y: top + Math.floor(i / cols) * (mh + mg), s: g.ms, o: 1, z: 2, mini: true};
    });
    const px = (g.side - (cols * mw + (cols - 1) * mg)) / 2;
    pile(bar, px, g.label + 30);
    pile(out, g.W - g.side + px, g.label + 30);
    for (const p of [0, 1]) {
      const row = p === this.bottom ? 2 : 0, shown = this.reveal[p];
      const hand = hands[p].map(a => p * 12 + a).sort((a, b) => shown ? a - b : this.order[a] - this.order[b]);
      const x0 = g.side + (g.mid - hand.length * g.cw - (hand.length - 1) * g.gap) / 2;
      hand.forEach((c, i) => { t[c] = {x: x0 + i * (g.cw + g.gap), y: g.y[row], s: 1, o: 1, z: 10, hid: !shown, hand: true}; });
      // Cards still in the deck wait, invisible, where the deck is drawn.
      const dx = g.side + g.mid - g.cw * 0.5 - 4, dy = g.y[row] + g.ch * 0.25;
      for (let a = 0; a < 12; a++) if (!t[p * 12 + a]) t[p * 12 + a] = {x: dx, y: dy, s: 0.5, o: 0, z: 1, hid: true};
    }
    return t;
  }

  go(targets, dur) {
    const t0 = now();
    dur = still() ? 0 : dur / this.speed;
    this.cards.forEach((k, c) => {
      const to = targets[c];
      k.from = k.cur || to; k.to = to; k.t0 = t0; k.dur = k.cur ? dur : 0;
      k.el.classList.toggle("mini", !!to.mini);
      k.el.classList.toggle("hid", !!to.hid);
      k.el.classList.toggle("inhand", !!to.hand);
      k.el.style.zIndex = to.z + (k.cur && (Math.abs(k.cur.x - to.x) > 1 || Math.abs(k.cur.y - to.y) > 1) ? 40 : 0);
    });
    if (!this.raf) this.raf = requestAnimationFrame(this.tick);
    return new Promise(r => setTimeout(r, dur));
  }

  tick() {
    const t = now();
    let busy = false;
    for (const k of this.cards) {
      const p = k.dur ? Math.min(1, (t - k.t0) / k.dur) : 1, e = ease(p), f = k.from, to = k.to;
      if (!to) continue;
      k.cur = {x: f.x + (to.x - f.x) * e, y: f.y + (to.y - f.y) * e, s: f.s + (to.s - f.s) * e, o: f.o + (to.o - f.o) * e};
      k.el.style.transform = `translate(${k.cur.x.toFixed(2)}px,${k.cur.y.toFixed(2)}px) scale(${k.cur.s.toFixed(4)})`;
      k.el.style.opacity = k.cur.o.toFixed(3);
      if (p < 1) busy = true;
    }
    this.raf = busy ? requestAnimationFrame(this.tick) : 0;
  }

  chrome(state) {
    const b = this.bottom, sc = this.el.querySelectorAll(".zone.bar .sc b");
    sc[0].textContent = state.scores[1 - b];
    sc[1].textContent = state.scores[b];
    for (const [cls, p] of [["top", 1 - b], ["bot", b]]) {
      this.q(`.who.${cls} b`).textContent = this.names[p];
      this.q(`.who.${cls}`).classList.toggle("turn", !state.over && state.toMove === p);
      const d = this.q(`.deck.${cls}`), g = this.g, row = cls === "top" ? 0 : 2;
      d.style.cssText = `left:${g.side + g.mid - g.cw * 0.5 - 4}px;top:${g.y[row] + g.ch * 0.25}px;opacity:${state.deckLen[p] ? 1 : 0}`;
      d.querySelector("span").textContent = state.deckLen[p];
    }
    this.el.querySelectorAll(".slots i").forEach((s, i) => s.classList.toggle("on", i < state.queue.length));
    this.el.classList.toggle("over", state.over);
  }

  // Lay `state` out. Animated unless `instant`.
  set(state, {instant = false, dur = 260} = {}) {
    this.run++;
    this.state = state;
    this.chrome(state);
    return this.go(this.targets(state), instant ? 0 : dur);
  }

  // Play one move in its real order: the card joins the back of the queue, the animals act, the door opens.
  async move(before, action, after) {
    const run = ++this.run, d = describe(before, action, after), alive = () => run === this.run;
    this.state = after;
    const hands = before.hands.map((h, p) => p === before.toMove ? h.filter(a => a !== d.animal) : h);
    this.el.querySelector(`.card[data-c="${d.card}"]`).classList.add("played");
    await this.go(this.targets(before, [...before.queue, d.card], before.bar, before.out, hands), 240);
    if (!alive()) return d;
    const acted = d.mid.join() !== [...before.queue, d.card].join() || d.bounced.length;
    if (acted) {
      await this.go(this.targets(before, d.mid, before.bar, [...before.out, ...d.bounced], hands), 300);
      if (!alive()) return d;
    }
    if (d.checked) {
      await new Promise(r => setTimeout(r, still() ? 0 : 140 / this.speed));
      if (!alive()) return d;
    }
    this.chrome(after);
    await this.go(this.targets(after), d.checked ? 320 : 180);
    this.el.querySelectorAll(".card.played").forEach(e => e.classList.remove("played"));
    return d;
  }
}
