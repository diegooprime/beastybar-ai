import {load, decode, encode, ANIMALS, POINTS, animalOf, ownerOf} from "./engine.js";
import {Stage, describe, chip, cardHTML} from "./board.js";
import {GAMES} from "./games.js";
import {DATA} from "./data.js";

const $ = id => document.getElementById(id);
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const engine = await load("bb.wasm");
const P = Object.fromEntries(DATA.players.map(p => [p.id, p]));
const AGENT = {pimc8: "Search, 8 plies", pimc4: "Search, 4 plies", greedy: "Greedy", random: "Random"};

// ---- replay viewer -------------------------------------------------------------------------------------------
const view = {g: null, states: [], at: 0, playing: false, token: 0};
const stage = new Stage($("stage"));

function optionLabel(state, action) {
  const m = decode(action), q = state.queue, name = i => ANIMALS[animalOf(q[i])];
  if (m.animal === 1) return "bounce " + name(m.params[0]);
  if (m.animal === 2 && m.params.length) return "hop " + m.params[0];
  if (m.animal === 4) return "copy " + name(m.params[0]) + (m.params.length === 2 ? (animalOf(q[m.params[0]]) === 1 ? ", bounce " + name(m.params[1]) : ", hop " + m.params[1]) : "");
  return "";
}

function openGame(g, at = 0) {
  view.token++;
  view.g = g;
  let s = engine.newGame(g.seed, g.start);
  view.states = [s];
  for (const a of g.moves) view.states.push(s = engine.apply(a));
  stage.names = g.p.map(k => AGENT[k]);
  $("scrub").innerHTML = g.moves.map((_, i) => `<i class="s${view.states[i].toMove}"></i>`).join("");
  $("log").innerHTML = g.moves.map((a, i) => {
    const d = describe(view.states[i], a, view.states[i + 1]);
    const fx = [d.how, d.entered.length ? "in " + d.entered.map(c => chip(c)).join(" ") : "", d.bounced.length || d.last !== null
      ? "out " + [...d.bounced, ...(d.last !== null ? [d.last] : [])].map(c => chip(c, "dim")).join(" ") : ""].filter(Boolean).join(" · ");
    return `<button data-i="${i}"><span class="k">${i + 1}</span>${chip(d.card)}<span class="fx">${fx}</span></button>`;
  }).join("");
  show(at, true);
}

function paint() {
  const {g, states, at} = view, n = g.moves.length;
  $("scrub").querySelectorAll("i").forEach((e, i) => { e.classList.toggle("done", i < at); e.classList.toggle("at", i === at - 1); });
  $("plyno").textContent = `${at}/${n}`;
  $("pp").setAttribute("d", view.playing ? "M2 1h3v10H2zM7 1h3v10H7z" : "M3 1l8 5-8 5z");
  $("log").querySelectorAll("button").forEach((b, i) => {
    b.classList.toggle("at", i === at - 1);
    if (i === at - 1) $("log").scrollTop = b.offsetTop - $("log").offsetTop - 72;
  });
  // The options of the move just played (or the first move, before anything is played).
  const i = Math.max(0, at - 1), s = states[i], vals = g.vals[i], chosen = g.moves[i], mover = s.toMove;
  $("optsh").textContent = vals ? `What ${AGENT[g.p[mover]].toLowerCase()} saw at move ${i + 1}` : `Move ${i + 1}: ${AGENT[g.p[mover]]} does not search`;
  let rows = [];
  if (vals) for (let k = 0; k < vals.length; k += 2) rows.push([vals[k], vals[k + 1]]);
  else rows = [[chosen, null]];
  rows.sort((x, y) => y[1] - x[1]);
  const lo = Math.min(...rows.map(r => r[1])), hi = Math.max(...rows.map(r => r[1]));
  $("opts").innerHTML = rows.slice(0, 6).map(([a, v]) => {
    const w = v === null ? 100 : hi === lo ? 100 : 8 + 92 * (v - lo) / (hi - lo);
    return `<div class="opt s${mover} ${a === chosen ? "pick" : ""}">${chip(mover * 12 + decode(a).animal)}<span class="lbl">${optionLabel(s, a) || ANIMALS[decode(a).animal]}</span>` +
      `<span class="v"><i style="width:${w}%"></i></span><span class="n">${v === null ? "" : (v > 0 ? "+" : "") + v.toFixed(1)}</span></div>`;
  }).join("") + (rows.length > 6 ? `<div class="opt"><span></span><span class="lbl">${rows.length - 6} more</span></div>` : "");
}

function show(at, instant = false) {
  view.at = Math.max(0, Math.min(view.g.moves.length, at));
  stage.set(view.states[view.at], {instant, dur: 220});
  paint();
}

async function step() {
  const {g, states, at} = view;
  if (at >= g.moves.length) return false;
  view.at = at + 1;
  paint();
  await stage.move(states[at], g.moves[at], states[at + 1]);
  return true;
}

async function play() {
  const token = ++view.token;
  view.playing = true;
  if (view.at >= view.g.moves.length) show(0, true);
  paint();
  while (view.playing && token === view.token) {
    if (!(await step())) break;
    await new Promise(r => setTimeout(r, 620));
  }
  if (token === view.token) { view.playing = false; paint(); }
}
const pause = () => { view.playing = false; view.token++; paint(); };

$("toggle").onclick = () => (view.playing ? pause() : play());
$("prev").onclick = () => { pause(); show(view.at - 1); };
$("next").onclick = () => { pause(); step(); };
$("log").onclick = e => { const b = e.target.closest("button"); if (b) { pause(); show(+b.dataset.i + 1); } };
const scrubTo = e => {
  const r = $("scrub").getBoundingClientRect(), n = view.g.moves.length;
  pause(); show(Math.round((e.clientX - r.left) / r.width * n));
};
$("scrub").onpointerdown = e => { $("scrub").setPointerCapture(e.pointerId); scrubTo(e); $("scrub").onpointermove = scrubTo; };
$("scrub").onpointerup = () => { $("scrub").onpointermove = null; };
addEventListener("keydown", e => {
  if (e.target.matches("input, select, textarea") || e.metaKey || e.ctrlKey) return;
  if (e.key === "ArrowLeft") { pause(); show(view.at - 1); }
  else if (e.key === "ArrowRight") { pause(); step(); }
  else if (e.key === " " && document.activeElement === document.body) { e.preventDefault(); view.playing ? pause() : play(); }
});
$("pick").innerHTML = GAMES.map((g, i) => `<option value="${i}">${AGENT[g.p[0]]} ${g.score[0]} : ${g.score[1]} ${AGENT[g.p[1]]}${g.tags[0] ? " · " + g.tags[0] : ""}</option>`).join("");
$("pick").onchange = e => { openGame(GAMES[+e.target.value]); play(); };
const q = new URLSearchParams(location.search);
openGame(GAMES[+(q.get("game") || 0)], +(q.get("ply") || 0));
if (!q.has("ply") && !matchMedia("(prefers-reduced-motion: reduce)").matches) setTimeout(play, 500);

// ---- the animals ---------------------------------------------------------------------------------------------
// [rule, queue before (card ids, front first), move parameters]. Seat 0 plays the animal; card id = seat * 12 + animal.
const B = 12;
const RULES = {
  11: ["Goes to the front and throws out every monkey. A second lion is thrown out instead.", [B + 6, 3, B + 7], []],
  10: ["Every turn, pushes past anything weaker. A zebra stops it.", [B + 6, 5, B + 1], []],
  9: ["Every turn, eats the weaker animals ahead of it. A zebra stops it.", [B + 6, 5, B + 3], []],
  8: ["Sorts the whole queue, strongest in front.", [1, B + 11, 5], []],
  7: ["Every turn, steps over one weaker animal ahead of it.", [B + 11, 5], []],
  6: ["Does nothing, and nothing gets past it: hippos and crocodiles stop behind a zebra.", [B + 5, 1], []],
  5: ["Reverses the queue.", [B + 11, 3, B + 0], []],
  4: ["Copies the move of any animal in the queue.", [B + 11, 5, B + 0], [1]],
  3: ["Alone, nothing. With another monkey: hippos and crocodiles are out and the monkeys take the front.", [B + 10, B + 3, 9], []],
  2: ["Hops over the last one or two animals.", [B + 11, 5, B + 6], [2]],
  1: ["Throws out any one animal.", [B + 11, 5], [0]],
  0: ["Throws out every animal of the two strongest kinds in the queue.", [B + 11, 10, B + 5], []],
};
const order = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
$("zoo").innerHTML = order.map(a => `<button data-a="${a}" aria-label="${ANIMALS[a]}"><div class="card s0">${cardHTML(a)}</div></button>`).join("");
let demoToken = 0;
function demo(a) {
  const token = ++demoToken, [rule, queue, params] = RULES[a];
  $("zoo").querySelectorAll("button").forEach(b => b.classList.toggle("on", +b.dataset.a === a));
  const before = engine.setup(queue, [[a], [0]], 0), action = encode(a, params);
  const after = engine.apply(action), d = describe(before, action, after);
  const extra = a === 4 ? " Here it copies the seal." : a === 2 ? " Here it hops two." : a === 1 ? " Here, the lion." : "";
  $("rule").innerHTML = `<b>${ANIMALS[a]}</b><span>${rule}${extra}</span><small>strength ${a + 1} · ${POINTS[a]} points</small>`;
  const lane = $("lane"), gap = 8, cards = [...queue, d.card];
  const cw = Math.min(76, (lane.clientWidth - 80) / 4.6 - gap), x = i => i * (cw + gap), outX = 4 * (cw + gap) + 26;
  lane.style.setProperty("--cw", cw + "px");
  lane.style.height = cw * 1.4 + 26 + "px";
  lane.innerHTML = cards.map(c => `<div class="card s${ownerOf(c)}" data-c="${c}">${cardHTML(c)}</div>`).join("") +
    `<span class="tag" style="left:0;top:${cw * 1.4 + 8}px">front</span><span class="tag" style="left:${outX}px;top:${cw * 1.4 + 8}px">out</span>`;
  const put = (list, out) => {
    list.forEach((c, i) => { const e = lane.querySelector(`[data-c="${c}"]`); e.style.transform = `translate(${x(i)}px,0)`; e.style.opacity = 1; });
    out.forEach((c, i) => { const e = lane.querySelector(`[data-c="${c}"]`); e.style.transform = `translate(${outX + i * cw * .34}px,${cw * .26}px) scale(.62)`; e.style.opacity = .45; });
  };
  put(cards, []);
  lane.querySelector(`[data-c="${d.card}"]`).style.transform = `translate(${x(cards.length - 1) + 40}px,0)`;
  lane.querySelector(`[data-c="${d.card}"]`).style.opacity = 0;
  setTimeout(() => token === demoToken && put(cards, []), 60);
  setTimeout(() => token === demoToken && put(d.mid, d.bounced), 620);
}
$("zoo").onclick = e => { const b = e.target.closest("button"); if (b) demo(+b.dataset.a); };
$("zoo").onpointerover = e => { const b = e.target.closest("button"); if (b && e.pointerType === "mouse" && !b.classList.contains("on")) demo(+b.dataset.a); };
demo(11);

// ---- strength ladder -----------------------------------------------------------------------------------------
// Score of a against b from the trail, whichever way round it was played.
function score(a, b) {
  for (const m of DATA.matches) {
    if (m.argmax) continue;
    if (m.a === a && m.b === b) return {...m, flip: false};
    if (m.a === b && m.b === a) return {...m, score: 1 - m.score, lo: 1 - m.hi, hi: 1 - m.lo, flip: true};
  }
  return null;
}
const ladderIds = DATA.players.filter(p => !p.extra).map(p => p.id);
const REFS = [["bp", "vs the blueprint"], ["pimc8", "vs search, 8 plies"], ["pimc4", "vs search, 4 plies"]];
let ref = "bp";
const CLAIM = {
  bp: "Score against the blueprint network, over duplicate deals with seats swapped. The re-solvers all land near 60%: head to head they are within a point or two of each other.",
  pimc8: "Score against the search player that runs in your browser (128 sampled worlds, 8 plies deep).",
  pimc4: "Score against a shallow search (8 sampled worlds, 4 plies deep).",
};
function ladder() {
  $("refs").innerHTML = REFS.map(([id, label]) => `<button data-r="${id}" class="${id === ref ? "on" : ""}">${label}</button>`).join("");
  $("ladderclaim").textContent = CLAIM[ref];
  const rows = ladderIds.map(id => ({id, m: id === ref ? null : score(id, ref)})).filter(r => r.id === ref || r.m).reverse();
  $("ladder").innerHTML = rows.map(({id, m}) => {
    const cls = id === "c5-2" ? "best" : id === ref ? "ref" : "";
    if (!m) return `<div class="row ref"><span class="nm2">${P[id].name}</span><div class="track"><i class="mid" style="left:50%"></i></div><span class="num"><i>the opponent</i></span></div>`;
    return `<div class="row ${cls}" data-id="${id}"><span class="nm2">${P[id].name}</span><div class="track"><i class="mid" style="left:50%"></i>` +
      `<i class="ci" style="left:${m.lo * 100}%;width:${(m.hi - m.lo) * 100}%"></i><i class="dot" style="left:${m.score * 100}%"></i></div>` +
      `<span class="num">${pct(m.score)} <i>[${(m.lo * 100).toFixed(1)}, ${(m.hi * 100).toFixed(1)}]</i></span></div>`;
  }).join("");
  $("lax").innerHTML = [0, 25, 50, 75, 100].map(v => `<span class="${v % 50 ? "minor" : ""}" style="left:${v}%">${v}%</span>`).join("");
  $("prov").innerHTML = "";
}
$("refs").onclick = e => { const b = e.target.closest("button"); if (b) { ref = b.dataset.r; ladder(); } };
$("ladder").onpointerover = e => {
  const r = e.target.closest(".row[data-id]");
  if (!r) return;
  const m = score(r.dataset.id, ref);
  $("prov").innerHTML = `<b>${P[r.dataset.id].name}</b> ${P[r.dataset.id].note || ""}<br>${m.pairs} deal pairs · ${m.time.replace("T", " ")} UTC · commit ${m.commit}${m.flip ? " · played the other way round" : ""}<br>${m.command.replace(/checkpoints\/(value\/)?/g, "").slice(0, 300)}`;
};
ladder();

// ---- exploitability ------------------------------------------------------------------------------------------
let T = 8;
function exploit() {
  const rows = DATA.exploitability.filter(e => e.T === T).sort((x, y) => y.mean - x.mean);
  // PPO exploiters are a different, weaker kind of bound; shown on the ply-9 view where the exact number exists too.
  const ppo = DATA.matches.find(m => m.a === "ex-bp" && m.argmax);
  const lo = Math.log10(0.004), hi = Math.log10(0.2), x = v => (Math.log10(v) - lo) / (hi - lo) * 100;
  const line = (name, sub, mean, l, h, n, cls = "") => `<div class="row ${cls}"><span class="nm2">${name}<small>${sub}</small></span><div class="track">` +
    `<i class="ci" style="left:${x(l)}%;width:${x(h) - x(l)}%"></i><i class="dot" style="left:${x(mean)}%"></i></div>` +
    `<span class="num">${mean.toFixed(4)} <i>n=${n}</i></span></div>`;
  let html = rows.map(e => line(P[e.player].name, "", e.mean, e.lo, e.hi, e.n, e.player === "c5-2" ? "best" : "")).join("");
  if (T === 9) {
    const n = DATA.noted.bp8k_delta;
    html = line("Blueprint, step 8000", "exact", n.mean, n.lo, n.hi, 200) + html +
      line("Blueprint, step 8000", "PPO exploiter", ppo.score - 0.5, ppo.lo - 0.5, ppo.hi - 0.5, ppo.pairs);
  }
  $("expl").innerHTML = html;
  $("eax").innerHTML = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2].map(v => `<span class="${v === 0.01 || v === 0.1 ? "" : "minor"}" style="left:${x(v)}%">${v}</span>`).join("");
  $("plys").querySelectorAll("button").forEach(b => b.classList.toggle("on", +b.dataset.t === T));
}
$("plys").onclick = e => { const b = e.target.closest("button"); if (b) { T = +b.dataset.t; exploit(); } };
exploit();

// ---- how it plays --------------------------------------------------------------------------------------------
const choose4 = n => (n < 4 ? 1 : n * (n - 1) * (n - 2) * (n - 3) / 24);
const hands = ply => choose4(12 - Math.floor(ply / 2));          // hands the player who moved second could hold
const delta = t => DATA.exploitability.find(e => e.player === "bp" && e.T === t);
const KINDS = [
  {name: "Blueprint network", from: 0, to: 7, text: "A 3.4M-parameter policy trained by self-play (MMD-PPO) with a critic that sees both hands. It plays from what it has learned, with no search.",
    kv: [["Trained", "30,518 steps"], ["Thinking time", "instant"], ["Measured from here", "not yet"]]},
  {name: "Depth-limited solve", from: 8, to: 8, text: "Counterfactual regret minimization over the next two plies of the public tree. Where the tree is cut, a value network says what each possible hand is worth.",
    kv: [["CFR iterations", "200"], ["Value network", "15.4M parameters"], ["Trained on", "931k solved positions"], ["Held-out error", "0.0444"]]},
  {name: "Exact window", from: 9, to: 14, text: "The whole rest of the hidden-information game is small enough to solve outright: every hand either player could hold, every card they could draw.",
    kv: []},
  {name: "Minimax", from: 15, to: 23, text: "Both decks are empty or known by elimination. Nothing is hidden, so the game is searched to the end.",
    kv: [["Thinking time", "microseconds"], ["Exploitability", "0"]]},
];
const kindOf = p => KINDS.findIndex(k => p >= k.from && p <= k.to);
$("legend").innerHTML = KINDS.map((k, i) => `<span><i class="k${i}"></i>${k.name}, ${k.from === k.to ? "ply " + k.from : k.from === 15 ? "ply 15 on" : `plies ${k.from} to ${k.to}`}</span>`).join("");
$("rail").innerHTML = Array.from({length: 24}, (_, p) => `<div class="p" data-p="${p}"><div class="hands" style="height:${Math.max(2, Math.sqrt(hands(p) / 495) * 100)}px"><b>${p % 2 === 0 && hands(p) > 1 ? hands(p) : ""}</b></div><div class="band k${kindOf(p)}"></div><div class="no">${p}</div></div>`).join("");
function how(p) {
  const k = KINDS[kindOf(p)], i = kindOf(p), d = delta(p), e = DATA.exploitability;
  $("rail").querySelectorAll(".p").forEach(x => x.classList.toggle("at", +x.dataset.p === p));
  $("howtext").innerHTML = `<h3><i class="k${i}"></i>${k.name}</h3><p>${k.text}</p>`;
  const kv = [["Ply", p], ["Hands the opponent could hold", hands(p)], ...k.kv];
  if (i === 2) {
    const secs = {9: "1 to 5 s", 10: "0.4 s"}[p], best = e.find(x => x.player === "c5-2" && x.T === 8);
    kv.push(["Solve time", secs || "under 0.4 s"]);
    if (d) kv.push(["Blueprint alone loses here", d.mean.toFixed(3)]);
    kv.push(["This player, from ply 8", best.mean.toFixed(4)]);
  }
  if (i === 1) kv.push(["Blueprint alone loses here", delta(8).mean.toFixed(3)]);
  $("howkv").innerHTML = kv.map(([a, b]) => `<div><dt>${a}</dt><dd>${typeof b === "number" ? b.toLocaleString("en-US") : b}</dd></div>`).join("");
}
$("rail").onpointerover = e => { const x = e.target.closest(".p"); if (x) how(+x.dataset.p); };
$("rail").onclick = e => { const x = e.target.closest(".p"); if (x) how(+x.dataset.p); };
how(8);

// ---- findings ------------------------------------------------------------------------------------------------
const hb = (label, v, max, text, hi = false) => `<div class="hb ${hi ? "hi" : ""}"><span>${label}</span><span class="t"><i style="width:${v / max * 100}%"></i></span><span class="n">${text}</span></div>`;
const s = (a, b) => score(a, b).score;
function lineChart(points, {w = 460, h = 150, fmtX, fmtY}) {
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys) * 0.9, y1 = Math.max(...ys) * 1.05;
  const X = v => 34 + (v - x0) / (x1 - x0) * (w - 64), Y = v => h - 22 - (v - y0) / (y1 - y0) * (h - 44);
  return `<svg viewBox="0 0 ${w} ${h}"><path d="M${points.map(p => `${X(p[0])},${Y(p[1])}`).join("L")}" fill="none" stroke="var(--faint)" stroke-width="1.5"/>` +
    points.map((p, i) => `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="${i === points.length - 1 ? 5 : 3.5}" fill="${i === points.length - 1 ? "var(--a)" : "var(--soft)"}"/>` +
      `<text x="${X(p[0])}" y="${Y(p[1]) - 10}" text-anchor="middle" ${i === points.length - 1 ? 'style="fill:var(--text)"' : ""}>${fmtY(p[1])}</text>` +
      `<text x="${X(p[0])}" y="${h - 4}" text-anchor="middle">${fmtX(p[0])}</text>`).join("") + `</svg>`;
}
const ex = DATA.matches.find(m => m.a === "ex-bp" && m.argmax), nd = DATA.noted;
$("finds").innerHTML = [
  `<h3>Reading the opponent's hand is where the strength is. Seeing it wins 66.8% against the same search without it. Seeing the deck order as well adds nothing.</h3>` +
    hb("Search", s("pimc6", "pimc-hand6"), 1, pct(s("pimc6", "pimc-hand6"))) + hb("Search that sees your hand", s("pimc-hand6", "pimc6"), 1, pct(s("pimc-hand6", "pimc6")), true) +
    hb("Sees everything, vs sees hand", s("cheat6", "pimc-hand6"), 1, pct(s("cheat6", "pimc-hand6"))),
  `<h3>Search gets stronger by looking deeper, not by sampling more hidden worlds. Score against a 4-ply search:</h3>` +
    [["4 plies, 32 worlds", "pimc4w"], ["6 plies, 32 worlds", "pimc6"], ["6 plies, 128 worlds", "pimc6w"], ["8 plies, 32 worlds", "pimc8n"], ["8 plies, 128 worlds", "pimc8"]]
      .map(([l, id]) => hb(l, s(id, "pimc4"), 1, pct(s(id, "pimc4")), id === "pimc8")).join(""),
  `<h3>A learned exploiter badly understates the worst case. Against the same early blueprint, a PPO exploiter gains 0.022. An exact best response from ply 9 gains ${nd.bp8k_delta.mean}.</h3>` +
    hb("PPO exploiter, whole game", ex.score - 0.5, 0.14, (ex.score - 0.5).toFixed(3)) + hb("Exact, from ply 9", nd.bp8k_delta.mean, 0.14, nd.bp8k_delta.mean.toFixed(3), true),
  `<h3>Better value labels kept paying. Held-out error of the value network against the number of solved positions it learned from:</h3>` +
    lineChart(nd.value_error.filter(v => v.net !== "v3h").map(v => [Math.log10(v.labels), v.error]), {fmtX: v => Math.round(10 ** v / 1000) + "k", fmtY: v => v.toFixed(3)}),
].map(h => `<div class="find">${h}</div>`).join("");
