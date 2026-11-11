// Play against the search player. You are seat 1 (mint, bottom); the AI is seat 0 and sees only its own view.
import {load, decode, encode, ANIMALS, animalOf, ownerOf} from "./engine.js";
import {Stage, describe, chip, glyph} from "./board.js";

const $ = id => document.getElementById(id);
const ME = 1, AI = 0;
const OPPONENTS = [
  ["Search, 8 plies", {kind: "pimc", worlds: 128, depth: 8}],
  ["Search, 4 plies", {kind: "pimc", worlds: 8, depth: 4}],
  ["Greedy", {kind: "greedy"}],
];
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem("bb." + k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("bb." + k, JSON.stringify(v)); } catch {} },
};

const engine = await load("bb.wasm");
const worker = new Worker("worker.js", {type: "module"});
const stage = new Stage($("stage"), {reveal: [false, true], bottom: ME, names: ["", "You"], onCard});
$("stage").classList.add("live");

let game = null;      // {seed, start, moves, state, sel, pick, busy, log}
let opp = store.get("opp", 0), jobs = 0;

function think() {
  const id = ++jobs, g = game;
  return new Promise(resolve => {
    const on = ({data}) => {
      if (data.id !== id) return;
      worker.removeEventListener("message", on);
      resolve(g === game ? data : null);
    };
    worker.addEventListener("message", on);
    worker.postMessage({id, seed: g.seed, start: g.start, moves: g.moves, spec: OPPONENTS[opp][1], thinkSeed: Math.floor(Math.random() * 2 ** 40)});
  });
}

function newGame() {
  const n = store.get("games", 0);
  store.set("games", n + 1);
  const seed = Math.floor(Math.random() * 2 ** 40), start = n % 2 ? AI : ME;   // the starting player alternates
  game = {seed, start, moves: [], state: engine.newGame(seed, start), sel: null, pick: [], busy: false, log: []};
  stage.names = [OPPONENTS[opp][0], "You"];
  stage.order = stage.order.map(() => Math.random());
  stage.set(game.state, {instant: true});
  $("saw").innerHTML = ""; $("sawh").textContent = "What it saw";
  paint();
  if (start === AI) aiTurn();
}

// The legal actions that play `animal`, as parameter lists.
const actionsFor = animal => engine.legal().filter(a => decode(a).animal === animal);

function onCard(c) {
  const g = game, s = g.state;
  if (g.busy || s.over || s.toMove !== ME) return;
  const inHand = ownerOf(c) === ME && s.hands[ME].includes(animalOf(c)), qi = s.queue.indexOf(c);
  if (inHand) {
    const a = animalOf(c);
    if (g.sel === a && actionsFor(a).length === 1) return commit(actionsFor(a)[0]);
    g.sel = a; g.pick = [];
  } else if (qi >= 0 && g.sel !== null && targets().includes(qi)) {
    g.pick.push(qi);
    const done = actionsFor(g.sel).filter(x => decode(x).params.join() === g.pick.join());
    if (done.length === 1 && !needsMore()) return commit(done[0]);
  } else return;
  paint();
}

// Queue positions that can be clicked next for the selected card.
function targets() {
  const g = game;
  if (g.sel === null) return [];
  const acts = actionsFor(g.sel).map(a => decode(a).params), k = g.pick.length;
  if (g.sel === 2) return [];                                  // kangaroo hops are buttons
  if (g.sel === 4 && k === 1 && animalOf(g.state.queue[g.pick[0]]) === 2) return [];
  return [...new Set(acts.filter(p => p.length > k && p.slice(0, k).join() === g.pick.join()).map(p => p[k]))];
}
const needsMore = () => actionsFor(game.sel).some(a => { const p = decode(a).params; return p.length > game.pick.length && p.slice(0, game.pick.length).join() === game.pick.join(); });

function paint() {
  const g = game, s = g.state, mine = !s.over && s.toMove === ME && !g.busy;
  $("stage").classList.toggle("wait", !mine);
  const tg = mine ? targets() : [];
  stage.cards.forEach((k, c) => {
    k.el.classList.toggle("sel", mine && g.sel !== null && c === ME * 12 + g.sel);
    k.el.classList.toggle("tgt", tg.includes(s.queue.indexOf(c)));
  });
  // Action line.
  let html = "";
  if (s.over) {
    const [ai, me] = s.scores;
    html = `<span class="result">${me > ai ? "You win" : me < ai ? "It wins" : "Draw"} <b>${me} : ${ai}</b></span><button class="btn go" id="again">New game</button>`;
  } else if (g.busy && s.toMove === AI) html = `<span class="think"></span><span>${OPPONENTS[opp][0]} is thinking</span>`;
  else if (mine && g.sel === null) html = `<span class="fx">Your move: pick a card</span>`;
  else if (mine) {
    const acts = actionsFor(g.sel), name = ANIMALS[g.sel];
    const hop = g.sel === 2 || (g.sel === 4 && g.pick.length === 1 && animalOf(s.queue[g.pick[0]]) === 2);
    if (acts.length === 1) html = `<button class="btn go" data-a="${acts[0]}">Play ${name}</button>` + effect(acts[0]);
    else if (hop) html = acts.filter(a => decode(a).params.slice(0, g.pick.length).join() === g.pick.join() && decode(a).params.length > g.pick.length)
      .map(a => `<button class="btn" data-a="${a}">Hop ${decode(a).params.at(-1)}</button>`).join("") || `<button class="btn go" data-a="${acts[0]}">Play ${name}</button>`;
    else html = `<span class="fx">${g.sel === 1 || g.pick.length ? "Pick the animal to bounce" : "Pick the animal to copy"}</span>`;
  }
  $("act").innerHTML = html;
  // Side panel.
  const seen = new Set([...s.queue, ...s.bar, ...s.out].filter(c => ownerOf(c) === AI).map(animalOf));
  $("could").innerHTML = Array.from({length: 12}, (_, i) => 11 - i).filter(a => !seen.has(a)).map(a => chip(AI * 12 + a)).join("") || `<span class="note">nothing left</span>`;
  $("mine").innerHTML = [...s.decks[ME]].sort((a, b) => b - a).map(a => chip(ME * 12 + a)).join("") || `<span class="note">empty</span>`;
  $("opp").innerHTML = OPPONENTS.map(([n], i) => `<button data-o="${i}" class="${i === opp ? "on" : ""}">${n}</button>`).join("");
  $("log").innerHTML = g.log.map((d, i) => {
    const fx = [d.how, d.entered.length ? "in " + d.entered.map(c => chip(c)).join(" ") : "", d.bounced.length || d.last !== null
      ? "out " + [...d.bounced, ...(d.last !== null ? [d.last] : [])].map(c => chip(c, "dim")).join(" ") : ""].filter(Boolean).join(" · ");
    return `<button tabindex="-1"><span class="k">${i + 1}</span>${chip(d.card)}<span class="fx">${fx}</span></button>`;
  }).reverse().join("");
}

// What a move would do, before it is played.
function effect(action) {
  const after = engine.preview(action), d = describe(game.state, action, after);
  const parts = [d.entered.length ? "in " + d.entered.map(c => chip(c)).join(" ") : "",
    d.bounced.length || d.last !== null ? "out " + [...d.bounced, ...(d.last !== null ? [d.last] : [])].map(c => chip(c)).join(" ") : ""].filter(Boolean);
  return parts.length ? `<span class="fx">${parts.join(" · ")}</span>` : "";
}

async function commit(action) {
  const g = game, before = g.state;
  g.busy = true; g.sel = null; g.pick = [];
  const after = g.state = engine.apply(action);
  g.moves.push(action);
  paint();
  g.log.push(await stage.move(before, action, after));
  if (g !== game) return;
  g.busy = false;
  paint();
  if (!after.over && after.toMove === AI) aiTurn();
}

async function aiTurn() {
  const g = game, before = g.state;
  g.busy = true;
  paint();
  const [r] = await Promise.all([think(), new Promise(res => setTimeout(res, 450))]);
  if (!r || g !== game) return;
  const after = g.state = engine.apply(r.action);
  g.moves.push(r.action);
  // Its options, with the hidden card names: it has played, so this reveals only the card it chose.
  const rows = Object.entries(r.values).map(([a, v]) => [+a, v]).sort((x, y) => y[1] - x[1]);
  const lo = Math.min(...rows.map(x => x[1])), hi = Math.max(...rows.map(x => x[1]));
  $("sawh").textContent = `What it saw: ${rows.length} option${rows.length === 1 ? "" : "s"}, ${r.ms} ms`;
  $("saw").innerHTML = rows.slice(0, 5).map(([a, v]) => {
    const w = hi === lo ? 100 : 8 + 92 * (v - lo) / (hi - lo), pick = a === r.action;
    return `<div class="opt s0 ${pick ? "pick" : ""}">${pick ? chip(AI * 12 + decode(a).animal) : `<span class="chip ghost">?</span>`}<span class="lbl">${pick ? ANIMALS[decode(a).animal] : "not played"}</span>` +
      `<span class="v"><i style="width:${w}%"></i></span><span class="n">${(v > 0 ? "+" : "") + v.toFixed(1)}</span></div>`;
  }).join("");
  g.log.push(await stage.move(before, r.action, after));
  if (g !== game) return;
  g.busy = false;
  paint();
  if (!after.over && after.toMove === AI) aiTurn();
}

$("act").onclick = e => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.id === "again") newGame();
  else if (b.dataset.a) commit(+b.dataset.a);
};
$("new").onclick = newGame;
$("opp").onclick = e => { const b = e.target.closest("button"); if (b) { opp = +b.dataset.o; store.set("opp", opp); newGame(); } };
addEventListener("keydown", e => {
  if (e.key === "Escape" && game.sel !== null) { game.sel = null; game.pick = []; paint(); }
  if (e.key === "Enter") { const b = $("act").querySelector("button.go"); if (b) b.click(); }
});
newGame();
