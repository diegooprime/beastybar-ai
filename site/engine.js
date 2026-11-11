// The Beasty Bar engine in the browser: site/bb.wasm (the Rust engine in core/, built by scripts/build_site.sh).
// One game per Engine. Cards are ids: owner * 12 + animal; animal id = strength - 1.

export const ANIMALS = ["skunk", "parrot", "kangaroo", "monkey", "chameleon", "seal", "zebra", "giraffe", "snake",
  "crocodile", "hippo", "lion"];
export const POINTS = [4, 4, 4, 3, 3, 2, 4, 3, 2, 3, 2, 2];
export const animalOf = c => c % 12;
export const ownerOf = c => (c >= 12 ? 1 : 0);
export const strengthOf = c => (c % 12) + 1;
export const pointsOf = c => POINTS[c % 12];

// Action layout (core/src/action.rs): parameters are queue positions before the move, 0 = front.
export function encode(animal, params = []) {
  const [t, x] = params;
  if (animal === 0) return 0;
  if (animal === 1) return 1 + t;
  if (animal === 2) return params.length ? 5 + t : 5;
  if (animal === 3) return 8;
  if (animal === 4) return params.length === 2 ? 13 + 4 * t + x : 9 + t;
  return animal + 24;
}

export function decode(a) {
  if (a === 0) return {animal: 0, params: []};
  if (a <= 4) return {animal: 1, params: [a - 1]};
  if (a === 5) return {animal: 2, params: []};
  if (a <= 7) return {animal: 2, params: [a - 5]};
  if (a === 8) return {animal: 3, params: []};
  if (a <= 12) return {animal: 4, params: [a - 9]};
  if (a <= 28) return {animal: 4, params: [(a - 13) >> 2, (a - 13) & 3]};
  return {animal: a - 24, params: []};
}

const bits = m => { const out = []; for (let i = 0; i < 12; i++) if (m >> i & 1) out.push(i); return out; };

function read(b) {
  const u16 = i => b[i] | b[i + 1] << 8;
  return {
    toMove: b[0], start: b[1], over: !!b[2], scores: [b[3], b[4]],
    queue: Array.from(b.subarray(6, 6 + b[5])),
    hands: [bits(u16(11)), bits(u16(13))],
    deckLen: [b[15], b[16]],
    bar: Array.from(b.subarray(18, 18 + b[17])),
    out: Array.from(b.subarray(35, 35 + b[34])),
    decks: [Array.from(b.subarray(59, 59 + b[15])), Array.from(b.subarray(67, 67 + b[16]))],
  };
}

const split = seed => [Math.floor(seed / 4294967296) >>> 0, seed >>> 0];

export class Engine {
  constructor(instance) { this.x = instance.exports; }
  get bytes() { return new Uint8Array(this.x.memory.buffer, this.x.bb_out(), 75); }
  newGame(seed, start = 0) { this.x.bb_new(...split(seed), start); return this.state(); }
  // A constructed position: `queue` card ids front first (at most 4), `hands` two lists of animal ids.
  setup(queue, hands, toMove = 0) {
    const b = new Uint8Array(this.x.memory.buffer, this.x.bb_in(), 16), mask = h => h.reduce((m, a) => m | 1 << a, 0);
    b.fill(0); b[0] = toMove; b[1] = queue.length; b.set(queue, 2);
    b[6] = mask(hands[0]) & 255; b[7] = mask(hands[0]) >> 8; b[8] = mask(hands[1]) & 255; b[9] = mask(hands[1]) >> 8;
    this.x.bb_setup(); return this.state();
  }
  state() { this.x.bb_state(); return read(this.bytes); }
  legal() {
    const lo = this.x.bb_legal_lo() >>> 0, hi = this.x.bb_legal_hi() >>> 0, out = [];
    for (let a = 0; a < 36; a++) if ((a < 32 ? lo >>> a : hi >>> (a - 32)) & 1) out.push(a);
    return out;
  }
  preview(action) { return this.x.bb_preview(action) ? read(this.bytes) : null; }
  apply(action) { if (!this.x.bb_apply(action)) throw new Error("illegal action " + action); return this.state(); }
  undo() { return !!this.x.bb_undo(); }
  // kind: "random" | "greedy" | "pimc". Returns the move and, for pimc, the value it gave every legal action.
  think({kind = "pimc", worlds = 128, depth = 8} = {}, seed = 1) {
    const k = kind === "random" ? 0 : kind === "greedy" ? 1 : 2;
    const action = this.x.bb_think(k, worlds, depth, ...split(seed));
    const v = new Float32Array(this.x.memory.buffer, this.x.bb_values(), 36), values = {};
    for (let a = 0; a < 36; a++) if (!Number.isNaN(v[a])) values[a] = v[a];
    return {action, values};
  }
}

export async function load(source) {
  const bytes = source instanceof ArrayBuffer || ArrayBuffer.isView(source) ? source
    : await (await fetch(source)).arrayBuffer();
  const {instance} = await WebAssembly.instantiate(bytes, {});
  return new Engine(instance);
}
