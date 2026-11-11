// The opponent thinks here so the page stays smooth. Each request replays the game from its seed and moves.
import {load} from "./engine.js";
const engine = load("bb.wasm");
onmessage = async ({data}) => {
  const e = await engine;
  e.newGame(data.seed, data.start);
  for (const a of data.moves) e.apply(a);
  const t = performance.now();
  postMessage({id: data.id, ...e.think(data.spec, data.thinkSeed), ms: Math.round(performance.now() - t)});
};
