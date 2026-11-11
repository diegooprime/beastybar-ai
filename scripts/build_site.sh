#!/bin/sh
# Rebuilds what the site is generated from: the engine as WebAssembly (site/bb.wasm) and the results
# (site/data.js, from runs/*.jsonl). With --games, also plays new replay games (site/games.js).
# Needs Rust with the wasm32-unknown-unknown target (rustup target add wasm32-unknown-unknown), Python 3, Node 20+.
set -e
cd "$(dirname "$0")/.."
(cd site/engine && cargo build --release --target wasm32-unknown-unknown)
cp site/engine/target/wasm32-unknown-unknown/release/bbweb.wasm site/bb.wasm
python3 scripts/site_data.py
if [ "$1" = "--games" ]; then node scripts/site_games.mjs; fi
echo "site built: python3 -m http.server -d site 8000"
