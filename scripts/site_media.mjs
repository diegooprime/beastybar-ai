#!/usr/bin/env node
// Renders the images and the replay clip from the site itself, so they always match it:
//   site/media/card-*.png   1600 x 900 share cards (site/cards.html)
//   docs/media/*.png        README figures (sections of site/index.html)
//   docs/media/replay.mp4   one full game, 1920 x 1080, 60 fps (site/stage.html), plus replay.gif for the README
// Needs: npm i playwright (and its chromium), ffmpeg. Run: node scripts/site_media.mjs [cards|figures|clip]...
import {chromium} from "playwright";
import http from "node:http";
import {execFileSync} from "node:child_process";
import {readFileSync, existsSync, mkdirSync, mkdtempSync} from "node:fs";
import {tmpdir} from "node:os";
import {dirname, extname, join} from "node:path";
import {fileURLToPath} from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const want = new Set(process.argv.slice(2).length ? process.argv.slice(2) : ["cards", "figures", "clip"]);
const types = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".wasm": "application/wasm", ".png": "image/png"};
const server = http.createServer((q, r) => {
  const p = join(root, "site", decodeURIComponent(q.url.split("?")[0]));
  if (!existsSync(p)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, {"content-type": types[extname(p)] || "application/octet-stream"});
  r.end(readFileSync(p));
}).listen(0);
const base = `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch();
for (const d of ["site/media", "docs/media"]) mkdirSync(join(root, d), {recursive: true});

if (want.has("cards")) {
  const page = await browser.newPage({viewport: {width: 1600, height: 900}, deviceScaleFactor: 2});
  for (const c of ["exploitability", "ladder", "method", "hidden", "animals", "table"]) {
    await page.goto(`${base}cards.html?c=${c}`, {waitUntil: "networkidle"});
    await page.waitForSelector("html[data-ready]");
    await page.waitForTimeout(400);
    await page.locator(".cardimg").screenshot({path: join(root, `site/media/card-${c}.png`)});
    console.log("card", c);
  }
  await page.close();
}

if (want.has("figures")) {
  const page = await browser.newPage({viewport: {width: 1280, height: 900}, deviceScaleFactor: 2, reducedMotion: "reduce"});
  await page.goto(`${base}index.html?game=0&ply=13`, {waitUntil: "networkidle"});
  await page.waitForTimeout(700);
  await page.addStyleTag({content: "section{border-top:0!important;padding:28px 0!important} .prov{display:none}"});
  for (const [id, name] of [["viewer", "table"], ["strength", "ladder"], ["exploit", "exploitability"], ["how", "method"], ["animals", "animals"], ["found", "findings"]]) {
    await page.locator("#" + id).screenshot({path: join(root, `docs/media/${name}.png`)});
    console.log("figure", name);
  }
  await page.close();
}

if (want.has("clip")) {
  // Time is stepped by hand (Playwright's clock), one frame per 1/60 s, so the clip is smooth whatever the machine.
  const fps = 60, dir = mkdtempSync(join(tmpdir(), "bbclip-"));
  const page = await browser.newPage({viewport: {width: 1920, height: 1080}, deviceScaleFactor: 1});
  await page.clock.install();
  await page.goto(`${base}stage.html?game=0`, {waitUntil: "networkidle"});
  await page.waitForSelector("html[data-ready]");
  let n = 0;
  for (; n < fps * 90; n++) {
    await page.clock.runFor(1000 / fps);
    await page.screenshot({path: join(dir, `f${String(n).padStart(5, "0")}.png`)});
    if (await page.evaluate(() => document.documentElement.dataset.done === "1")) break;
  }
  await page.close();
  const out = join(root, "docs/media/replay.mp4");
  execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-framerate", String(fps), "-i", join(dir, "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", "-movflags", "+faststart", out]);
  execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", out, "-vf", "fps=12,scale=880:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48[p];[b][p]paletteuse=dither=none", join(root, "docs/media/replay.gif")]);
  console.log("clip", n, "frames");
}

await browser.close();
server.close();
