// Builds the submission PDF from docs/CAPSTONE_DOCUMENTATION.md (the single source of the report).
//   node scripts/build-submission.mjs   (needs Node 20+ and Google Chrome)  →  the PDF named in CONFIG (and a temporary HTML in reports/, not committed)
// Markdown → styled HTML (small converter for what the report uses) → headless Chrome "print to PDF".
// The same script is used in every capstone project; only CONFIG differs.
const CONFIG = {
  source: "docs/CAPSTONE_DOCUMENTATION.md",
  pdf: "docs/NeuroQuest_Capstone_Submission.pdf",
  eyebrow: "IIT Patna · AI/ML Capstone · Project Submission Report",
  title: "NEUROQUEST",
  subtitle: "The game that learns you: a maze survival game with search, supervised learning, adaptive difficulty and reinforcement learning",
  note: "Self-directed AI/ML capstone (Python and Pygame). Every result in this report is measured, seeded and reproducible.",
  hero: { file: "screenshots/ui5-playing-panel.png", caption: "Playing with the AI panel open: A* hunters, the live player-style prediction and the adaptive difficulty." },
  colours: { band: "linear-gradient(135deg, #1d5a3a, #2f8f5b 55%, #3a7bd5)", accent: "#ffe27a", rule: "#3fae6a" },
};
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const docs = path.join(root, "docs");
const src = fs.readFileSync(path.join(root, CONFIG.source), "utf8");
fs.mkdirSync(path.join(root, "reports"), { recursive: true });
const outHtml = path.join(root, "reports", "submission.html");
const outPdf = path.join(root, CONFIG.pdf);

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
// backslash escapes (A\*, \|) become placeholders so the emphasis rules leave them alone
const ESC = ["*", "_", "|", "`", "[", "]"];
const inline = (s) => esc(s.replace(/\\([*_|`[\]])/g, (_, c) => `\u0000${ESC.indexOf(c)}\u0000`))
  .replace(/`([^`]+)`/g, "<code>$1</code>")
  .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
  .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>")
  .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
  .replace(/(^|[\s(])(https:\/\/[^\s<)]+)/g, '$1<a href="$2">$2</a>')
  .replace(/\u0000(\d)\u0000/g, (_, i) => ESC[+i]);
const img = (file) => pathToFileURL(path.join(docs, file)).href;

/** Markdown blocks → HTML: headings, paragraphs, lists, tables, code fences, images with an italic caption line. */
function render(md) {
  const lines = md.split("\n"), out = [];
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (!l.trim() || /^-{3,}$/.test(l.trim())) continue;
    if (l.startsWith("```")) { const buf = []; while (++i < lines.length && !lines[i].startsWith("```")) buf.push(lines[i]); out.push(`<pre>${esc(buf.join("\n"))}</pre>`); continue; }
    const h = l.match(/^(#{2,3}) (.*)/);
    if (h) { out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }
    const im = l.match(/^!\[([^\]]*)\]\(([^)]+)\)/);
    if (im) { const cap = lines[i + 1]?.match(/^\*(.*)\*$/); if (cap) i++; out.push(`<figure><img src="${img(im[2])}" alt="${esc(im[1])}"><figcaption>${inline(cap ? cap[1] : im[1])}</figcaption></figure>`); continue; }
    if (l.startsWith("|")) {
      const rows = []; while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]); i--;
      const cells = (r) => r.replace(/\\\|/g, "\u0001").replace(/^\||\|$/g, "").split("|").map((c) => c.trim().replace(/\u0001/g, "\\|"));
      const head = cells(rows[0]), body = rows.slice(2).map(cells);
      const headless = head.every((c) => !c);
      out.push(`<table${headless ? ' class="kv"' : ""}>${headless ? "" : `<thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead>`}<tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    const li = l.match(/^(\s*)(-|\d+\.) (.*)/);
    if (li) {
      const ordered = /\d/.test(li[2]), items = [];
      while (i < lines.length && /^(-|\d+\.) /.test(lines[i])) items.push(lines[i++].replace(/^(-|\d+\.) /, "")); i--;
      out.push(`<${ordered ? "ol" : "ul"}>${items.map((x) => `<li>${inline(x)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }
    const buf = [l]; while (i + 1 < lines.length && lines[i + 1].trim() && !/^(#|\||-|\d+\.|!\[|```)/.test(lines[i + 1])) buf.push(lines[++i]);
    out.push(`<p>${inline(buf.join(" "))}</p>`);
  }
  return out.join("\n");
}

// Cover: title, the cover-page table and a hero screenshot; the rest of the report follows on new pages.
const [, afterTitle] = src.split(/^# .*$/m);
const sections = afterTitle.split(/^(?=## )/m);
sections.shift(); // the intro line under the title: the cover uses CONFIG.note instead
const coverIdx = sections.findIndex((s) => /^## 1\. Cover page/.test(s));
const coverTable = render(sections[coverIdx].replace(/^## .*$/m, ""));
const body = sections.filter((_, i) => i !== coverIdx).map(render).join("\n");

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${esc(CONFIG.title)} – Capstone Submission Report</title>
<style>
  @page { size: A4; margin: 16mm 15mm 18mm; }
  * { box-sizing: border-box; }
  body { font: 10.5pt/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1c2033; margin: 0; }
  a { color: #0b6fa4; text-decoration: none; word-break: break-all; }
  h2 { font-size: 15pt; color: #10142b; border-bottom: 2px solid ${CONFIG.colours.rule}; padding-bottom: 4px; margin: 22px 0 10px; break-after: avoid; }
  h3 { font-size: 12pt; margin: 14px 0 6px; break-after: avoid; }
  p { margin: 6px 0 8px; } ul, ol { margin: 6px 0 10px; padding-left: 28px; } li { margin: 3px 0; }
  code { font: 9pt/1.3 Menlo, Consolas, monospace; background: #eef2f7; padding: 1px 4px; border-radius: 3px; }
  pre { font: 7.8pt/1.4 Menlo, Consolas, monospace; background: #0f1430; color: #d7e3ff; padding: 10px 12px; border-radius: 6px; white-space: pre; overflow: hidden; break-inside: avoid; }
  table { width: 100%; border-collapse: collapse; margin: 8px 0 12px; font-size: 9.4pt; break-inside: auto; }
  th, td { border: 1px solid #d5dbe6; padding: 5px 7px; text-align: left; vertical-align: top; }
  th { background: #eaf6fa; } tr { break-inside: avoid; }
  figure { margin: 10px 0 14px; break-inside: avoid; text-align: center; }
  figure img { max-width: 100%; max-height: 118mm; border: 1px solid #c9d1de; border-radius: 6px; }
  figcaption { font-size: 9pt; color: #4a536b; margin-top: 4px; }
  .cover { height: 258mm; overflow: hidden; display: flex; flex-direction: column; break-after: page; }
  .cover .band { background: ${CONFIG.colours.band}; color: #fff; border-radius: 10px; padding: 26px 26px 22px; }
  .cover .eyebrow { letter-spacing: .18em; text-transform: uppercase; font-size: 9pt; color: #7fe9ff; margin: 0 0 6px; }
  .cover h1 { font-size: 34pt; margin: 0; letter-spacing: .04em; color: ${CONFIG.colours.accent}; }
  .cover .sub { font-size: 13pt; margin: 6px 0 0; color: #e6e9ff; }
  .cover .note { font-size: 9pt; color: #b8bfe6; margin: 12px 0 0; }
  .cover table.kv td:first-child { width: 34%; background: #f4f7fb; }
  .cover table.kv { font-size: 10pt; margin-top: 16px; }
  /* the hero takes whatever height the cover has left, so a long cover table never pushes it onto page 2 */
  .cover figure { flex: 1 1 0; min-height: 0; margin: 14px 0 0; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; }
  .cover figure img { flex: 0 1 auto; min-height: 0; max-height: 92mm; max-width: 100%; object-fit: contain; }
</style></head><body>
<section class="cover">
  <div class="band"><p class="eyebrow">${esc(CONFIG.eyebrow)}</p><h1>${esc(CONFIG.title)}</h1>
  <p class="sub">${esc(CONFIG.subtitle)}</p>
  <p class="note">${inline(CONFIG.note)}</p></div>
  ${coverTable}
  ${CONFIG.hero ? `<figure><img src="${img(CONFIG.hero.file)}" alt=""><figcaption>${inline(CONFIG.hero.caption)}</figcaption></figure>` : ""}
</section>
${body}
</body></html>`;

fs.writeFileSync(outHtml, html);
const chrome = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
execFileSync(chrome, ["--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files", `--print-to-pdf=${outPdf}`, pathToFileURL(outHtml).href], { stdio: "ignore" });
console.log(`submission: ${path.relative(root, outPdf)} (${(fs.statSync(outPdf).size / 1024).toFixed(0)} KB) from ${CONFIG.source}`);
