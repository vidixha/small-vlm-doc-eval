#!/usr/bin/env python3
"""Build the QA annotation page from images in custom_docs/.

Scans /content/drive/MyDrive/vlm_eval/custom_docs/ for images, downscales to
max 1100px wide (JPEG q72) so 25 docs stay within artifact size budget, embeds
them as data URIs in a self-contained HTML annotator (3 QA pairs per doc,
localStorage autosave, JSON export). Output: vlm_eval/annotator/annotator.html
— published to a hosted page via the Artifact tool afterwards.
"""
import base64
import io
import json
from pathlib import Path

from PIL import Image

DOCS = Path("/content/drive/MyDrive/vlm_eval/custom_docs")
OUT = Path("/content/drive/MyDrive/vlm_eval/annotator/annotator.html")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
MAX_W = 1100

docs = []
files = sorted(p for p in DOCS.iterdir() if p.suffix.lower() in EXTS) if DOCS.exists() else []
def load_rgb(p):
    img = Image.open(p)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # composite transparency onto white — plain convert("RGB") uses black,
        # which renders black-text-on-transparent docs as a black rectangle
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")

import numpy as np

def is_blank(p):
    arr = np.asarray(Image.open(p).convert("RGBA"))
    vis = arr[..., 3] > 10
    return not vis.any() or arr[..., :3][vis].std() < 5

skipped = [p.name for p in files if is_blank(p)]
files = [p for p in files if p.name not in skipped]
if skipped:
    print(f"SKIPPED {len(skipped)} blank images: {', '.join(skipped)}")

for p in files:
    img = load_rgb(p)
    if img.width > MAX_W:
        img = img.resize((MAX_W, int(img.height * MAX_W / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=72, optimize=True)
    docs.append({"file": p.name, "w": img.width, "h": img.height,
                 "src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})
    print(f"  {p.name}: {img.width}x{img.height}, {buf.tell()//1024}KB")

print(f"{len(docs)} documents embedded")
DATA = json.dumps(docs)

HTML = """<title>Document QA Annotator</title>
<style>
:root{
  --ground:#FAF9F6; --surface:#FFFFFF; --ink:#201F1B; --ink-2:#6B6A63;
  --line:#E3E1D8; --accent:#0E6E6E; --accent-ink:#FFFFFF; --accent-soft:#E3EEED;
  --chip-empty:#EFEEE8; --focus:#0E6E6E;
}
@media (prefers-color-scheme: dark){:root{
  --ground:#141817; --surface:#1C2120; --ink:#E8EAE7; --ink-2:#9AA39F;
  --line:#2C3331; --accent:#4FB3AE; --accent-ink:#10201F; --accent-soft:#1E3230;
  --chip-empty:#232927; --focus:#4FB3AE;
}}
:root[data-theme="dark"]{
  --ground:#141817; --surface:#1C2120; --ink:#E8EAE7; --ink-2:#9AA39F;
  --line:#2C3331; --accent:#4FB3AE; --accent-ink:#10201F; --accent-soft:#1E3230;
  --chip-empty:#232927; --focus:#4FB3AE;
}
:root[data-theme="light"]{
  --ground:#FAF9F6; --surface:#FFFFFF; --ink:#201F1B; --ink-2:#6B6A63;
  --line:#E3E1D8; --accent:#0E6E6E; --accent-ink:#FFFFFF; --accent-soft:#E3EEED;
  --chip-empty:#EFEEE8; --focus:#0E6E6E;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
.mono{font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:16px;
  padding:10px 20px;background:var(--surface);border-bottom:1px solid var(--line)}
header h1{font-size:15px;margin:0;font-weight:650;letter-spacing:.01em}
.meter{flex:1;display:flex;align-items:center;gap:10px;min-width:0}
.bar{flex:1;height:6px;border-radius:3px;background:var(--chip-empty);overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent);width:0%;transition:width .2s}
.count{color:var(--ink-2);font-size:12px;white-space:nowrap}
button{font:inherit;cursor:pointer;border-radius:6px;border:1px solid var(--line);
  background:var(--surface);color:var(--ink);padding:7px 14px}
button:focus-visible,textarea:focus-visible,input:focus-visible{
  outline:2px solid var(--focus);outline-offset:1px}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:600}
main{display:grid;grid-template-columns:1fr 400px;height:calc(100vh - 53px)}
.viewer{overflow:auto;padding:24px;display:flex;justify-content:center;align-items:flex-start}
.viewer img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;
  box-shadow:0 2px 10px rgba(0,0,0,.07);cursor:zoom-in}
.viewer img.zoom{max-width:none;cursor:zoom-out}
aside{border-left:1px solid var(--line);background:var(--surface);overflow-y:auto;
  padding:18px 18px 28px;display:flex;flex-direction:column;gap:14px}
.eyebrow{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-2);font-weight:650}
.docnav{display:flex;align-items:center;gap:8px}
.docnav .file{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}
.chip{font-size:10.5px;padding:2px 8px;border-radius:99px;background:var(--chip-empty);
  color:var(--ink-2);white-space:nowrap}
.chip.done{background:var(--accent-soft);color:var(--accent);font-weight:650}
.qa{display:flex;flex-direction:column;gap:6px;padding:12px;border:1px solid var(--line);border-radius:8px}
.qa label{font-size:11px;color:var(--ink-2)}
textarea,input[type=text]{width:100%;font:inherit;color:var(--ink);background:var(--ground);
  border:1px solid var(--line);border-radius:6px;padding:7px 9px;resize:vertical}
textarea{min-height:44px}
.hint{font-size:12px;color:var(--ink-2)}
.strip{display:flex;gap:6px;flex-wrap:wrap}
.strip button{width:30px;height:30px;padding:0;font-size:11.5px}
.strip button[aria-current="true"]{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.strip button.has{border-color:var(--accent)}
.empty{margin:auto;text-align:center;color:var(--ink-2)}
@media (max-width:860px){main{grid-template-columns:1fr;height:auto}
  aside{border-left:none;border-top:1px solid var(--line)}}
</style>
<header>
  <h1>Document QA Annotator</h1>
  <div class="meter"><div class="bar"><i id="fill"></i></div>
  <span class="count mono" id="count"></span></div>
  <button id="copyBtn" title="Copy annotations JSON to clipboard">Copy JSON</button>
  <button class="primary" id="exportBtn">Export annotations.json</button>
</header>
<main>
  <div class="viewer"><img id="img" alt="document page" /></div>
  <aside>
    <div>
      <div class="eyebrow">Document</div>
      <div class="docnav">
        <button id="prev" title="Previous (←)">←</button>
        <span class="file mono" id="fname"></span>
        <span class="chip" id="chip"></span>
        <button id="next" title="Next (→)">→</button>
      </div>
    </div>
    <div class="strip" id="strip"></div>
    <div>
      <div class="eyebrow">Failure mode under test</div>
      <input type="text" id="fmode" list="fmodes"
        placeholder="e.g. blur, dense-table, handwriting, low-contrast…" />
      <datalist id="fmodes">
        <option value="blur"></option><option value="rotation"></option>
        <option value="jpeg-compression"></option><option value="dense-table"></option>
        <option value="handwriting"></option><option value="low-contrast"></option>
        <option value="small-font"></option><option value="skewed-scan"></option>
        <option value="multi-column"></option><option value="clean"></option>
      </datalist>
    </div>
    <div class="eyebrow">Question–answer pairs <span style="text-transform:none;font-weight:400">(up to 3 — leave unused blank)</span></div>
    <div id="qas"></div>
    <p class="hint">Answers should be short and extractive — a value, date, name, or phrase visible in the document, like DocVQA ground truth. Everything autosaves in this browser; “Export” downloads the JSON file to drop back into <span class="mono">vlm_eval/custom_docs/</span>.</p>
  </aside>
</main>
<script>
const DOCS = __DATA__;
const KEY = "docqa-annotations-v1";
let store = JSON.parse(localStorage.getItem(KEY) || "{}");
let cur = 0;
const $ = id => document.getElementById(id);

function rec(f){ return store[f] || {qa:[{q:"",a:""},{q:"",a:""},{q:"",a:""}], fmode:""}; }
function pairs(f){ return rec(f).qa; }
function nDone(f){ return pairs(f).filter(p=>p.q.trim()&&p.a.trim()).length; }
function save(){ localStorage.setItem(KEY, JSON.stringify(store)); paint(); }

function paint(){
  const done = DOCS.filter(d=>nDone(d.file)>0).length;
  const qa = DOCS.reduce((s,d)=>s+nDone(d.file),0);
  $("fill").style.width = DOCS.length? (100*done/DOCS.length)+"%" : "0%";
  $("count").textContent = done+"/"+DOCS.length+" docs · "+qa+" QA pairs";
  const n = nDone(DOCS[cur].file);
  const chip = $("chip");
  chip.textContent = n? n+" pair"+(n>1?"s":"") : "empty";
  chip.className = "chip"+(n?" done":"");
  [...$("strip").children].forEach((b,i)=>{
    b.setAttribute("aria-current", i===cur ? "true":"false");
    b.className = nDone(DOCS[i].file)? "has":"";
  });
}

function show(i){
  cur = (i+DOCS.length)%DOCS.length;
  const d = DOCS[cur];
  $("img").src = d.src; $("img").classList.remove("zoom");
  $("fname").textContent = d.file;
  const fm = $("fmode");
  fm.value = rec(d.file).fmode || "";
  fm.oninput = () => {
    const r = rec(d.file); r.fmode = fm.value; store[d.file] = r; save();
  };
  const qas = $("qas"); qas.innerHTML = "";
  pairs(d.file).forEach((p,k)=>{
    const div = document.createElement("div");
    div.className = "qa"; div.style.marginBottom = "10px";
    div.innerHTML = '<label>Question '+(k+1)+'</label>'+
      '<textarea data-k="'+k+'" data-t="q" placeholder="e.g. What is the invoice total?"></textarea>'+
      '<label>Answer '+(k+1)+'</label>'+
      '<input type="text" data-k="'+k+'" data-t="a" placeholder="e.g. $1,240.00" />';
    qas.appendChild(div);
    div.querySelector('[data-t=q]').value = p.q;
    div.querySelector('[data-t=a]').value = p.a;
  });
  qas.querySelectorAll("textarea,input").forEach(el=>{
    el.addEventListener("input", () => {
      const r = rec(d.file);
      r.qa[+el.dataset.k][el.dataset.t] = el.value;
      store[d.file] = r; save();
    });
  });
  paint();
}

function exportJson(download){
  const out = {version:2, exported_at:new Date().toISOString(),
    docs: DOCS.map(d=>({file:d.file,
      failure_mode: rec(d.file).fmode.trim(),
      qa: pairs(d.file).filter(p=>p.q.trim()&&p.a.trim())
          .map(p=>({question:p.q.trim(), answer:p.a.trim()}))}))
      .filter(d=>d.qa.length)};
  const text = JSON.stringify(out, null, 1);
  if(download){
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text],{type:"application/json"}));
    a.download = "annotations.json"; a.click();
  } else {
    navigator.clipboard.writeText(text);
    $("copyBtn").textContent = "Copied ✓";
    setTimeout(()=>$("copyBtn").textContent="Copy JSON", 1500);
  }
}

if(!DOCS.length){
  document.querySelector("main").innerHTML =
    '<p class="empty">No documents found. Upload images to <span class="mono">vlm_eval/custom_docs/</span> and rebuild.</p>';
} else {
  $("prev").onclick = ()=>show(cur-1);
  $("next").onclick = ()=>show(cur+1);
  $("img").onclick = e=>e.target.classList.toggle("zoom");
  $("exportBtn").onclick = ()=>exportJson(true);
  $("copyBtn").onclick = ()=>exportJson(false);
  const strip = $("strip");
  DOCS.forEach((d,i)=>{
    const b = document.createElement("button");
    b.textContent = i+1; b.title = d.file;
    b.onclick = ()=>show(i); strip.appendChild(b);
  });
  document.addEventListener("keydown", e=>{
    if(e.target.matches("textarea,input")) return;
    if(e.key==="ArrowLeft") show(cur-1);
    if(e.key==="ArrowRight") show(cur+1);
  });
  show(0);
}
</script>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML.replace("__DATA__", DATA))
print(f"annotator written -> {OUT} ({OUT.stat().st_size//1024}KB)")
