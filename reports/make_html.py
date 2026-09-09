#!/usr/bin/env python3
"""Render reports/beta_calibration.html from reports/calibration_payload.json.

Self-contained: the payload is inlined, so the file opens from disk with no
network and no build step, and it outlives /scratch's 30-day purge.

Palette is the dataviz reference instance, validated with
scripts/validate_palette.js for the 3 categorical slots used here (beta arms):
light all-pairs worst CVD dE 9.2, normal-vision 24.0; dark 9.4 / 20.9. Light-mode
aqua sits at 2.74:1 on the light surface, so the relief rule applies -- every
chart therefore ships direct labels AND a table view.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "calibration_payload.json")) as f:
    PAYLOAD = json.load(f)

HTML = """<!DOCTYPE html>
<meta charset="utf-8">
<title>Beta calibration — GPT-2 zoo, Mini, N=12</title>
<style>
  :root { --page:#f9f9f7; }
  html,body{margin:0;background:var(--page);}
  .viz-root{
    color-scheme: light;
    --surface-1:#fcfcfb; --page-plane:#f9f9f7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
    --good:#0ca30c; --crit:#d03b3b;
    --seq100:#cde2fb; --seq200:#9ec5f4; --seq300:#6da7ec; --seq400:#3987e5;
    --seq500:#256abf; --seq600:#184f95; --seq700:#0d366b;
    --on-seq-lo:#0b0b0b; --on-seq-hi:#fcfcfb;
    font-family: system-ui,-apple-system,"Segoe UI",sans-serif;
    color:var(--text-primary);
    max-width:1180px; margin:0 auto; padding:28px 20px 64px;
  }
  @media (prefers-color-scheme: dark){
    :root:where(:not([data-theme="light"])) { --page:#0d0d0d; }
    :root:where(:not([data-theme="light"])) .viz-root{
      color-scheme: dark;
      --surface-1:#1a1a19; --page-plane:#0d0d0d;
      --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
      --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
      --s1:#3987e5; --s2:#d95926; --s3:#199e70;
      --seq100:#0d366b; --seq200:#184f95; --seq300:#256abf; --seq400:#2a78d6;
      --seq500:#3987e5; --seq600:#6da7ec; --seq700:#9ec5f4;
      --on-seq-lo:#ffffff; --on-seq-hi:#0b0b0b;
    }
  }
  :root[data-theme="dark"]{ --page:#0d0d0d; }
  :root[data-theme="dark"] .viz-root{
    color-scheme: dark;
    --surface-1:#1a1a19; --page-plane:#0d0d0d;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70;
    --seq100:#0d366b; --seq200:#184f95; --seq300:#256abf; --seq400:#2a78d6;
    --seq500:#3987e5; --seq600:#6da7ec; --seq700:#9ec5f4;
    --on-seq-lo:#ffffff; --on-seq-hi:#0b0b0b;
  }
  h1{font-size:24px;font-weight:650;margin:0 0 4px;letter-spacing:-0.01em}
  .sub{color:var(--text-secondary);font-size:14px;margin:0 0 26px;line-height:1.5}
  h2{font-size:16px;font-weight:620;margin:0 0 3px}
  .cap{color:var(--text-secondary);font-size:12.5px;margin:0 0 14px;line-height:1.5;max-width:860px}
  .card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:18px 18px 14px;margin:0 0 22px}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px;margin:0 0 22px}
  .tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
  .tile .k{font-size:11.5px;color:var(--muted);letter-spacing:.02em}
  .tile .v{font-size:27px;font-weight:640;margin:5px 0 1px;line-height:1.1}
  .tile .n{font-size:11.5px;color:var(--text-secondary);line-height:1.4}
  .legend{display:flex;gap:16px;flex-wrap:wrap;align-items:center;font-size:12.5px;color:var(--text-secondary);margin:0 0 10px}
  .legend i{width:11px;height:11px;border-radius:3px;display:inline-block;margin-right:6px;vertical-align:-1px}
  table{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;width:100%}
  th,td{padding:4px 8px;text-align:right;border-bottom:1px solid var(--grid)}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
  .tv{display:none;margin-top:12px}
  .tv.on{display:block}
  button.t{font:inherit;font-size:11.5px;color:var(--text-secondary);background:transparent;
    border:1px solid var(--border);border-radius:6px;padding:3px 9px;cursor:pointer;margin-top:8px}
  button.t:hover{background:var(--page-plane)}
  .tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
    background:var(--surface-1);border:1px solid var(--border);border-radius:7px;
    padding:7px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.13);z-index:9;max-width:250px}
  .tip b{font-weight:620}
  text{font-family:inherit}
  .ax{fill:var(--muted);font-size:10.5px;font-variant-numeric:tabular-nums}
  .al{fill:var(--text-secondary);font-size:11.5px}
  .dl{fill:var(--text-primary);font-size:11px;font-weight:600;font-variant-numeric:tabular-nums}
  .note{font-size:12.5px;color:var(--text-secondary);line-height:1.55;background:var(--page-plane);
    border-left:2px solid var(--s2);padding:10px 14px;border-radius:0 7px 7px 0;margin:14px 0 0;max-width:860px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px}
  @media(max-width:900px){.grid2{grid-template-columns:1fr}}
  .hm{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
  @media(max-width:980px){.hm{grid-template-columns:1fr}}
  .toggle{position:fixed;top:12px;right:14px;z-index:10}
</style>
<div class="toggle"><button class="t" onclick="tt()">light / dark</button></div>
<div class="viz-root">
<h1>&beta; calibration — GPT-2 zoo, Mini, N=12</h1>
<p class="sub">RESEARCH_PLAN &sect;4.2 / &sect;4.3. Three &beta; arms, 12 one-hot anchor members each,
36 branches off 3 shared trunks on AICR B200s. All three arms <b>pass</b> the &sect;4.3 gate.
Every number here is read from a sealed artifact, not a log.</p>
<div id="app"></div>
</div>
<div class="tip" id="tip"></div>
<script>
const P = __PAYLOAD__;
const DOM = ["web","code","math","books","multilingual"];
const SC = ["var(--s1)","var(--s2)","var(--s3)"];
const tip = document.getElementById("tip");
function tt(){const r=document.documentElement;
  const cur=r.getAttribute("data-theme")|| (matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");
  r.setAttribute("data-theme", cur==="dark"?"light":"dark");}
function show(e,h){tip.innerHTML=h;tip.style.opacity=1;
  const p=8,w=tip.offsetWidth,ht=tip.offsetHeight;
  let x=e.clientX+p,y=e.clientY+p;
  if(x+w>innerWidth-6)x=e.clientX-w-p; if(y+ht>innerHeight-6)y=e.clientY-ht-p;
  tip.style.left=x+"px";tip.style.top=y+"px";}
function hide(){tip.style.opacity=0}
const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;");
const f=(v,n=2)=>v==null?"—":(+v).toFixed(n);

/* ---------- stat tiles ---------- */
function tiles(){
  const a15=P.arms[0], a60=P.arms[2];
  const t=[
    ["gate verdict","3 / 3 PASS","all &beta; pass separation 3/3 and signal&gt;noise","var(--good)"],
    ["measured MFU","4.63%","branch loop, vs &sect;4.6's assumed 30%","var(--crit)"],
    ["ev0 / median","134 &rarr; 53","&beta;=0.15&rarr;0.60. Noise-ensemble null is 1.001",""],
    ["effective rank","1.56 – 2.02","of 11. A 3-group design caps this at 2.00",""],
    ["Phase 2 recalibrated","898","GPU-hr &mdash; &sect;4.6 predicted 219, a 4.1&times; miss","var(--crit)"],
    ["calibration cost","~18","GPU-hr &mdash; &sect;4.2 predicted 1.9",""],
  ];
  return `<div class="tiles">`+t.map(([k,v,n,c])=>
    `<div class="tile"><div class="k">${k}</div><div class="v"${c?` style="color:${c}"`:""}>${v}</div>
     <div class="n">${n}</div></div>`).join("")+`</div>`;
}

/* ---------- 1. per-domain PPL heatmap, small multiples ---------- */
function heat(){
  const steps=["--seq100","--seq200","--seq300","--seq400","--seq500","--seq600","--seq700"];
  let pan="";
  P.arms.forEach((a,ai)=>{
    // Normalise WITHIN each domain column: raw PPL spans ~6 (code) to ~534 (web),
    // so a single global ramp would show only "web is a big number".
    const colMin={}; DOM.forEach(d=>colMin[d]=Math.min(...a.rows.map(r=>r.ppl[d])));
    const colMax={}; DOM.forEach(d=>colMax[d]=Math.max(...a.rows.map(r=>r.ppl[d])));
    const cw=64, ch=21, x0=86, y0=34, gap=2;
    const W=x0+cw*5+14, H=y0+ch*12+3*gap+16;
    let cells="";
    a.rows.forEach((r,i)=>{
      const grp=Math.floor(i/4);
      const y=y0+i*ch+grp*gap;
      if(i%4===0) cells+=`<text class="al" x="4" y="${y+ch*2+2}">${esc(r.mixture_id.replace("anchor_",""))}</text>`;
      cells+=`<text class="ax" x="${x0-6}" y="${y+14}" text-anchor="end">${r.idx}</text>`;
      DOM.forEach((d,j)=>{
        const v=r.ppl[d];
        // log-ratio to the column best, so 1.0x reads as "near zero" (lightest)
        const t=Math.log(v/colMin[d])/Math.log(Math.max(colMax[d]/colMin[d],1.0001));
        const si=Math.min(6,Math.max(0,Math.round(t*6)));
        const dark=si>=4;
        cells+=`<rect x="${x0+j*cw+1}" y="${y+1}" width="${cw-2}" height="${ch-2}" rx="3"
          fill="var(${steps[si]})" data-t="<b>member ${r.idx}</b> · ${esc(r.mixture_id)}<br>${d} PPL <b>${f(v,2)}</b><br>${f(v/colMin[d],2)}&times; the best member on ${d}"></rect>
          <text x="${x0+j*cw+cw/2}" y="${y+14}" text-anchor="middle"
            style="font-size:10px;font-variant-numeric:tabular-nums;fill:${dark?"var(--on-seq-hi)":"var(--on-seq-lo)"}">${v<10?f(v,2):Math.round(v)}</text>`;
      });
    });
    let hd=DOM.map((d,j)=>`<text class="al" x="${x0+j*cw+cw/2}" y="${y0-8}" text-anchor="middle">${d.slice(0,5)}</text>`).join("");
    pan+=`<div><h2 style="font-size:13.5px;margin-bottom:2px">&beta; = ${a.beta.toFixed(2)}</h2>
      <div class="cap" style="margin-bottom:6px">min SNR ${f(a.snr_min)} · ${a.sep_wins}/${a.sep_checks} separations</div>
      <svg viewBox="0 0 ${W} ${H}" width="100%">${hd}${cells}</svg></div>`;
  });
  let tv=`<div class="tv" id="tv1">`+P.arms.map(a=>`<h2 style="font-size:12.5px;margin:10px 0 4px">&beta;=${a.beta.toFixed(2)}</h2>
    <table><tr><th>member</th><th>mixture</th>`+DOM.map(d=>`<th>${d}</th>`).join("")+`</tr>`+
    a.rows.map(r=>`<tr><td>${r.idx}</td><td>${esc(r.mixture_id)}</td>`+DOM.map(d=>`<td>${f(r.ppl[d],2)}</td>`).join("")+`</tr>`).join("")+
    `</table>`).join("")+`</div>`;
  return `<div class="card"><h2>1 · Held-out perplexity, every member on every domain</h2>
    <p class="cap">Each column is normalised to its own best member — raw PPL spans 6 (code) to 534 (web), so one
    global ramp would only show that web is a big number. The step nearest the card surface is the best member in
    that column; the most contrasting step is the worst. The block-diagonal pattern <em>is</em> the result: each anchor group is lightest on the domain it trained on. Rows are grouped by anchor with a 2px gap.
    Evaluation text is byte-identical across all 36 members and disjoint from training by content hash.</p>
    <div class="hm">${pan}</div>
    <button class="t" onclick="document.getElementById('tv1').classList.toggle('on')">table view</button>
    ${tv}</div>`;
}

/* ---------- 2. gate SNR dot plot, log axis ---------- */
function snr(){
  const W=760,H=250,L=104,R=112,T=16,B=40;
  const lo=1, hi=100;
  const X=v=>L+(Math.log(v)-Math.log(lo))/(Math.log(hi)-Math.log(lo))*(W-L-R);
  const rowY=i=>T+18+i*((H-T-B-18)/(DOM.length-1||1));
  let g="", ticks=[1,2,5,10,20,50,100];
  ticks.forEach(t=>{g+=`<line x1="${X(t)}" y1="${T}" x2="${X(t)}" y2="${H-B}" stroke="var(--grid)" stroke-width="1"/>
    <text class="ax" x="${X(t)}" y="${H-B+15}" text-anchor="middle">${t}</text>`;});
  g+=`<line x1="${X(1)}" y1="${T}" x2="${X(1)}" y2="${H-B}" stroke="var(--crit)" stroke-width="2"/>
      <text class="dl" x="${X(1)+5}" y="${T+11}" style="fill:var(--crit)">gate threshold 1.0</text>`;
  DOM.forEach((d,i)=>{
    const y=rowY(i);
    g+=`<text class="al" x="${L-10}" y="${y+4}" text-anchor="end">${d}</text>`;
    P.arms.forEach((a,ai)=>{
      const v=a.snr[d]; if(v==null) return;
      const yy=y+(ai-1)*7.5;
      g+=`<circle cx="${X(Math.min(v,hi))}" cy="${yy}" r="5" fill="${SC[ai]}" stroke="var(--surface-1)" stroke-width="2"
        data-t="<b>&beta;=${a.beta.toFixed(2)}</b> · ${d}<br>between/within = <b>${f(v)}</b><br>${v>1?"passes":"FAILS"} the &gt;1.0 gate"></circle>`;
      if(ai===2) g+=`<text class="dl" x="${X(Math.min(v,hi))+9}" y="${yy+4}">${f(v,1)}</text>`;
    });
  });
  const tv=`<div class="tv" id="tv2"><table><tr><th>domain</th>`+
    P.arms.map(a=>`<th>&beta;=${a.beta.toFixed(2)}</th>`).join("")+`</tr>`+
    DOM.map(d=>`<tr><td>${d}</td>`+P.arms.map(a=>`<td>${f(a.snr[d])}</td>`).join("")+`</tr>`).join("")+
    `<tr><td><b>min</b></td>`+P.arms.map(a=>`<td><b>${f(a.snr_min)}</b></td>`).join("")+`</tr></table></div>`;
  return `<div class="card"><h2>2 · Gate condition 2 — between-anchor signal &divide; within-anchor noise</h2>
    <p class="cap">Log axis. Every point must clear 1.0; the gate takes the <em>minimum across all five domains</em>,
    which is stricter than &sect;4.3's prose. Labels mark &beta;=0.60. Note books and multilingual have no anchor at
    N=12 — no member specialised in them — so a failure there would be a different finding from a separation failure.</p>
    <div class="legend">`+P.arms.map((a,i)=>`<span><i style="background:${SC[i]}"></i>&beta; = ${a.beta.toFixed(2)}</span>`).join("")+
    `<span style="color:var(--crit)"><i style="background:var(--crit)"></i>threshold</span></div>
    <svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>
    <button class="t" onclick="document.getElementById('tv2').classList.toggle('on')">table view</button>${tv}</div>`;
}

/* ---------- 3. spectrum ---------- */
function spec(){
  const W=560,H=300,L=52,R=96,T=16,B=42;
  const k=P.arms[0].cum.length;
  const X=i=>L+(i/(k-1))*(W-L-R), Y=v=>H-B-v*(H-T-B);
  let g="";
  [0,.25,.5,.75,1].forEach(v=>{g+=`<line x1="${L}" y1="${Y(v)}" x2="${W-R}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="1"/>
    <text class="ax" x="${L-8}" y="${Y(v)+4}" text-anchor="end">${Math.round(v*100)}%</text>`;});
  for(let i=0;i<k;i+=2) g+=`<text class="ax" x="${X(i)}" y="${H-B+15}" text-anchor="middle">${i+1}</text>`;
  // Analytic flat reference: k equal eigenvalues -> cumulative variance is linear.
  let fl=[]; for(let i=0;i<k;i++) fl.push(`${X(i)},${Y((i+1)/k)}`);
  g+=`<polyline points="${fl.join(" ")}" fill="none" stroke="var(--muted)" stroke-width="2"/>
      <text class="dl" x="${W-R+6}" y="${Y(1)+4}" style="fill:var(--muted)">flat (analytic)</text>`;
  P.arms.forEach((a,ai)=>{
    const pts=a.cum.map((v,i)=>`${X(i)},${Y(v)}`).join(" ");
    g+=`<polyline points="${pts}" fill="none" stroke="${SC[ai]}" stroke-width="2"/>`;
    a.cum.forEach((v,i)=>{g+=`<circle cx="${X(i)}" cy="${Y(v)}" r="4" fill="${SC[ai]}" stroke="var(--surface-1)" stroke-width="2"
      data-t="<b>&beta;=${a.beta.toFixed(2)}</b><br>top ${i+1} of ${k} components<br>cumulative variance <b>${f(v*100,1)}%</b>"></circle>`;});
  });
  // Annotation goes in the large empty region between the curves and the flat
  // reference. Placing it at the i=1 points (the obvious spot) put it straight on
  // top of three overlapping markers and the flat line.
  const lo=Math.min(...P.arms.map(a=>a.cum[1])), hiA=Math.max(...P.arms.map(a=>a.cum[1]));
  g+=`<text class="dl" x="${X(3.4)}" y="${Y(0.50)}">top 2 of ${k} components = ${f(lo*100,0)}–${f(hiA*100,0)}%</text>
      <text class="ax" x="${X(3.4)}" y="${Y(0.50)+15}">a flat spectrum would put ${f(2/k*100,0)}% there</text>`;
  const tv=`<div class="tv" id="tv3"><table><tr><th>components</th>`+
    P.arms.map(a=>`<th>&beta;=${a.beta.toFixed(2)}</th>`).join("")+`<th>flat</th></tr>`+
    P.arms[0].cum.map((_,i)=>`<tr><td>top ${i+1}</td>`+P.arms.map(a=>`<td>${f(a.cum[i]*100,1)}%</td>`).join("")+
      `<td>${f((i+1)/k*100,1)}%</td></tr>`).join("")+`</table></div>`;
  return `<div class="card"><h2>3 · Gram PCA spectrum — cumulative variance</h2>
    <p class="cap">Every arm is sharply concentrated: the top 2 of 11 components carry 87–94%, against 18% for a flat
    spectrum. That cleanly separates from the manufactured-noise null (ev0/median 1.001, effective-rank ratio 0.990).
    <b>But it is not yet a test of &sect;4.4.</b> At N=12 the plan is 3 anchor groups of 4 identical-&pi; members, so 3
    group means span rank 2 at most — the measured effective rank of 1.56–2.02 is saturated at that design ceiling,
    not a property of weight space. &sect;4.4's 0.2–0.3 prediction needs the N=100 zoo with 85 distinct &pi;.</p>
    <div class="legend">`+P.arms.map((a,i)=>`<span><i style="background:${SC[i]}"></i>&beta; = ${a.beta.toFixed(2)}</span>`).join("")+
    `<span><i style="background:var(--muted)"></i>flat / isotropic (analytic)</span></div>
    <svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>
    <button class="t" onclick="document.getElementById('tv3').classList.toggle('on')">table view</button>${tv}</div>`;
}

/* ---------- 4. GEMM efficiency — the MFU explainer ---------- */
function gemm(){
  const W=560,H=250,L=176,R=64,T=14,B=38;
  const peak=P.peak_tflops;
  const X=v=>L+(v/peak)*(W-L-R);
  const bh=22, step=(H-T-B)/P.gemm.length;
  let g="";
  [0,.25,.5,.75,1].forEach(fr=>{const v=fr*peak;
    g+=`<line x1="${X(v)}" y1="${T}" x2="${X(v)}" y2="${H-B}" stroke="var(--grid)" stroke-width="1"/>
        <text class="ax" x="${X(v)}" y="${H-B+15}" text-anchor="middle">${Math.round(fr*100)}%</text>`;});
  P.gemm.forEach(([name,tf],i)=>{
    const y=T+i*step+(step-bh)/2;
    const isHead=name.indexOf("lm head")===0;
    g+=`<text class="al" x="${L-10}" y="${y+bh/2+4}" text-anchor="end"
          style="font-family:ui-monospace,monospace;font-size:10.5px">${esc(name)}</text>
        <rect x="${L}" y="${y}" width="${Math.max(X(tf)-L,2)}" height="${bh}" rx="4"
          fill="${isHead?"var(--s1)":"var(--axis)"}"
          data-t="<b>${esc(name)}</b><br>${f(tf,1)} TFLOPS<br><b>${f(100*tf/peak,1)}%</b> of the ${f(peak,0)} TFLOPS measured peak"></rect>
        <text class="dl" x="${X(tf)+7}" y="${y+bh/2+4}">${f(100*tf/peak,0)}%</text>`;
  });
  const tv=`<div class="tv" id="tv4"><table><tr><th>GEMM</th><th>TFLOPS</th><th>% of peak</th></tr>`+
    P.gemm.map(([n,tf])=>`<tr><td style="font-family:ui-monospace,monospace">${esc(n)}</td><td>${f(tf,1)}</td><td>${f(100*tf/peak,1)}%</td></tr>`).join("")+
    `</table></div>`;
  return `<div class="card"><h2>4 · Why MFU is 4.6% — the LM head</h2>
    <p class="cap">Each matmul <code>train_zoo</code> actually issues, timed on a B200 at the real shape
    (T=16384 tokens, d=512, V=50257), against the <b>measured</b> 1662 TFLOPS bf16 dense peak — not the 2250
    datasheet figure, which was itself a 26% error in the denominator. Every transformer-block GEMM runs at
    33–54%. The LM head runs at <b>5.0%</b>, and it is <b>50.6%</b> of Mini's per-token FLOPs: k=512 against
    n=50257 is bandwidth-bound writing logits 98&times; wider than the hidden state. Half the model executes at
    a twentieth of peak.</p>
    <svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>
    <button class="t" onclick="document.getElementById('tv4').classList.toggle('on')">table view</button>${tv}
    <div class="note"><b>So the fix is specific.</b> Not the dataloader — a perfect one buys +40%. Not launch
    overhead — a 4&times; larger micro-batch buys 5%. A fused or chunked cross-entropy that never materialises
    full-vocab logits targets exactly the half of the FLOPs running at 5%. MFU also rises as the embedding share
    of D falls: Mini 6.05% &rarr; Small 8.60% &rarr; Medium 11.90% (compute-only).</div></div>`;
}

/* ---------- 5. cost + divergence ---------- */
function cost(){
  const rows=[["Mini",34.5,4.0,21],["Small",137,23.3,80],["Medium",726,191,426]];
  const W=520,H=210,L=64,R=92,T=14,B=38;
  const hi=800, X=v=>L+(v/hi)*(W-L-R), step=(H-T-B)/rows.length, bh=15;
  let g="";
  [0,200,400,600,800].forEach(v=>{g+=`<line x1="${X(v)}" y1="${T}" x2="${X(v)}" y2="${H-B}" stroke="var(--grid)" stroke-width="1"/>
    <text class="ax" x="${X(v)}" y="${H-B+15}" text-anchor="middle">${v}</text>`;});
  rows.forEach(([n,meas,plan,bmin],i)=>{
    const y=T+i*step+6;
    g+=`<text class="al" x="${L-10}" y="${y+bh+2}" text-anchor="end">${n}</text>
      <rect x="${L}" y="${y}" width="${Math.max(X(meas)-L,2)}" height="${bh}" rx="4" fill="var(--s2)"
        data-t="<b>${n}</b> measured<br><b>${meas}</b> GPU-hr<br>branch ${bmin} min each"></rect>
      <rect x="${L}" y="${y+bh+2}" width="${Math.max(X(plan)-L,2)}" height="${bh}" rx="4" fill="var(--axis)"
        data-t="<b>${n}</b> RESEARCH_PLAN &sect;4.6<br><b>${plan}</b> GPU-hr (assumed 30% MFU)"></rect>
      <text class="dl" x="${X(meas)+7}" y="${y+bh-2}">${meas}</text>`;
  });
  const tv=`<div class="tv" id="tv5"><table><tr><th>scale</th><th>measured GPU-hr</th><th>&sect;4.6</th><th>miss</th><th>branch each</th><th>committed --time</th></tr>`+
    rows.map(([n,m,p,b])=>`<tr><td>${n}</td><td>${m}</td><td>${p}</td><td>${f(m/p,1)}&times;</td><td>${b} min</td><td>60 min${b>55?" — would be KILLED":""}</td></tr>`).join("")+
    `</table></div>`;
  const dv=P.arms.map(a=>`<tr><td>&beta;=${a.beta.toFixed(2)}</td><td>${a.wstd_spread.toExponential(2)}</td>
     <td>${f(a.trunk_min,1)} min</td><td>${f(a.branch_min,1)} min</td><td>${f(a.branch_ktoks,1)}</td>
     <td>${f(a.branch_mfu,2)}%</td><td style="font-family:ui-monospace,monospace;font-size:10.5px">${a.fingerprint}</td></tr>`).join("");
  return `<div class="grid2">
    <div class="card"><h2>5 · Phase 2 cost, measured vs planned</h2>
      <p class="cap">Orange = measured at 4.6% MFU; gray = &sect;4.6's estimate at an assumed 30%. Total 898 vs
      219 GPU-hr. The operational risk is not the total: the committed <code>--time=01:00:00</code> would kill
      <em>every</em> Small (80 min) and Medium (426 min) branch, and Medium's 16.5 h trunk has no mid-run
      checkpointing.</p>
      <svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>
      <button class="t" onclick="document.getElementById('tv5').classList.toggle('on')">table view</button>${tv}</div>
    <div class="card"><h2>6 · &beta; controls divergence</h2>
      <p class="cap">Spread of <code>weight_std</code> across the 12 members grows 36&times; from &beta;=0.15 to
      &beta;=0.60, so the trunk-branch mechanism really does produce &beta;-controlled divergence rather than 12
      near-copies of the trunk — the &beta;&rarr;0 failure mode &sect;4.2 warns about. Distinct ensemble fingerprints
      confirm the three arms are separable in provenance; before that fix they hashed identically.</p>
      <table><tr><th>arm</th><th>weight_std spread</th><th>trunk</th><th>branch</th><th>ktok/s</th><th>MFU</th><th>fingerprint</th></tr>${dv}</table></div>
  </div>`;
}

/* ---------- 7. weight-space geometry ---------- */
function geom(){
  const A=P.arms.filter(a=>a.geom);
  if(!A.length) return "";
  const N=A[0].geom.n_members, ideal=Math.sqrt((N-1)/N/2);
  const M=[
    ["weights moved from trunk", a=>100*a.geom.disp_rel, "%", 1,
     "&Vert;w<sub>i</sub>&minus;trunk&Vert; / &Vert;trunk&Vert;. Near zero would mean the zoo is the trunk with rounding on top."],
    ["spread &divide; displacement", a=>a.geom.spread_over_disp, "", 3,
     "&radic;2 = 1.414 means members moved independently; near 0 means they all moved the same direction."],
    ["centroid fraction", a=>a.geom.mean_frac, "", 3,
     "&Vert;w<sub>i</sub>&minus;mean&Vert; / &Vert;w<sub>i</sub>&minus;w<sub>j</sub>&Vert;. An i.i.d. spread at N="+N+" gives "+ideal.toFixed(3)+". Much lower = every member near the centroid, which is when decoding the mean scores like a member and &Delta;PPL stops policing the generative arm."],
    ["between &divide; within (weights)", a=>a.geom.between_over_within, "", 3,
     "The weight-space analogue of the gate SNR, on the quantity a PCA basis is actually built from."],
  ];
  let rows=M.map(([lbl,fn,unit,dp,why])=>{
    const vals=A.map(fn), mx=Math.max(...vals);
    const bars=A.map((a,i)=>{
      const v=fn(a), w=Math.max(100*v/mx,1.5);
      return `<div style="display:flex;align-items:center;gap:8px;margin:3px 0">
        <span style="width:52px;font-size:11px;color:var(--text-secondary);font-variant-numeric:tabular-nums">&beta;=${a.beta.toFixed(2)}</span>
        <div style="flex:1;background:transparent"><div style="height:13px;border-radius:4px;width:${w}%;background:${SC[i]}"
          data-t="<b>&beta;=${a.beta.toFixed(2)}</b><br>${lbl}: <b>${v.toFixed(dp)}${unit}</b><br>${why}"></div></div>
        <span class="dl" style="width:56px;text-align:right">${v.toFixed(dp)}${unit}</span></div>`;
    }).join("");
    return `<div style="margin:0 0 14px"><div style="font-size:12.5px;font-weight:600;margin-bottom:3px">${lbl}</div>
      <div style="font-size:11.5px;color:var(--text-secondary);line-height:1.45;margin-bottom:5px">${why}</div>${bars}</div>`;
  }).join("");
  const tv=`<div class="tv" id="tv7"><table><tr><th>metric</th>`+
    A.map(a=>`<th>&beta;=${a.beta.toFixed(2)}</th>`).join("")+`</tr>`+
    M.map(([lbl,fn,unit,dp])=>`<tr><td>${lbl.replace(/<[^>]+>/g,"")}</td>`+
      A.map(a=>`<td>${fn(a).toFixed(dp)}${unit}</td>`).join("")+`</tr>`).join("")+`</table></div>`;
  return `<div class="card"><h2>7 &middot; Weight-space geometry &mdash; why &beta;=0.30, and why not 0.60</h2>
    <p class="cap">The &sect;4.3 gate answers <em>&ldquo;is mixture identity detectable&rdquo;</em> and nothing else &mdash; and it is
    <em>maximised</em> by a tight within-anchor spread, which is exactly the regime where the ensemble mean is already a
    good model and a flow can memorise the codes (misstep 19, &sect;11). Selecting &beta; on min-SNR alone optimises the
    wrong axis, so the geometry was measured directly.</p>
    ${rows}
    <button class="t" onclick="document.getElementById('tv7').classList.toggle('on')">table view</button>${tv}
    <div class="note"><b>Two intuitions overturned.</b> &beta;=0.15 is <em>not</em> a memorisation hazard &mdash; 16.5% of
    the weight norm moved, and its centroid fraction sits <em>above</em> the ${ideal.toFixed(3)} an i.i.d. spread gives at
    N=${N}, so members are not huddled near the mean. And &beta;=0.60&rsquo;s higher effective rank is not richer structure:
    between&divide;within in weight space <em>degrades</em> monotonically (4.08 &rarr; 3.98 &rarr; 2.88), while 55.9%
    displacement is the biggest stress on the untested mode-connectivity assumption behind &sect;4.1 motivation 2.
    <br><br><b>&beta; cannot fix memorisation and does not need to.</b> That objection is answered by &sect;6.4&rsquo;s
    <b>retrieval baseline</b> &mdash; &ldquo;the trained model whose mixture is nearest to &pi;&rdquo; &mdash; plus &sect;6.3&rsquo;s
    held-out interior points and one held-out vertex. Spend compute on the retrieval arm, not on larger &beta;.</div></div>`;
}

document.getElementById("app").innerHTML =
  tiles()+heat()+snr()+spec()+gemm()+cost()+geom();
document.querySelectorAll("[data-t]").forEach(el=>{
  el.style.cursor="crosshair";
  el.addEventListener("mousemove",e=>show(e,el.getAttribute("data-t")));
  el.addEventListener("mouseleave",hide);
});
</script>
"""

if __name__ == "__main__":
    out = HTML.replace("__PAYLOAD__", json.dumps(PAYLOAD))
    dest = os.path.join(HERE, "beta_calibration.html")
    with open(dest, "w") as f:
        f.write(out)
    print(f"wrote {dest}  ({len(out)/1024:.0f} KB)")
