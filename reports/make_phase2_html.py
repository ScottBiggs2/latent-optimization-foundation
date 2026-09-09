#!/usr/bin/env python3
"""
Render reports/phase2_report.html from reports/phase2_payload.json.

Self-contained: the payload is inlined and the CSS tokens are read from
reports/_tokens.css (the same block the Phase 1 report uses, whose categorical
triple passed the dataviz validator in BOTH modes -- light surface #fcfcfb and
dark #1a1a19). The file opens from disk with no network.

    python reports/build_phase2_figures.py && python reports/make_phase2_html.py

Accessibility, per the checks that apply here:
  * every chart carries a legend AND direct labels, so identity is never
    colour-alone. That also discharges the one validator WARN: light-mode
    --s3 (#1baf7a) sits at 2.74:1 against the surface, which obligates visible
    labels or a table view. Both are present.
  * every chart has a table view (the ⊞ toggle).
  * dark mode is SELECTED from the same ramps, not an automatic inversion.
  * no dual-axis chart anywhere; where two measures differ in scale they are
    separate panels.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "phase2_payload.json")) as f:
    PAYLOAD = json.load(f)
with open(os.path.join(HERE, "_tokens.css")) as f:
    TOKENS = f.read()

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 2 — the zoo at N=100, Mini</title>
<style>
__TOKENS__
  body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
       color:var(--text-primary);}
  .viz-root{max-width:1120px;margin:0 auto;padding:28px 20px 80px;}
  h1{font-size:26px;margin:0 0 4px;letter-spacing:-0.01em;}
  h2{font-size:17px;margin:38px 0 2px;letter-spacing:-0.005em;}
  .sub{color:var(--text-secondary);font-size:13px;margin:0 0 6px;}
  .note{color:var(--muted);font-size:12px;margin:6px 0 0;}
  .card{background:var(--surface-1);border:1px solid var(--border);
        border-radius:10px;padding:16px 18px;margin:12px 0;}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:10px;}
  .tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px;}
  .tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);}
  .tile .v{font-size:27px;font-weight:600;letter-spacing:-0.02em;margin:3px 0 1px;}
  .tile .d{font-size:12px;color:var(--text-secondary);}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
  @media(max-width:820px){.row{grid-template-columns:1fr;}}
  .lg{display:flex;gap:14px;flex-wrap:wrap;align-items:center;font-size:12px;
      color:var(--text-secondary);margin:2px 0 10px;}
  .lg i{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px;
        vertical-align:-1px;}
  table{border-collapse:collapse;font-size:12px;width:100%;}
  th,td{padding:4px 8px;border-bottom:1px solid var(--border);text-align:right;
        font-variant-numeric:tabular-nums;}
  th:first-child,td:first-child{text-align:left;}
  th{color:var(--muted);font-weight:500;text-transform:uppercase;font-size:10px;letter-spacing:.06em;}
  .tv{display:none;margin-top:10px;}
  .tgl{float:right;cursor:pointer;color:var(--muted);border:1px solid var(--border);
       border-radius:5px;padding:1px 7px;font-size:11px;background:none;}
  .tgl:hover{color:var(--text-primary);}
  #tip{position:fixed;pointer-events:none;background:var(--surface-1);
       border:1px solid var(--border);border-radius:7px;padding:7px 10px;font-size:12px;
       box-shadow:0 6px 22px rgba(0,0,0,.16);opacity:0;transition:opacity .08s;z-index:9;
       max-width:280px;}
  .pass{color:var(--good);font-weight:600;}
  .fail{color:var(--crit);font-weight:600;}
  .hdr{display:flex;justify-content:space-between;align-items:baseline;gap:12px;}
  svg{display:block;overflow:visible;}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;}
</style></head>
<body><div class="viz-root" id="app"></div><div id="tip"></div>
<script>
const P = __PAYLOAD__;
const DOM = P.domains, ARMS = P.arms, SDIM = P.simplex_dim, K = P.k;
const REG = ["whole_stack","block_only","embeddings_only"];
const REGL = {whole_stack:"whole stack", block_only:"blocks only",
              embeddings_only:"embeddings only"};
const CS = {whole_stack:"var(--s1)", block_only:"var(--s2)", embeddings_only:"var(--s3)"};
const AC = ["var(--s1)","var(--s2)"];
const f=(v,n)=>v==null?"—":(+v).toFixed(n===undefined?2:n);
const pc=(v,n)=>v==null?"—":(100*v).toFixed(n===undefined?1:n)+"%";
const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;");

let _n=0;
function card(title, sub, body, table, note){
  const id="t"+(_n++);
  return `<div class="card"><div class="hdr"><div>
    <div style="font-weight:600">${title}</div>
    ${sub?`<div class="sub">${sub}</div>`:""}</div>
    <button class="tgl" onclick="const e=document.getElementById('${id}');
      e.style.display=e.style.display==='block'?'none':'block'">&#8862; table</button>
    </div>${body}
    ${note?`<div class="note">${note}</div>`:""}
    <div class="tv" id="${id}">${table}</div></div>`;
}
function legend(items){
  return `<div class="lg">`+items.map(([c,l])=>
    `<span><i style="background:${c}"></i>${l}</span>`).join("")+`</div>`;
}
const tip=document.getElementById("tip");
function show(e,t){tip.innerHTML=t;tip.style.opacity=1;
  const r=tip.getBoundingClientRect();
  tip.style.left=Math.min(e.clientX+14,innerWidth-r.width-10)+"px";
  tip.style.top=Math.max(e.clientY-r.height-12,8)+"px";}
function hide(){tip.style.opacity=0;}

/* ---------- 0. headline tiles ---------- */
function tiles(){
  const a15=ARMS[0], a30=ARMS[1];
  const t=(k,v,d,cls)=>`<div class="tile"><div class="k">${k}</div>
    <div class="v ${cls||""}">${v}</div><div class="d">${d}</div></div>`;
  return `<h1>Phase 2 — the zoo at N=100, GPT-2 Mini</h1>
  <p class="sub">100 models per arm · 20 anchors (5 mixtures × 4 branches) + 80 Dirichlet
  singletons · 85 distinct π · k=99 · β = 0.15 and 0.30 · 2026-09-09</p>
  <div class="tiles">
  ${t("§4.3 gate","PASS &times;2","5/5 separation on both arms","pass")}
  ${t("min SNR","18.4 / 10.9","β=0.15 / β=0.30, threshold &gt;1.0")}
  ${t("ev0 / median","309 / 325","≫ 1: the spectrum is structured")}
  ${t("effective rank","4.47 / 4.48","of 99 &mdash; whole stack")}
  ${t("blocks only","7.20 / 7.31","1.6&times; the whole-stack figure")}
  ${t("embeddings","86.6% / 84.5%","of the variance, from 51.0% of D")}
  </div>`;
}

/* ---------- 1. effective rank by region: THE result ---------- */
function effrank(){
  const W=1040,H=290,L=150,R=24,T=18,B=52, xmax=9;
  const rows=[];
  ARMS.forEach(a=>REG.forEach(r=>rows.push(
    {beta:a.beta,reg:r,v:a.n100.regions[r].effective_rank,
     ratio:a.n100.regions[r].effective_rank_ratio,
     ev0:a.n100.regions[r].ev0_over_median})));
  const bh=(H-T-B)/rows.length, x=v=>L+(W-L-R)*v/xmax;
  let g="";
  // §4.4's predicted band, 0.2-0.3 of k=99 -> 19.8-29.7 effective dims: entirely
  // off this axis. Stated in the note rather than drawn, so the scale stays honest.
  [0,2,4,6,8].forEach(v=>{g+=`<line x1="${x(v)}" y1="${T}" x2="${x(v)}" y2="${H-B}"
    stroke="var(--grid)" stroke-width="1"/><text x="${x(v)}" y="${H-B+15}"
    fill="var(--muted)" font-size="10" text-anchor="middle">${v}</text>`;});
  g+=`<line x1="${x(SDIM)}" y1="${T-6}" x2="${x(SDIM)}" y2="${H-B+2}"
      stroke="var(--text-secondary)" stroke-width="1.5" stroke-dasharray="4 3"/>
      <text x="${x(SDIM)+6}" y="${T+2}" fill="var(--text-secondary)" font-size="10.5">
      dim(&Delta;<tspan dy="-3" font-size="8">4</tspan>)<tspan dy="3"> = ${SDIM}</tspan></text>`;
  rows.forEach((r,i)=>{
    const y=T+i*bh+4, h=bh-8, w=Math.max(x(r.v)-L,2);
    g+=`<rect x="${L}" y="${y}" width="${w}" height="${h}" rx="4" fill="${CS[r.reg]}"
        data-t="&lt;b&gt;${REGL[r.reg]}&lt;/b&gt;, &beta;=${r.beta}&lt;br&gt;
        effective rank ${f(r.v)} of ${K}&lt;br&gt;ratio ${f(r.ratio,4)}&lt;br&gt;
        ev0/median ${f(r.ev0,1)}"/>
        <text x="${L-8}" y="${y+h/2+4}" text-anchor="end" fill="var(--text-secondary)"
        font-size="11">&beta;=${r.beta} · ${REGL[r.reg]}</text>
        <text x="${L+w+7}" y="${y+h/2+4}" fill="var(--text-primary)" font-size="11.5"
        font-weight="600">${f(r.v)}</text>`;
  });
  g+=`<text x="${(L+W-R)/2}" y="${H-B+34}" fill="var(--muted)" font-size="10.5"
     text-anchor="middle">effective rank (participation ratio) out of k=${K}</text>`;
  const tbl=`<table><tr><th>arm</th><th>region</th><th>eff. rank</th>
    <th>ratio</th><th>ev0/median</th><th>var share</th></tr>`+rows.map(r=>{
    const a=ARMS.find(x=>x.beta===r.beta);
    return `<tr><td>β=${r.beta}</td><td>${REGL[r.reg]}</td><td>${f(r.v)}</td>
      <td>${f(r.ratio,4)}</td><td>${f(r.ev0,1)}</td>
      <td>${pc(a.n100.regions[r.reg].variance_share)}</td></tr>`;}).join("")+`</table>`;
  return card("Effective rank of the weight distribution, by region",
    "The §4.4 test, at last unceilinged: 85 distinct π means the mean-centred rank is the full 99.",
    legend(REG.map(r=>[CS[r],REGL[r]]))+`<svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>`,
    tbl,
    `§4.4 pre-registered <span class="mono">effective_rank_ratio ≈ 0.2–0.3</span>, i.e. 19.8–29.7
     effective dimensions of 99 — far off the right of this axis. Measured is 4.5 whole-stack.
     It misses the band, but in the opposite direction from the stated disproof condition
     (a <i>flat</i> spectrum, ratio → 1.0): <span class="mono">ev0/median</span> of 309–325 says the
     structure is emphatically present. The dashed line is the dimension of the mixture simplex
     Δ<sup>4</sup> over five domains — the embeddings land on it to within 0.05, and the blocks
     carry ~1.8× more.`);
}

/* ---------- 2. cumulative variance: where the elbow is ---------- */
function cumvar(){
  const W=505,H=250,L=42,R=46,T=14,B=44, n=10;
  function panel(a){
    const x=i=>L+(W-L-R)*i/(n-1), y=v=>H-B-(H-T-B)*v;
    let g="";
    [0,.25,.5,.75,1].forEach(v=>{g+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}"
      stroke="var(--grid)" stroke-width="1"/><text x="${L-7}" y="${y(v)+3.5}"
      fill="var(--muted)" font-size="10" text-anchor="end">${(100*v)|0}%</text>`;});
    for(let i=0;i<n;i+=2) g+=`<text x="${x(i)}" y="${H-B+15}" fill="var(--muted)"
      font-size="10" text-anchor="middle">${i+1}</text>`;
    g+=`<line x1="${x(SDIM-1)}" y1="${T}" x2="${x(SDIM-1)}" y2="${H-B}"
        stroke="var(--text-secondary)" stroke-width="1.4" stroke-dasharray="4 3"/>`;
    REG.forEach(r=>{
      const cv=a.n100.regions[r].cumvar.slice(0,n);
      const d=cv.map((v,i)=>`${i?"L":"M"}${x(i)},${y(v)}`).join("");
      g+=`<path d="${d}" fill="none" stroke="${CS[r]}" stroke-width="2"
           stroke-linejoin="round"/>`;
      cv.forEach((v,i)=>{g+=`<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="${CS[r]}"
        stroke="var(--surface-1)" stroke-width="2"
        data-t="&lt;b&gt;${REGL[r]}&lt;/b&gt;, &beta;=${a.beta}&lt;br&gt;
        first ${i+1} component${i?"s":""} capture ${pc(v)}"/>`;});
      g+=`<text x="${x(n-1)+6}" y="${y(cv[n-1])+3.5}" fill="${CS[r]}" font-size="10.5"
           font-weight="600">${pc(cv[n-1],0)}</text>`;
    });
    g+=`<text x="${x(SDIM-1)+5}" y="${T+11}" fill="var(--text-secondary)"
        font-size="10">c${SDIM}</text>`;
    g+=`<text x="${L}" y="${T+11}" fill="var(--text-primary)" font-size="11.5"
        font-weight="600">&beta; = ${a.beta}</text>`;
    g+=`<text x="${(L+W-R)/2}" y="${H-B+32}" fill="var(--muted)" font-size="10.5"
        text-anchor="middle">component index</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>`;
  }
  let tbl=`<table><tr><th>arm</th><th>region</th>`+
    Array.from({length:n},(_,i)=>`<th>c${i+1}</th>`).join("")+`</tr>`;
  ARMS.forEach(a=>REG.forEach(r=>{tbl+=`<tr><td>β=${a.beta}</td><td>${REGL[r]}</td>`+
    a.n100.regions[r].cumvar.slice(0,n).map(v=>`<td>${pc(v)}</td>`).join("")+`</tr>`;}));
  tbl+=`</table>`;
  return card("Cumulative variance captured",
    "Same three regions, as a share of each region's own centred variance.",
    legend(REG.map(r=>[CS[r],REGL[r]]))+
    `<div class="row">${panel(ARMS[0])}${panel(ARMS[1])}</div>`, tbl,
    `The elbow sits at component 3–4 for the <b>embeddings</b>, which is what makes their
     participation ratio land on dim(Δ<sup>4</sup>). But it is an elbow, not a cliff: 18% of
     embedding variance and <b>32% of block variance</b> lie beyond c${SDIM}. So "the zoo is
     4-dimensional" is a statement about where the mass is, not a rank claim — and the blocks
     are visibly the richer object, never reaching 90% inside 24 components where the
     embeddings reach it by c13.`);
}

/* ---------- 3. the §4.5 confound in one picture ---------- */
function confound(){
  const W=1040,H=150,L=150,R=120,T=16,B=34;
  const rows=[];
  ARMS.forEach(a=>{
    const e=a.n100.regions.embeddings_only, b=a.n100.regions.block_only;
    rows.push({beta:a.beta,lab:"embeddings",d:0.51,v:e.variance_share,c:CS.embeddings_only});
    rows.push({beta:a.beta,lab:"blocks",d:0.49,v:b.variance_share,c:CS.block_only});
  });
  const bh=(H-T-B)/rows.length, x=v=>L+(W-L-R)*v;
  let g="";
  [0,.25,.5,.75,1].forEach(v=>{g+=`<line x1="${x(v)}" y1="${T}" x2="${x(v)}" y2="${H-B}"
    stroke="var(--grid)" stroke-width="1"/><text x="${x(v)}" y="${H-B+14}"
    fill="var(--muted)" font-size="10" text-anchor="middle">${(100*v)|0}%</text>`;});
  rows.forEach((r,i)=>{
    const y=T+i*bh+3, h=(bh-10)/2;
    g+=`<rect x="${L}" y="${y}" width="${Math.max(x(r.d)-L,2)}" height="${h}" rx="3"
         fill="${r.c}" opacity="0.34"
         data-t="&lt;b&gt;${r.lab}&lt;/b&gt;&lt;br&gt;${pc(r.d)} of D (parameter count)"/>
        <rect x="${L}" y="${y+h+3}" width="${Math.max(x(r.v)-L,2)}" height="${h}" rx="3"
         fill="${r.c}"
         data-t="&lt;b&gt;${r.lab}&lt;/b&gt;, &beta;=${r.beta}&lt;br&gt;${pc(r.v)} of the centred variance"/>
        <text x="${L-8}" y="${y+bh/2}" text-anchor="end" fill="var(--text-secondary)"
         font-size="11">&beta;=${r.beta} · ${r.lab}</text>
        <text x="${x(Math.max(r.d,r.v))+8}" y="${y+bh/2}" fill="var(--text-primary)"
         font-size="11">${pc(r.d,0)} of D → <tspan font-weight="600">${pc(r.v)}</tspan> of var</text>`;
  });
  const tbl=`<table><tr><th>arm</th><th>region</th><th>share of D</th>
    <th>share of variance</th><th>eff. rank</th></tr>`+rows.map(r=>{
    const a=ARMS.find(x=>x.beta===r.beta);
    const k=r.lab==="embeddings"?"embeddings_only":"block_only";
    return `<tr><td>β=${r.beta}</td><td>${r.lab}</td><td>${pc(r.d)}</td>
      <td>${pc(r.v)}</td><td>${f(a.n100.regions[k].effective_rank)}</td></tr>`;
  }).join("")+`</table>`;
  return card("§4.5's embedding-share confound, measured",
    "Pale bar: share of the parameter count D. Solid bar: share of the centred variance.",
    legend([[CS.embeddings_only,"embeddings (wte + wpe + final LN)"],
            [CS.block_only,"transformer blocks"]])+
    `<svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>`, tbl,
    `§4.5 flagged this as something to <i>declare</i> next to the scaling figure. Measured, it is
     <b>dominant</b>: the embedding table is barely over half of D and carries ~86% of the
     variance, so a whole-stack spectrum is substantially an embedding spectrum. Every §4.7
     ladder point needs both curves. Note the vocab is fixed at 50257, so this share falls
     51.0% → 31.6% → 14.8% from Mini to Small to Medium — the composition of D changes along
     the ladder, not only its size.`);
}

/* ---------- 4. the gate, N=12 vs N=100 ---------- */
function gate(){
  const W=1040,H=250,L=54,R=18,T=18,B=56;
  const series=[];
  ARMS.forEach((a,ai)=>{
    series.push({lab:`β=${a.beta}, N=12`, c:AC[ai], dash:"3 3",
                 v:DOM.map(d=>a.n12.snr[d])});
    series.push({lab:`β=${a.beta}, N=100`, c:AC[ai], dash:"",
                 v:DOM.map(d=>a.n100.snr[d])});
  });
  const vmax=90, gw=(W-L-R)/DOM.length, bw=gw/(series.length+1.6);
  const y=v=>H-B-(H-T-B)*Math.min(v,vmax)/vmax;
  let g="";
  [1,20,40,60,80].forEach(v=>{g+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}"
    stroke="var(--grid)" stroke-width="1"/><text x="${L-7}" y="${y(v)+3.5}"
    fill="var(--muted)" font-size="10" text-anchor="end">${v}</text>`;});
  g+=`<line x1="${L}" y1="${y(1)}" x2="${W-R}" y2="${y(1)}" stroke="var(--crit)"
      stroke-width="1.4" stroke-dasharray="5 3"/>
      <text x="${W-R-4}" y="${y(1)-5}" fill="var(--crit)" font-size="10"
      text-anchor="end">gate threshold  SNR &gt; 1</text>`;
  DOM.forEach((d,di)=>{
    const x0=L+di*gw+gw*0.14;
    series.forEach((s,si)=>{
      const v=s.v[di]; if(v==null) return;
      const x=x0+si*bw;
      g+=`<rect x="${x}" y="${y(v)}" width="${bw-2}" height="${H-B-y(v)}" rx="4"
           fill="${s.c}" ${s.dash?'opacity="0.4"':''}
           data-t="&lt;b&gt;${d}&lt;/b&gt;&lt;br&gt;${s.lab}&lt;br&gt;signal/noise ${f(v)}"/>`;
    });
    g+=`<text x="${x0+bw*series.length/2}" y="${H-B+15}" fill="var(--text-secondary)"
        font-size="10.5" text-anchor="middle">${d}</text>`;
  });
  g+=`<text x="${L-40}" y="${T+4}" fill="var(--muted)" font-size="10.5"
      transform="rotate(-90 ${L-40} ${T+4})" text-anchor="end">between / within</text>`;
  let tbl=`<table><tr><th>domain</th>`+series.map(s=>`<th>${s.lab}</th>`).join("")+`</tr>`;
  DOM.forEach((d,di)=>{tbl+=`<tr><td>${d}</td>`+
    series.map(s=>`<td>${f(s.v[di])}</td>`).join("")+`</tr>`;});
  tbl+=`<tr><td><b>separation</b></td>`+ARMS.map(a=>
    `<td>${a.n12.separation.wins}/${a.n12.separation.checks}</td>
     <td>${a.n100.separation.wins}/${a.n100.separation.checks}</td>`).join("")+`</tr></table>`;
  return card("§4.3 gate: signal &divide; noise per domain",
    "Between-anchor-group spread over the within-group spread. Pale = N=12 (Phase 1), solid = N=100.",
    legend(series.map(s=>[s.c,s.lab]))+
    `<svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>`, tbl,
    `The N=100 gate is a <b>stricter</b> test that passes more comfortably. At N=12 only three
     domains had an anchor group, so books and multilingual had no specialist and scored 5.96 and
     7.02 — those were the min-SNR domains that decided Phase 1. With all five anchors present
     every domain has a specialist, separation is checked on 5/5 rather than 3/3, and the
     <i>minimum</i> SNR rises 5.96 → 18.37 (β=0.15). The three originally-anchored domains fall
     because their between-group spread is now averaged over five groups instead of three.`);
}

/* ---------- 5. §6.5 slope: the control test ---------- */
function slopes(){
  if(!P.probe_slopes || !P.probe_slopes.length) return "";
  const W=196,H=168,L=34,R=10,T=12,B=32;
  function panel(arm,d,di){
    const pts=arm.points.map(r=>({x:r.pi[di], lp:Math.log(r.ppl[d])}));
    const mlp=pts.reduce((s,p)=>s+p.lp,0)/pts.length;
    pts.forEach(p=>p.y=mlp-p.lp);
    const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
    const x0=Math.min(...xs), x1=Math.max(...xs);
    const ya=Math.min(...ys), yb=Math.max(...ys);
    const pad=(yb-ya)*0.25||0.02;
    const X=v=>L+(W-L-R)*(v-x0)/((x1-x0)||1);
    const Y=v=>H-B-(H-T-B)*(v-(ya-pad))/((yb-ya+2*pad)||1);
    const st=arm.per_domain[d]||{};
    const mx=xs.reduce((a,b)=>a+b,0)/xs.length, my=ys.reduce((a,b)=>a+b,0)/ys.length;
    let g="";
    g+=`<line x1="${L}" y1="${Y(0)}" x2="${W-R}" y2="${Y(0)}" stroke="var(--grid)"
        stroke-width="1"/>`;
    if(st.slope!=null){
      const yA=my+st.slope*(x0-mx), yB=my+st.slope*(x1-mx);
      g+=`<line x1="${X(x0)}" y1="${Y(yA)}" x2="${X(x1)}" y2="${Y(yB)}"
          stroke="${arm.c}" stroke-width="2" stroke-linecap="round"/>`;
    }
    pts.forEach(p=>{g+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="4.5" fill="${arm.c}"
      stroke="var(--surface-1)" stroke-width="2"
      data-t="&lt;b&gt;${d}&lt;/b&gt;, &beta;=${arm.beta}&lt;br&gt;requested &pi;=${f(p.x,3)}
      &lt;br&gt;advantage ${f(p.y,3)} nats"/>`;});
    const sgn=st.slope>0?"pass":"fail";
    g+=`<text x="${L}" y="${T+4}" fill="var(--text-primary)" font-size="10.5"
        font-weight="600">${d}</text>
        <text x="${W-R}" y="${T+4}" text-anchor="end" font-size="10"
        class="${sgn}">${st.slope>0?"+":""}${f(st.slope)}</text>
        <text x="${W-R}" y="${H-B+13}" text-anchor="end" fill="var(--muted)"
        font-size="9.5">p=${f(st.p_one_sided,3)}</text>
        <text x="${L}" y="${H-B+13}" fill="var(--muted)" font-size="9.5">requested &pi;</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>`;
  }
  let body="";
  P.probe_slopes.forEach((arm,ai)=>{
    arm.c=AC[ai];
    // swatch on the row header: the two arms are distinguished by colour, so
    // without this the identity would be colour-alone (dataviz check 6).
    body+=`<div style="margin:6px 0 2px;font-size:12px;color:var(--text-secondary)">
      <i style="background:${arm.c};width:10px;height:10px;border-radius:2px;
        display:inline-block;margin-right:5px;vertical-align:-1px"></i>
      <b style="color:var(--text-primary)">&beta; = ${arm.beta}</b> &nbsp;·&nbsp;
      pooled slope <b>${arm.pooled_slope>0?"+":""}${f(arm.pooled_slope,3)}</b> ·
      exact p = ${f(arm.pooled_p,4)} over ${arm.n_perms} permutations ·
      ${arm.n_positive}/5 domains positive</div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px">`+
      DOM.map((d,di)=>panel(arm,d,di)).join("")+`</div>`;
  });
  let tbl=`<table><tr><th>arm</th><th>domain</th><th>slope</th><th>r</th>
    <th>exact p</th><th>π range</th></tr>`;
  P.probe_slopes.forEach(a=>DOM.forEach(d=>{const s=a.per_domain[d]||{};
    tbl+=`<tr><td>β=${a.beta}</td><td>${d}</td><td>${f(s.slope,3)}</td>
      <td>${f(s.r,3)}</td><td>${f(s.p_one_sided,3)}</td>
      <td>${f(s.pi_min,3)}–${f(s.pi_max,3)}</td></tr>`;}));
  P.probe_slopes.forEach(a=>{tbl+=`<tr><td>β=${a.beta}</td><td><b>pooled</b></td>
    <td><b>${f(a.pooled_slope,3)}</b></td><td>—</td>
    <td><b>${f(a.pooled_p,4)}</b></td><td>—</td></tr>`;});
  tbl+=`</table>`;
  return card("§6.5 slope — does the requested mixture <i>control</i> the model?",
    "The singleton probe: 6 members at min pairwise L1 = 0.164, harder than Phase 2's closest pair (0.173).",
    body, tbl,
    `Vertical axis is the cohort-relative log-PPL advantage
     <span class="mono">mean_j(ln ppl_j) − ln ppl_i</span>; positive is better than the cohort.
     A <b>control</b> test rather than a quality test, which is why §6.5 calls it immune to
     misstep 19's collapse trap. The null is exact — all 6! = 720 permutations of the
     member↔π pairing, enumerated — so the p floor is 1/720 = 0.0014 and both arms sit on it.
     <b>books</b> is the outlier at both β and negative at 0.30: Gutenberg is generic English, so
     web and math training also improve it, and π<sub>books</sub> has little marginal effect
     inside a blend.`);
}

/* ---------- 6. geometry ---------- */
function geom(){
  const keys=[["disp_rel","displacement from trunk","how much of the weight norm moved",1],
              ["spread_over_disp","spread / displacement","√2 = moved independently",0],
              ["mean_frac","centroid fraction","i.i.d. → 0.704 at N=100",0],
              ["between_over_within","between / within","the PCA basis's own SNR",0]];
  let tbl=`<table><tr><th>statistic</th>
    <th>β=.15 N=12</th><th>β=.15 probe</th><th>β=.15 N=100</th>
    <th>β=.30 N=12</th><th>β=.30 probe</th><th>β=.30 N=100</th></tr>`;
  keys.forEach(([k,lab,hint,ispc])=>{
    tbl+=`<tr><td>${lab}<div class="note" style="margin:0">${hint}</div></td>`;
    ARMS.forEach(a=>{
      [a.n12.geom,a.probe.geom,a.n100.geom].forEach(g=>{
        const v=(g||{})[k];
        tbl+=`<td>${v==null?"—":(ispc?pc(v,2):f(v,3))}</td>`;});
    });
    tbl+=`</tr>`;
  });
  tbl+=`</table>`;
  return card("Weight-space geometry across the three zoos",
    "The half of the β question the gate cannot see (misstep 19's mechanism).",
    tbl, tbl,
    `The N=100 zoos move <b>far more independently</b> than either earlier zoo:
     <span class="mono">spread/displacement</span> is 1.136 at β=0.15 against 0.950 for the N=12
     anchors and 0.748 for the probe. That confirms the probe was a lower bound by construction —
     matching Phase 2's closest pair with only 6 draws forced α=8, which clusters them at the
     barycentre, and the barycentre <i>is</i> the uniform mixture the trunk trained on.
     <span class="mono">between/within</span> is now essentially identical across β (3.963 vs
     3.925), so unlike at N=12 this statistic no longer discriminates.`);
}

document.getElementById("app").innerHTML =
  tiles()+`<h2>The §4.4 test</h2>`+effrank()+cumvar()+
  `<h2>The confound §4.5 asked us to control</h2>`+confound()+
  `<h2>The gate, and the control test</h2>`+gate()+slopes()+
  `<h2>Geometry</h2>`+geom();
document.querySelectorAll("[data-t]").forEach(el=>{
  el.style.cursor="crosshair";
  el.addEventListener("mousemove",e=>show(e,el.getAttribute("data-t")));
  el.addEventListener("mouseleave",hide);
});
</script>
</body></html>
"""

if __name__ == "__main__":
    out = HTML.replace("__TOKENS__", TOKENS).replace("__PAYLOAD__", json.dumps(PAYLOAD))
    dest = os.path.join(HERE, "phase2_report.html")
    with open(dest, "w") as f:
        f.write(out)
    print(f"wrote {dest}  ({len(out)/1024:.0f} KB)")
