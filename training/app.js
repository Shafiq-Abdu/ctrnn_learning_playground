(() => {
"use strict";

const $ = id => document.getElementById(id);

const CFG = {
  inputSize: 1,
  hiddenSize: 4,
  outputSize: 1,
  dt: 0.1,
  tau: 1.0,
  duration: 6.0,
  inputStart: 1.0,
  inputEnd: 2.0,
  targetStart: 3.0,
  targetEnd: 5.0
};

function refreshDerivedConfig(){
  CFG.steps = Math.round(CFG.duration / CFG.dt);
  CFG.alpha = CFG.dt / CFG.tau;
}
refreshDerivedConfig();

const S = {
  running: false,
  epoch: 0,
  lossHistory: [],
  lastLoss: NaN,
  lastGradNorm: NaN,
  Win: null,
  Wrec: null,
  Wout: null,
  b: null,
  bout: null,
  lastForward: null,
  animationId: null
};

function zeros(n){ return Array(n).fill(0); }
function matZeros(r,c){ return Array.from({length:r}, () => zeros(c)); }

function randn(){
  let u = 0, v = 0;
  while(u === 0) u = Math.random();
  while(v === 0) v = Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}

function resetWeights(){
  S.running = false;
  S.epoch = 0;
  S.lossHistory = [];

  S.Win = matZeros(CFG.hiddenSize, CFG.inputSize);
  S.Wrec = matZeros(CFG.hiddenSize, CFG.hiddenSize);
  S.Wout = matZeros(CFG.outputSize, CFG.hiddenSize);
  S.b = zeros(CFG.hiddenSize);
  S.bout = zeros(CFG.outputSize);

  const inScale = 0.5 / Math.sqrt(Math.max(1, CFG.inputSize));
  const recScale = 0.22 / Math.sqrt(Math.max(1, CFG.hiddenSize));
  const outScale = 0.35 / Math.sqrt(Math.max(1, CFG.hiddenSize));

  for(let i=0;i<CFG.hiddenSize;i++){
    for(let k=0;k<CFG.inputSize;k++){
      S.Win[i][k] = inScale * randn();
    }
    for(let j=0;j<CFG.hiddenSize;j++){
      S.Wrec[i][j] = recScale * randn() + (i===j ? 0.70 : 0);
    }
  }

  for(let o=0;o<CFG.outputSize;o++){
    for(let j=0;j<CFG.hiddenSize;j++){
      S.Wout[o][j] = outScale * randn();
    }
  }

  S.lastForward = forwardPass();
  S.lastLoss = S.lastForward.loss;
  S.lastGradNorm = NaN;

  updateArchitectureText();
  updateAll();
  $("statusText").textContent = "Weights reset";
}

function inputVectorAt(t){
  const x = zeros(CFG.inputSize);
  x[0] = (t >= CFG.inputStart && t < CFG.inputEnd) ? 1 : 0;
  return x;
}

function targetVectorAt(t){
  const y = zeros(CFG.outputSize);
  y[0] = (t >= CFG.targetStart && t < CFG.targetEnd) ? 1 : 0;
  return y;
}

function tanhDerivFromA(a){
  const z = Math.tanh(a);
  return 1 - z*z;
}

function forwardPass(){
  const T = CFG.steps, H = CFG.hiddenSize, I = CFG.inputSize, O = CFG.outputSize;

  const h = Array.from({length:T+1}, () => zeros(H));
  const a = Array.from({length:T}, () => zeros(H));
  const y = Array.from({length:T}, () => zeros(O));
  const x = Array.from({length:T}, () => zeros(I));
  const target = Array.from({length:T}, () => zeros(O));
  const time = zeros(T);

  let lossSum = 0;

  for(let t=0;t<T;t++){
    const timeNow = t * CFG.dt;
    time[t] = timeNow;
    x[t] = inputVectorAt(timeNow);
    target[t] = targetVectorAt(timeNow);

    for(let i=0;i<H;i++){
      let z = S.b[i];

      for(let k=0;k<I;k++){
        z += S.Win[i][k] * x[t][k];
      }

      for(let j=0;j<H;j++){
        z += S.Wrec[i][j] * h[t][j];
      }

      a[t][i] = z;
      h[t+1][i] =
        (1-CFG.alpha)*h[t][i] +
        CFG.alpha*Math.tanh(z);
    }

    for(let o=0;o<O;o++){
      let out = S.bout[o];
      for(let j=0;j<H;j++){
        out += S.Wout[o][j] * h[t+1][j];
      }
      y[t][o] = out;

      const e = out - target[t][o];
      lossSum += e*e;
    }
  }

  return {
    h,a,y,x,target,time,
    loss: lossSum/(T*O)
  };
}

function backwardPass(F){
  const T = CFG.steps, H = CFG.hiddenSize, I = CFG.inputSize, O = CFG.outputSize;

  const gWin = matZeros(H,I);
  const gWrec = matZeros(H,H);
  const gWout = matZeros(O,H);
  const gb = zeros(H);
  const gbout = zeros(O);

  let gFuture = zeros(H);

  for(let t=T-1;t>=0;t--){
    const dy = zeros(O);

    for(let o=0;o<O;o++){
      const err = F.y[t][o] - F.target[t][o];
      dy[o] = 2 * err / (T*O);

      for(let j=0;j<H;j++){
        gWout[o][j] += dy[o] * F.h[t+1][j];
      }

      gbout[o] += dy[o];
    }

    // Current readout contribution + future recurrent contribution.
    const gNext = zeros(H);

    for(let i=0;i<H;i++){
      let readoutTerm = 0;
      for(let o=0;o<O;o++){
        readoutTerm += S.Wout[o][i] * dy[o];
      }
      gNext[i] = gFuture[i] + readoutTerm;
    }

    // Euler CTRNN derivative.
    const q = zeros(H);

    for(let i=0;i<H;i++){
      q[i] =
        CFG.alpha *
        gNext[i] *
        tanhDerivFromA(F.a[t][i]);

      gb[i] += q[i];

      for(let k=0;k<I;k++){
        gWin[i][k] += q[i] * F.x[t][k];
      }

      for(let j=0;j<H;j++){
        gWrec[i][j] += q[i] * F.h[t][j];
      }
    }

    const gPrev = zeros(H);

    for(let j=0;j<H;j++){
      let recurrentTerm = 0;

      for(let i=0;i<H;i++){
        recurrentTerm += S.Wrec[i][j] * q[i];
      }

      gPrev[j] =
        (1-CFG.alpha)*gNext[j] +
        recurrentTerm;
    }

    gFuture = gPrev;
  }

  return {gWin,gWrec,gWout,gb,gbout};
}

function globalNorm(G){
  let s = 0;

  for(const row of G.gWin) for(const v of row) s += v*v;
  for(const row of G.gWrec) for(const v of row) s += v*v;
  for(const row of G.gWout) for(const v of row) s += v*v;
  for(const v of G.gb) s += v*v;
  for(const v of G.gbout) s += v*v;

  return Math.sqrt(s);
}

function clipGradients(G, threshold){
  const n = globalNorm(G);

  if(n <= threshold || threshold <= 0) return n;

  const c = threshold / (n + 1e-12);

  for(const row of G.gWin){
    for(let j=0;j<row.length;j++) row[j] *= c;
  }

  for(const row of G.gWrec){
    for(let j=0;j<row.length;j++) row[j] *= c;
  }

  for(const row of G.gWout){
    for(let j=0;j<row.length;j++) row[j] *= c;
  }

  for(let i=0;i<G.gb.length;i++) G.gb[i] *= c;
  for(let i=0;i<G.gbout.length;i++) G.gbout[i] *= c;

  return n;
}

function applySGD(G, lr){
  for(let i=0;i<CFG.hiddenSize;i++){
    for(let k=0;k<CFG.inputSize;k++){
      S.Win[i][k] -= lr * G.gWin[i][k];
    }

    S.b[i] -= lr * G.gb[i];

    for(let j=0;j<CFG.hiddenSize;j++){
      S.Wrec[i][j] -= lr * G.gWrec[i][j];
    }
  }

  for(let o=0;o<CFG.outputSize;o++){
    for(let j=0;j<CFG.hiddenSize;j++){
      S.Wout[o][j] -= lr * G.gWout[o][j];
    }
    S.bout[o] -= lr * G.gbout[o];
  }
}

function trainOneEpoch(){
  const F = forwardPass();
  const G = backwardPass(F);

  const clip = Math.max(0.0001, Number($("clipInput").value) || 1);
  const lr = Math.max(0.000001, Number($("lrInput").value) || 0.1);

  S.lastGradNorm = clipGradients(G, clip);
  applySGD(G, lr);

  S.epoch += 1;

  S.lastForward = forwardPass();
  S.lastLoss = S.lastForward.loss;
  S.lossHistory.push(S.lastLoss);

  return S.lastLoss;
}

function animateTraining(){
  if(!S.running) return;

  const limit = Math.max(1, parseInt($("epochLimitInput").value) || 1500);
  const perFrame = Math.max(1, parseInt($("epochsPerFrameInput").value) || 5);

  for(let k=0;k<perFrame && S.epoch<limit;k++){
    trainOneEpoch();
  }

  updateAll();

  if(S.epoch >= limit){
    S.running = false;
    $("statusText").textContent = "Training complete";
    return;
  }

  S.animationId = requestAnimationFrame(animateTraining);
}

function startTraining(){
  if(S.running) return;
  S.running = true;
  $("statusText").textContent = "Training";
  animateTraining();
}

function pauseTraining(){
  S.running = false;
  $("statusText").textContent = "Paused";
  updateAll();
}

function stepEpoch(){
  S.running = false;

  const limit = Math.max(1, parseInt($("epochLimitInput").value) || 1500);

  if(S.epoch < limit){
    trainOneEpoch();
    $("statusText").textContent = "Stepped 1 epoch";
  }else{
    $("statusText").textContent = "Epoch limit reached";
  }

  updateAll();
}

function applyArchitecture(){
  S.running = false;

  CFG.inputSize = parseInt($("inputSizeSelect").value);
  CFG.hiddenSize = parseInt($("hiddenSizeSelect").value);
  CFG.outputSize = parseInt($("outputSizeSelect").value);

  resetWeights();
  drawTask();

  $("statusText").textContent = "Architecture applied";
}

function updateArchitectureText(){
  $("archInputText").textContent = CFG.inputSize;
  $("archHiddenText").textContent = CFG.hiddenSize;
  $("archOutputText").textContent = CFG.outputSize;
}

/* ---------- Canvas utilities ---------- */

function clearCanvas(canvas){
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0,0,canvas.width,canvas.height);
  return ctx;
}

function bounds(values, pad=0.12){
  let mn = Infinity, mx = -Infinity;

  for(const v of values){
    if(!Number.isFinite(v)) continue;
    mn = Math.min(mn,v);
    mx = Math.max(mx,v);
  }

  if(!Number.isFinite(mn)){
    mn = -1;
    mx = 1;
  }

  if(Math.abs(mx-mn)<1e-9){
    mn -= 1;
    mx += 1;
  }

  const p = (mx-mn)*pad;
  return [mn-p,mx+p];
}

function drawAxes(ctx,w,h,xmin,xmax,ymin,ymax,xLabel="time (s)",yLabel=""){
  const L=54,R=18,T=18,B=38;

  ctx.strokeStyle="#9aa7b2";
  ctx.lineWidth=1;
  ctx.beginPath();
  ctx.moveTo(L,T);
  ctx.lineTo(L,h-B);
  ctx.lineTo(w-R,h-B);
  ctx.stroke();

  ctx.fillStyle="#667785";
  ctx.font="12px Arial";

  for(let k=0;k<=6;k++){
    const x=L+(w-L-R)*k/6;
    const v=xmin+(xmax-xmin)*k/6;
    ctx.fillText(v.toFixed(1),x-10,h-B+18);
  }

  for(let k=0;k<=4;k++){
    const y=T+(h-T-B)*k/4;
    const v=ymax-(ymax-ymin)*k/4;
    ctx.fillText(v.toFixed(2),5,y+4);
  }

  ctx.fillText(xLabel,(L+w-R)/2-25,h-7);
  if(yLabel) ctx.fillText(yLabel,8,12);

  return {L,R,T,B};
}

function plotSeries(canvas, series, xvals, opts={}){
  const ctx = clearCanvas(canvas);
  const w = canvas.width;
  const h = canvas.height;

  const all = [];
  for(const s of series) all.push(...s.values);

  const [ymin,ymax] = bounds(all,0.15);
  const xmin = xvals[0] || 0;
  const xmax = xvals[xvals.length-1] || 1;

  const A = drawAxes(
    ctx,w,h,xmin,xmax,ymin,ymax,
    opts.xLabel || "time (s)",
    opts.yLabel || ""
  );

  function sx(x){
    return A.L+(x-xmin)/(xmax-xmin)*(w-A.L-A.R);
  }

  function sy(y){
    return A.T+(ymax-y)/(ymax-ymin)*(h-A.T-A.B);
  }

  const colors = [
    "#222","#7a8792","#2b6fbb","#c46a27",
    "#5c8b57","#8c5baa","#b04c55","#2a8f8f"
  ];

  series.forEach((s,idx)=>{
    ctx.save();
    ctx.strokeStyle = s.color || colors[idx%colors.length];
    ctx.lineWidth = s.width || 2;

    if(s.dashed) ctx.setLineDash([7,5]);

    ctx.beginPath();

    s.values.forEach((v,i)=>{
      const X=sx(xvals[i]);
      const Y=sy(v);

      if(i===0) ctx.moveTo(X,Y);
      else ctx.lineTo(X,Y);
    });

    ctx.stroke();
    ctx.restore();
  });

  if(opts.legend){
    let x=65,y=17;
    ctx.font="12px Arial";

    opts.legend.forEach((item,idx)=>{
      ctx.strokeStyle=item.color||colors[idx%colors.length];
      ctx.lineWidth=3;
      ctx.beginPath();
      ctx.moveTo(x,y);
      ctx.lineTo(x+20,y);
      ctx.stroke();

      ctx.fillStyle="#556370";
      ctx.fillText(item.label,x+25,y+4);

      x += Math.max(85, item.label.length*7+45);
    });
  }
}

function drawTask(){
  const T=CFG.steps;
  const time=zeros(T);
  const inp=zeros(T);
  const tar=zeros(T);

  for(let i=0;i<T;i++){
    time[i]=i*CFG.dt;
    inp[i]=inputVectorAt(time[i])[0];
    tar[i]=targetVectorAt(time[i])[0];
  }

  plotSeries(
    $("taskCanvas"),
    [
      {values:inp,color:"#7a8792",width:2},
      {values:tar,color:"#222",dashed:true,width:2}
    ],
    time,
    {
      legend:[
        {label:"x1 input",color:"#7a8792"},
        {label:"y1 target",color:"#222"}
      ]
    }
  );
}

function drawOutput(){
  const F=S.lastForward;
  const series=[];
  const legend=[];
  const colors=["#222","#2b6fbb","#c46a27","#5c8b57","#8c5baa"];

  for(let o=0;o<CFG.outputSize;o++){
    series.push({
      values:F.y.map(v=>v[o]),
      color:colors[o%colors.length],
      width:2.5
    });

    series.push({
      values:F.target.map(v=>v[o]),
      color:colors[o%colors.length],
      dashed:true,
      width:1.5
    });

    legend.push({
      label:`y${o+1}`,
      color:colors[o%colors.length]
    });
  }

  series.push({
    values:F.x.map(v=>v[0]),
    color:"#b3bac1",
    dashed:true,
    width:1.2
  });

  legend.push({
    label:"x1",
    color:"#b3bac1"
  });

  plotSeries(
    $("outputCanvas"),
    series,
    F.time,
    {legend}
  );
}

function drawLoss(){
  const canvas=$("lossCanvas");
  const ctx=clearCanvas(canvas);

  if(S.lossHistory.length<2){
    ctx.fillStyle="#74818c";
    ctx.font="14px Arial";
    ctx.fillText(
      "Train for at least two epochs to see the loss curve.",
      35,45
    );
    return;
  }

  const x=S.lossHistory.map((_,i)=>i+1);

  plotSeries(
    canvas,
    [{values:S.lossHistory,color:"#222",width:2}],
    x,
    {xLabel:"epoch",yLabel:"MSE"}
  );
}

function drawHidden(){
  const F=S.lastForward;
  const series=[];
  const colors=[
    "#2b6fbb","#c46a27","#5c8b57","#8c5baa",
    "#b04c55","#2a8f8f","#7d6c44","#4b6b88"
  ];

  for(let j=0;j<CFG.hiddenSize;j++){
    series.push({
      values:F.h.slice(1).map(row=>row[j]),
      color:colors[j%colors.length],
      width:2
    });
  }

  plotSeries(
    $("hiddenCanvas"),
    series,
    F.time,
    {
      legend:colors
        .slice(0,CFG.hiddenSize)
        .map((c,i)=>({label:"h"+(i+1),color:c}))
    }
  );
}

/* ---------- PCA ---------- */

function dot(a,b){
  let s=0;
  for(let i=0;i<a.length;i++) s += a[i]*b[i];
  return s;
}

function norm(a){
  return Math.sqrt(dot(a,a));
}

function matVec(A,v){
  return A.map(row=>dot(row,v));
}

function normalize(v){
  const n=norm(v)||1;
  return v.map(x=>x/n);
}

function powerIteration(C, orthogonalTo=null){
  let v=normalize(
    Array.from({length:C.length},(_,i)=>1+0.2*i)
  );

  for(let it=0;it<80;it++){
    let w=matVec(C,v);

    if(orthogonalTo){
      const p=dot(w,orthogonalTo);
      w=w.map((x,i)=>x-p*orthogonalTo[i]);
    }

    v=normalize(w);
  }

  return v;
}

function pca2(points){
  const n=points.length;
  const d=points[0].length;

  const mean=zeros(d);

  points.forEach(p=>{
    p.forEach((v,j)=>{
      mean[j]+=v/n;
    });
  });

  const centered=points.map(
    p=>p.map((v,j)=>v-mean[j])
  );

  const C=matZeros(d,d);

  centered.forEach(p=>{
    for(let i=0;i<d;i++){
      for(let j=0;j<d;j++){
        C[i][j]+=p[i]*p[j]/Math.max(1,n-1);
      }
    }
  });

  const pc1=powerIteration(C);
  const pc2=powerIteration(C,pc1);

  return centered.map(
    p=>[dot(p,pc1),dot(p,pc2)]
  );
}

function drawPCA(){
  const canvas=$("pcaCanvas");
  const ctx=clearCanvas(canvas);
  const w=canvas.width;
  const h=canvas.height;

  if(CFG.hiddenSize < 2){
    ctx.fillStyle="#74818c";
    ctx.font="14px Arial";
    ctx.fillText("PCA requires at least two hidden units.",35,45);
    return;
  }

  const pts=pca2(S.lastForward.h);
  const xs=pts.map(p=>p[0]);
  const ys=pts.map(p=>p[1]);

  const [xmin,xmax]=bounds(xs);
  const [ymin,ymax]=bounds(ys);

  const A=drawAxes(ctx,w,h,xmin,xmax,ymin,ymax,"PC1","PC2");

  const sx=x=>A.L+(x-xmin)/(xmax-xmin)*(w-A.L-A.R);
  const sy=y=>A.T+(ymax-y)/(ymax-ymin)*(h-A.T-A.B);

  ctx.strokeStyle="#315d8f";
  ctx.lineWidth=2;
  ctx.beginPath();

  pts.forEach((p,i)=>{
    const X=sx(p[0]);
    const Y=sy(p[1]);

    if(i===0) ctx.moveTo(X,Y);
    else ctx.lineTo(X,Y);
  });

  ctx.stroke();

  const p0=pts[0];
  const p1=pts[pts.length-1];

  ctx.fillStyle="#5a9f68";
  ctx.beginPath();
  ctx.arc(sx(p0[0]),sy(p0[1]),6,0,2*Math.PI);
  ctx.fill();

  ctx.fillStyle="#222";
  ctx.beginPath();
  ctx.arc(sx(p1[0]),sy(p1[1]),6,0,2*Math.PI);
  ctx.fill();
}

/* ---------- Network visualization ---------- */

function edgeStyle(v){
  return {
    stroke: v>=0 ? "#2f72b7" : "#cf792d",
    width: 0.7 + 3.4*Math.min(1,Math.abs(v)/2.0),
    opacity: 0.35 + 0.55*Math.min(1,Math.abs(v)/1.5)
  };
}

function svgEl(tag,attrs={}){
  const el=document.createElementNS(
    "http://www.w3.org/2000/svg",
    tag
  );

  for(const [k,v] of Object.entries(attrs)){
    el.setAttribute(k,v);
  }

  return el;
}

function spreadY(n, top=70, bottom=365){
  if(n===1) return [215];
  return Array.from(
    {length:n},
    (_,i)=>top+(bottom-top)*i/(n-1)
  );
}

function drawNetwork(){
  const svg=$("networkSvg");
  while(svg.firstChild) svg.removeChild(svg.firstChild);

  const inputYs=spreadY(CFG.inputSize,85,345);
  const hiddenYs=spreadY(CFG.hiddenSize,65,365);
  const outputYs=spreadY(CFG.outputSize,110,320);

  const inP=inputYs.map(y=>({x:90,y}));
  const hP=hiddenYs.map(y=>({x:440,y}));
  const outP=outputYs.map(y=>({x:810,y}));

  function line(a,b,w){
    const s=edgeStyle(w);

    svg.appendChild(
      svgEl("line",{
        x1:a.x,y1:a.y,
        x2:b.x,y2:b.y,
        stroke:s.stroke,
        "stroke-width":s.width,
        opacity:s.opacity,
        class:"edge"
      })
    );
  }

  for(let i=0;i<CFG.hiddenSize;i++){
    for(let k=0;k<CFG.inputSize;k++){
      line(inP[k],hP[i],S.Win[i][k]);
    }
  }

  for(let i=0;i<CFG.hiddenSize;i++){
    for(let j=0;j<CFG.hiddenSize;j++){
      const w=S.Wrec[i][j];
      const s=edgeStyle(w);

      if(i===j){
        const p=hP[i];

        svg.appendChild(
          svgEl("path",{
            d:`M ${p.x+18} ${p.y-8}
               C ${p.x+70} ${p.y-48},
                 ${p.x+70} ${p.y+48},
                 ${p.x+18} ${p.y+8}`,
            fill:"none",
            stroke:s.stroke,
            "stroke-width":s.width,
            opacity:s.opacity,
            class:"edge"
          })
        );
      }else{
        line(hP[j],hP[i],w);
      }
    }
  }

  for(let o=0;o<CFG.outputSize;o++){
    for(let j=0;j<CFG.hiddenSize;j++){
      line(hP[j],outP[o],S.Wout[o][j]);
    }
  }

  inP.forEach((p,k)=>{
    svg.appendChild(
      svgEl("circle",{
        cx:p.x,cy:p.y,r:21,
        fill:"#eef3f7",
        stroke:"#607080",
        "stroke-width":2
      })
    );

    const t=svgEl("text",{
      x:p.x-9,y:p.y+5,
      class:"node-label"
    });

    t.textContent="x"+(k+1);
    svg.appendChild(t);
  });

  hP.forEach((p,i)=>{
    svg.appendChild(
      svgEl("circle",{
        cx:p.x,cy:p.y,r:21,
        fill:"#f7f9fb",
        stroke:"#4f6270",
        "stroke-width":2
      })
    );

    const t=svgEl("text",{
      x:p.x-10,y:p.y+5,
      class:"node-label"
    });

    t.textContent="h"+(i+1);
    svg.appendChild(t);
  });

  outP.forEach((p,o)=>{
    svg.appendChild(
      svgEl("circle",{
        cx:p.x,cy:p.y,r:21,
        fill:"#eef3f7",
        stroke:"#607080",
        "stroke-width":2
      })
    );

    const t=svgEl("text",{
      x:p.x-9,y:p.y+5,
      class:"node-label"
    });

    t.textContent="y"+(o+1);
    svg.appendChild(t);
  });

  [
    ["external inputs",35,35],
    ["recurrent hidden population",345,35],
    ["linear outputs",755,35]
  ].forEach(([txt,x,y])=>{
    const t=svgEl("text",{
      x,y,
      fill:"#647380",
      "font-size":"13"
    });

    t.textContent=txt;
    svg.appendChild(t);
  });
}


function heatColor(v, maxAbs){
  const a = Math.min(1, Math.abs(v)/(maxAbs || 1));
  if(Math.abs(v) < 1e-8) return "rgba(238,242,245,0.95)";
  if(v >= 0) return `rgba(47,114,183,${0.10 + 0.58*a})`;
  return `rgba(207,121,45,${0.10 + 0.58*a})`;
}

function matrixToHTML(A, rowPrefix, colPrefix){
  let maxAbs = 0;
  for(const row of A){
    for(const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));
  }

  let html = '<table class="matrix-table"><thead><tr><th></th>';
  for(let j=0;j<A[0].length;j++){
    html += `<th>${colPrefix}${j+1}</th>`;
  }
  html += '</tr></thead><tbody>';

  for(let i=0;i<A.length;i++){
    html += `<tr><td class="row-label">${rowPrefix}${i+1}</td>`;
    for(let j=0;j<A[i].length;j++){
      const v = A[i][j];
      html += `<td title="${v.toFixed(6)}" style="background:${heatColor(v,maxAbs)}">${v.toFixed(3)}</td>`;
    }
    html += '</tr>';
  }

  html += '</tbody></table>';
  return html;
}

function vectorToHTML(v, prefix){
  const A = v.map(x => [x]);
  return matrixToHTML(A, prefix, "value");
}

function renderWeightMatrices(){
  if(!S.Win || !S.Wrec || !S.Wout) return;

  $("winShape").textContent = `${CFG.hiddenSize} × ${CFG.inputSize}`;
  $("wrecShape").textContent = `${CFG.hiddenSize} × ${CFG.hiddenSize}`;
  $("woutShape").textContent = `${CFG.outputSize} × ${CFG.hiddenSize}`;

  $("winMatrix").innerHTML = matrixToHTML(S.Win, "h", "x");
  $("wrecMatrix").innerHTML = matrixToHTML(S.Wrec, "h", "h");
  $("woutMatrix").innerHTML = matrixToHTML(S.Wout, "y", "h");

  let biasHTML = '<div style="margin-bottom:6px;font-size:9px;color:#71808d">hidden b</div>';
  biasHTML += vectorToHTML(S.b, "h");
  biasHTML += '<div style="margin:7px 0 4px;font-size:9px;color:#71808d">output b</div>';
  biasHTML += vectorToHTML(S.bout, "y");
  $("biasMatrix").innerHTML = biasHTML;
}

function updateMetrics(){
  const limit=Math.max(
    1,
    parseInt($("epochLimitInput").value)||1500
  );

  $("epochText").textContent=`${S.epoch} / ${limit}`;

  $("lossText").textContent=
    Number.isFinite(S.lastLoss)
      ? S.lastLoss.toFixed(6)
      : "—";

  $("gradText").textContent=
    Number.isFinite(S.lastGradNorm)
      ? S.lastGradNorm.toFixed(5)
      : "—";
}

function updateAll(){
  updateMetrics();
  drawOutput();
  drawLoss();
  drawHidden();
  drawPCA();
  drawNetwork();
  renderWeightMatrices();
}

$("trainBtn").addEventListener("click",startTraining);
$("pauseBtn").addEventListener("click",pauseTraining);
$("stepBtn").addEventListener("click",stepEpoch);
$("resetBtn").addEventListener("click",resetWeights);
$("applyArchitectureBtn").addEventListener("click",applyArchitecture);
$("refreshWeightsBtn").addEventListener("click",renderWeightMatrices);
$("epochLimitInput").addEventListener("input",updateMetrics);

drawTask();
resetWeights();

})();
