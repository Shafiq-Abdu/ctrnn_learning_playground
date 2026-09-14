const app={
  running:false,time:0,duration:10,
  inputCount:2,hiddenCount:4,outputCount:1,
  tau:1.0,dt:0.02,activationName:"tanh",
  inputSpecs:[],h0:[],h:[],
  Win:[],Wrec:[],Wout:[],
  history:[],selectedWeight:null,
  trajectoryMode:"raw",
  pcaProjection:null,pcaReady:false,
  fixedPoint:null
};

const $=id=>document.getElementById(id);
const rand=(s=1)=>(Math.random()*2-1)*s;

function defaultInput(i){
  return {type:i===0?"pulse":"constant",amplitude:i===0?1:0.5,start:1,end:3,frequency:0.5};
}

function initializeInputs(){
  const old=app.inputSpecs;
  app.inputSpecs=Array.from({length:app.inputCount},(_,i)=>old[i]?{...old[i]}:defaultInput(i));
}

function initializeHidden(){
  const old=app.h0;
  app.h0=Array.from({length:app.hiddenCount},(_,i)=>old[i]!==undefined?old[i]:0);
}

function resizeMatrix(oldM,r,c,scale,diag=false){
  return Array.from({length:r},(_,i)=>Array.from({length:c},(_,j)=>{
    if(oldM[i] && oldM[i][j]!==undefined) return oldM[i][j];
    if(diag && i===j) return 0.6+rand(0.3);
    return rand(scale);
  }));
}

function initializeWeights(){
  app.Win=resizeMatrix(app.Win,app.hiddenCount,app.inputCount,1,false);
  app.Wrec=resizeMatrix(app.Wrec,app.hiddenCount,app.hiddenCount,0.8,true);
  app.Wout=resizeMatrix(app.Wout,app.outputCount,app.hiddenCount,1,false);
}

function initializeArchitecture(){
  initializeInputs();
  initializeHidden();
  initializeWeights();
  renderInputControls();
  renderHiddenInitialControls();
  drawNetwork();
  resetSimulation();
}

function resetSimulation(){
  app.running=false;
  app.time=0;
  app.h=[...app.h0];
  app.history=[];
  app.pcaProjection=null;
  app.pcaReady=false;
  app.fixedPoint=null;
  $("fixedPointText").textContent="Not analyzed yet.";
  $("stabilityText").textContent="Not analyzed yet.";
  $("eigenText").textContent="Not analyzed yet.";
  updateEverything();
}

function activation(x){
  if(app.activationName==="tanh") return Math.tanh(x);
  if(app.activationName==="relu") return Math.max(0,x);
  if(app.activationName==="sigmoid") return 1/(1+Math.exp(-x));
  return x;
}

function activationDerivative(x){
  if(app.activationName==="tanh"){ const t=Math.tanh(x); return 1-t*t; }
  if(app.activationName==="relu") return x>0?1:0;
  if(app.activationName==="sigmoid"){ const s=1/(1+Math.exp(-x)); return s*(1-s); }
  return 1;
}

function inputValue(spec,t){
  const A=spec.amplitude;
  if(spec.type==="constant") return A;
  if(spec.type==="step") return t>=spec.start?A:0;
  if(spec.type==="pulse") return (t>=spec.start && t<=spec.end)?A:0;
  if(spec.type==="sine"){
    if(t<spec.start || t>spec.end) return 0;
    return A*Math.sin(2*Math.PI*spec.frequency*(t-spec.start));
  }
  return 0;
}

function inputVector(t){ return app.inputSpecs.map(s=>inputValue(s,t)); }

function outputVector(){
  return app.Wout.map(row=>row.reduce((sum,w,j)=>sum+w*app.h[j],0));
}

function stepCTRNN(){
  const x=inputVector(app.time);
  const next=[...app.h];

  for(let i=0;i<app.hiddenCount;i++){
    let rec=0,ext=0;
    for(let j=0;j<app.hiddenCount;j++) rec+=app.Wrec[i][j]*app.h[j];
    for(let k=0;k<app.inputCount;k++) ext+=app.Win[i][k]*x[k];

    const dh=(-app.h[i]+activation(rec+ext))/app.tau;
    next[i]=app.h[i]+app.dt*dh;
  }

  app.h=next;
  app.time+=app.dt;
  if(app.time>app.duration) app.time=app.duration;

  app.history.push({time:app.time,h:[...app.h],x:[...x],y:outputVector()});

  if(app.time>=app.duration){
    app.running=false;
    computePCAFromHistory();
  }
}

function renderInputControls(){
  const c=$("inputControls");
  c.innerHTML="";

  app.inputSpecs.forEach((s,i)=>{
    const row=document.createElement("div");
    row.className="input-row";
    row.innerHTML=`
      <strong>x${i+1}(t)</strong>
      <div class="input-grid">
        <label class="wide">Type
          <select data-index="${i}" data-type="type">
            <option value="constant" ${s.type==="constant"?"selected":""}>Constant</option>
            <option value="step" ${s.type==="step"?"selected":""}>Step</option>
            <option value="pulse" ${s.type==="pulse"?"selected":""}>Pulse</option>
            <option value="sine" ${s.type==="sine"?"selected":""}>Sine</option>
          </select>
        </label>
        <label>Amplitude<input type="number" step="0.1" data-index="${i}" data-type="amplitude" value="${s.amplitude}"></label>
        <label>Frequency<input type="number" step="0.1" min="0" data-index="${i}" data-type="frequency" value="${s.frequency}"></label>
        <label>Start<input type="number" step="0.1" min="0" data-index="${i}" data-type="start" value="${s.start}"></label>
        <label>End<input type="number" step="0.1" min="0" data-index="${i}" data-type="end" value="${s.end}"></label>
      </div>`;
    c.appendChild(row);
  });

  c.querySelectorAll("input,select").forEach(ctrl=>{
    ctrl.addEventListener("change",e=>{
      const i=Number(e.target.dataset.index),type=e.target.dataset.type;
      if(type==="type"){ app.inputSpecs[i].type=e.target.value; return; }
      const v=Number(e.target.value);
      if(!Number.isFinite(v)) return;
      app.inputSpecs[i][type]=v;
      if(app.inputSpecs[i].end<app.inputSpecs[i].start){
        app.inputSpecs[i].end=app.inputSpecs[i].start;
        renderInputControls();
      }
    });
  });
}

function renderHiddenInitialControls(){
  const c=$("hiddenInitialControls");
  c.innerHTML="";

  app.h0.forEach((v,i)=>{
    const label=document.createElement("label");
    label.innerHTML=`h${i+1}(0)<input type="number" step="0.1" value="${v}" data-index="${i}">`;
    c.appendChild(label);
  });

  c.querySelectorAll("input").forEach(inp=>{
    inp.addEventListener("change",e=>{
      const i=Number(e.target.dataset.index),v=Number(e.target.value);
      if(Number.isFinite(v)) app.h0[i]=v;
    });
  });
}

function spacedPositions(n,height){
  if(n===1) return [height/2];
  const top=55,bottom=height-55;
  return Array.from({length:n},(_,i)=>top+i*(bottom-top)/(n-1));
}

function edgeStyle(w){
  return {color:w>=0?"#2b7dbd":"#d07a35",width:Math.min(6,1+Math.abs(w)*2.2),opacity:0.25+Math.min(0.65,Math.abs(w)/3)};
}

function drawNetwork(){
  const svg=$("networkSvg"); svg.innerHTML="";
  const height=400,inputX=90,hiddenX=380,outputX=670;
  const inputY=spacedPositions(app.inputCount,height);
  const hiddenY=spacedPositions(app.hiddenCount,height);
  const outputY=spacedPositions(app.outputCount,height);

  for(let i=0;i<app.hiddenCount;i++){
    for(let j=0;j<app.inputCount;j++)
      drawLine(svg,inputX+26,inputY[j],hiddenX-26,hiddenY[i],app.Win[i][j],{type:"input",i,j});
  }

  for(let i=0;i<app.hiddenCount;i++){
    for(let j=0;j<app.hiddenCount;j++){
      if(i===j) drawLoop(svg,hiddenX,hiddenY[i],app.Wrec[i][j],{type:"recurrent",i,j});
      else drawCurve(svg,hiddenX,hiddenY[j],hiddenX,hiddenY[i],app.Wrec[i][j],{type:"recurrent",i,j});
    }
  }

  for(let i=0;i<app.outputCount;i++){
    for(let j=0;j<app.hiddenCount;j++)
      drawLine(svg,hiddenX+26,hiddenY[j],outputX-26,outputY[i],app.Wout[i][j],{type:"output",i,j});
  }

  inputY.forEach((y,i)=>drawNode(svg,inputX,y,`x${i+1}`));
  hiddenY.forEach((y,i)=>drawNode(svg,hiddenX,y,`h${i+1}`));
  outputY.forEach((y,i)=>drawNode(svg,outputX,y,`y${i+1}`));

  $("hiddenLabel").textContent=`${app.hiddenCount} hidden units`;
  $("hiddenCountText").textContent=app.hiddenCount;
}

function drawNode(svg,x,y,text){
  const NS="http://www.w3.org/2000/svg";
  const c=document.createElementNS(NS,"circle");
  c.setAttribute("cx",x); c.setAttribute("cy",y); c.setAttribute("r",26);
  c.setAttribute("fill","#fff"); c.setAttribute("stroke","#34495e"); c.setAttribute("stroke-width",2);
  const t=document.createElementNS(NS,"text");
  t.setAttribute("x",x); t.setAttribute("y",y+5); t.setAttribute("text-anchor","middle"); t.setAttribute("class","node-label");
  t.textContent=text;
  svg.appendChild(c); svg.appendChild(t);
}

function drawLine(svg,x1,y1,x2,y2,w,meta){
  const NS="http://www.w3.org/2000/svg",line=document.createElementNS(NS,"line"),s=edgeStyle(w);
  line.setAttribute("x1",x1); line.setAttribute("y1",y1); line.setAttribute("x2",x2); line.setAttribute("y2",y2);
  line.setAttribute("stroke",s.color); line.setAttribute("stroke-width",s.width); line.setAttribute("opacity",s.opacity); line.setAttribute("class","edge");
  line.addEventListener("click",()=>selectWeight(meta)); svg.appendChild(line);
}

function drawCurve(svg,x1,y1,x2,y2,w,meta){
  const NS="http://www.w3.org/2000/svg",p=document.createElementNS(NS,"path"),s=edgeStyle(w);
  const dir=y2>y1?1:-1,cx=x1+dir*85,cy=(y1+y2)/2;
  p.setAttribute("d",`M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`);
  p.setAttribute("fill","none"); p.setAttribute("stroke",s.color); p.setAttribute("stroke-width",s.width); p.setAttribute("opacity",s.opacity); p.setAttribute("class","edge");
  p.addEventListener("click",()=>selectWeight(meta)); svg.appendChild(p);
}

function drawLoop(svg,x,y,w,meta){
  const NS="http://www.w3.org/2000/svg",p=document.createElementNS(NS,"path"),s=edgeStyle(w);
  p.setAttribute("d",`M ${x-18} ${y-20} C ${x-65} ${y-75}, ${x+65} ${y-75}, ${x+18} ${y-20}`);
  p.setAttribute("fill","none"); p.setAttribute("stroke",s.color); p.setAttribute("stroke-width",s.width); p.setAttribute("opacity",s.opacity); p.setAttribute("class","edge");
  p.addEventListener("click",()=>selectWeight(meta)); svg.appendChild(p);
}

function selectWeight(meta){
  app.selectedWeight=meta; $("weightSlider").disabled=false;
  let value,desc,typeTitle;
  if(meta.type==="input"){ value=app.Win[meta.i][meta.j]; desc=`x${meta.j+1} → h${meta.i+1}`; typeTitle="Input connection"; }
  else if(meta.type==="recurrent"){ value=app.Wrec[meta.i][meta.j]; desc=`h${meta.j+1} → h${meta.i+1}`; typeTitle="Recurrent connection"; }
  else{ value=app.Wout[meta.i][meta.j]; desc=`h${meta.j+1} → y${meta.i+1}`; typeTitle="Readout connection"; }
  $("connectionType").textContent=typeTitle;
  $("weightDescription").textContent=desc;

  if(meta.type==="input"){
    $("connectionMeaning").textContent=
      "This weight controls how strongly external input x"+(meta.j+1)+" influences hidden unit h"+(meta.i+1)+".";
  }else if(meta.type==="recurrent"){
    $("connectionMeaning").textContent=
      "This recurrent weight controls how strongly hidden unit h"+(meta.j+1)+" influences the future dynamics of h"+(meta.i+1)+".";
  }else{
    $("connectionMeaning").textContent=
      "This readout weight controls how strongly hidden unit h"+(meta.j+1)+" contributes to output y"+(meta.i+1)+".";
  }
  $("weightSlider").value=value;
  $("weightValue").textContent=value.toFixed(2);
}

function updateSelectedWeight(v){
  const m=app.selectedWeight; if(!m) return;
  if(m.type==="input") app.Win[m.i][m.j]=v;
  else if(m.type==="recurrent") app.Wrec[m.i][m.j]=v;
  else app.Wout[m.i][m.j]=v;
  drawNetwork(); selectWeight(m);
}

function updateOutputs(){
  const c=$("outputValues"); c.innerHTML="";
  outputVector().forEach((v,i)=>{
    const b=document.createElement("div"); b.className="output-box"; b.textContent=`y${i+1} = ${v.toFixed(3)}`; c.appendChild(b);
  });
}

function drawActivityPlot(){
  const canvas=$("activityCanvas"),ctx=canvas.getContext("2d"),W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);

  if(app.history.length<2){ drawTimeAxes(ctx,W,H,-1,1); return; }

  let min=Infinity,max=-Infinity;
  app.history.forEach(p=>{ p.h.forEach(v=>{min=Math.min(min,v);max=Math.max(max,v)}); p.x.forEach(v=>{min=Math.min(min,v);max=Math.max(max,v)}); });
  if(max===min){max+=1;min-=1}
  const pad=0.15*(max-min); max+=pad; min-=pad;
  const maxTime=Math.max(app.duration,app.time,0.1);
  const mapX=t=>55+(t/maxTime)*(W-80);
  const mapY=v=>H-40-((v-min)/(max-min))*(H-75);

  drawTimeAxes(ctx,W,H,min,max,mapX,mapY,maxTime);

  const colors=["#2b7dbd","#d07a35","#5a9f68","#8f63b8","#bd4f6c","#4e8c8c","#9a7b4f","#666"];
  for(let n=0;n<app.hiddenCount;n++){
    ctx.beginPath(); ctx.strokeStyle=colors[n%colors.length]; ctx.lineWidth=2;
    app.history.forEach((p,i)=>{ const x=mapX(p.time),y=mapY(p.h[n]); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.stroke();
  }

  for(let k=0;k<app.inputCount;k++){
    ctx.beginPath(); ctx.strokeStyle="#999"; ctx.lineWidth=1; ctx.setLineDash([6,5]);
    app.history.forEach((p,i)=>{ const x=mapX(p.time),y=mapY(p.x[k]); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.stroke(); ctx.setLineDash([]);
  }
}

function drawTimeAxes(ctx,W,H,min,max,mapX=null,mapY=null,maxTime=null){
  ctx.strokeStyle="#cfd8df"; ctx.fillStyle="#52616f"; ctx.lineWidth=1; ctx.font="12px Arial";
  ctx.beginPath(); ctx.moveTo(55,H-40); ctx.lineTo(W-25,H-40); ctx.moveTo(55,20); ctx.lineTo(55,H-40); ctx.stroke();

  if(!mapX||!mapY||maxTime===null){ ctx.fillText("time (s)",W-62,H-12); ctx.fillText("activity",8,18); return; }

  for(let i=0;i<=6;i++){
    const t=maxTime*i/6,x=mapX(t);
    ctx.beginPath();ctx.moveTo(x,H-40);ctx.lineTo(x,H-35);ctx.stroke();
    ctx.fillText(t.toFixed(1),x-10,H-18);
  }

  for(let i=0;i<=5;i++){
    const v=min+(max-min)*i/5,y=mapY(v);
    ctx.beginPath();ctx.moveTo(50,y);ctx.lineTo(55,y);ctx.stroke();
    ctx.fillText(v.toFixed(2),6,y+4);
  }

  if(min<=0&&max>=0){ctx.strokeStyle="#e0e5e9";ctx.beginPath();ctx.moveTo(55,mapY(0));ctx.lineTo(W-25,mapY(0));ctx.stroke();}
  ctx.fillStyle="#52616f";ctx.fillText("time (s)",W-62,H-12);ctx.fillText("activity",8,18);
}

function computePCAFromHistory(){
  if(app.history.length<5||app.hiddenCount<2){app.pcaReady=false;app.pcaProjection=null;return;}
  app.pcaProjection=computePCA2D(app.history.map(p=>p.h));
  app.pcaReady=true;
}

function computePCA2D(data){
  const n=data.length,d=data[0].length,mean=Array(d).fill(0);
  data.forEach(r=>r.forEach((v,j)=>mean[j]+=v/n));
  const X=data.map(r=>r.map((v,j)=>v-mean[j]));
  const C=Array.from({length:d},()=>Array(d).fill(0));
  for(let i=0;i<d;i++) for(let j=0;j<d;j++){
    let s=0; for(let r=0;r<n;r++) s+=X[r][i]*X[r][j];
    C[i][j]=s/Math.max(1,n-1);
  }
  const pc1=powerIteration(C),lam1=eigenvalue(C,pc1);
  const C2=C.map((row,i)=>row.map((v,j)=>v-lam1*pc1[i]*pc1[j]));
  const pc2=powerIteration(C2);
  return X.map(r=>[dot(r,pc1),dot(r,pc2)]);
}

function powerIteration(M){
  let v=normalize(Array.from({length:M.length},(_,i)=>i+1));
  for(let it=0;it<100;it++) v=normalize(M.map(row=>dot(row,v)));
  return v;
}
function normalize(v){ const n=Math.sqrt(dot(v,v)); return n<1e-12?v:v.map(x=>x/n); }
function dot(a,b){ let s=0; for(let i=0;i<Math.min(a.length,b.length);i++) s+=a[i]*b[i]; return s; }
function eigenvalue(M,v){ return dot(v,M.map(r=>dot(r,v))); }

function drawTrajectory(){
  if(app.trajectoryMode==="pca") drawPCATrajectory();
  else drawRawTrajectory();
}

function drawRawTrajectory(){
  $("trajectoryDescription").textContent="Raw population view: h1 versus h2";
  $("pcaStatus").textContent="Raw view updates live.";
  drawTrajectoryPoints(app.history.map(p=>[p.h[0],p.h[1]]),"h1","h2",false);
}

function drawPCATrajectory(){
  $("trajectoryDescription").textContent="Population PCA: PC1 versus PC2";
  if(!app.pcaReady){
    $("pcaStatus").textContent=app.running?"PCA will be computed when you Pause or when the run ends.":"Run and Pause, or let the run finish, to compute PCA.";
    drawTrajectoryPoints([],"PC1","PC2",false);
    return;
  }
  $("pcaStatus").textContent="PCA is computed from the recorded hidden-state trajectory.";
  drawTrajectoryPoints(app.pcaProjection,"PC1","PC2",true);
}

function drawTrajectoryPoints(points,xLabel,yLabel,isPCA){
  const canvas=$("trajectoryCanvas"),ctx=canvas.getContext("2d"),W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);

  if(points.length<2){ drawBasicAxes(ctx,W,H,xLabel,yLabel); return; }

  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  if(xmax===xmin){xmax+=1;xmin-=1}
  if(ymax===ymin){ymax+=1;ymin-=1}
  const xp=(xmax-xmin)*0.15,yp=(ymax-ymin)*0.15; xmin-=xp;xmax+=xp;ymin-=yp;ymax+=yp;
  const mapX=v=>55+((v-xmin)/(xmax-xmin))*(W-85);
  const mapY=v=>H-40-((v-ymin)/(ymax-ymin))*(H-75);

  ctx.strokeStyle="#d8dfe5";ctx.lineWidth=1;
  if(xmin<=0&&xmax>=0){ctx.beginPath();ctx.moveTo(mapX(0),20);ctx.lineTo(mapX(0),H-40);ctx.stroke();}
  if(ymin<=0&&ymax>=0){ctx.beginPath();ctx.moveTo(55,mapY(0));ctx.lineTo(W-30,mapY(0));ctx.stroke();}

  ctx.beginPath();ctx.strokeStyle="#34495e";ctx.lineWidth=2;
  points.forEach((p,i)=>{const x=mapX(p[0]),y=mapY(p[1]); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}); ctx.stroke();

  const s=points[0],e=points[points.length-1];
  ctx.fillStyle="#5a9f68";ctx.beginPath();ctx.arc(mapX(s[0]),mapY(s[1]),6,0,2*Math.PI);ctx.fill();
  ctx.fillStyle="#1f2933";ctx.beginPath();ctx.arc(mapX(e[0]),mapY(e[1]),7,0,2*Math.PI);ctx.fill();

  // Fixed point marker only for raw h1-h2 view.
  if(!isPCA && app.fixedPoint && app.fixedPoint.length>=2){
    const fx=app.fixedPoint[0],fy=app.fixedPoint[1];
    if(fx>=xmin&&fx<=xmax&&fy>=ymin&&fy<=ymax){
      ctx.strokeStyle="#b23a48";ctx.lineWidth=2;ctx.beginPath();
      ctx.moveTo(mapX(fx)-7,mapY(fy));ctx.lineTo(mapX(fx)+7,mapY(fy));
      ctx.moveTo(mapX(fx),mapY(fy)-7);ctx.lineTo(mapX(fx),mapY(fy)+7);
      ctx.stroke();
    }
  }

  ctx.fillStyle="#52616f";ctx.fillText(xLabel,W-50,H-12);ctx.fillText(yLabel,12,20);
}

function drawBasicAxes(ctx,W,H,xLabel,yLabel){
  ctx.strokeStyle="#d8dfe5";ctx.beginPath();ctx.moveTo(55,H-40);ctx.lineTo(W-30,H-40);ctx.moveTo(55,20);ctx.lineTo(55,H-40);ctx.stroke();
  ctx.fillStyle="#52616f";ctx.fillText(xLabel,W-50,H-12);ctx.fillText(yLabel,12,20);
}

// ---------- Fixed point + stability ----------
function analysisInputVector(){
  if($("analysisInputMode").value==="current") return inputVector(app.time);
  return Array(app.inputCount).fill(0);
}

function vectorField(h,x){
  const out=Array(app.hiddenCount).fill(0);
  for(let i=0;i<app.hiddenCount;i++){
    let rec=0,ext=0;
    for(let j=0;j<app.hiddenCount;j++) rec+=app.Wrec[i][j]*h[j];
    for(let k=0;k<app.inputCount;k++) ext+=app.Win[i][k]*x[k];
    out[i]=(-h[i]+activation(rec+ext))/app.tau;
  }
  return out;
}

function findFixedPoint(initial,x){
  let h=[...initial];
  const step=0.05;
  for(let iter=0;iter<20000;iter++){
    const f=vectorField(h,x);
    const norm=Math.sqrt(dot(f,f));
    if(norm<1e-8) return {point:h,converged:true,residual:norm};
    h=h.map((v,i)=>v+step*f[i]);
    if(h.some(v=>!Number.isFinite(v)||Math.abs(v)>1e6)) break;
  }
  const f=vectorField(h,x);
  return {point:h,converged:false,residual:Math.sqrt(dot(f,f))};
}

function jacobianAt(h,x){
  const n=app.hiddenCount;
  const J=Array.from({length:n},()=>Array(n).fill(0));
  const z=Array(n).fill(0);

  for(let i=0;i<n;i++){
    let total=0;
    for(let j=0;j<n;j++) total+=app.Wrec[i][j]*h[j];
    for(let k=0;k<app.inputCount;k++) total+=app.Win[i][k]*x[k];
    z[i]=total;
  }

  for(let i=0;i<n;i++){
    const fp=activationDerivative(z[i]);
    for(let j=0;j<n;j++){
      J[i][j]=((-1)*(i===j?1:0)+fp*app.Wrec[i][j])/app.tau;
    }
  }
  return J;
}

// QR iteration for small real matrices.
function eigenvaluesQR(A){
  let M=A.map(r=>[...r]);
  const n=M.length;

  function qrDecompose(B){
    const m=B.length;
    const Q=Array.from({length:m},()=>Array(m).fill(0));
    const R=Array.from({length:m},()=>Array(m).fill(0));
    const cols=Array.from({length:m},(_,j)=>B.map(r=>r[j]));

    const qcols=[];
    for(let j=0;j<m;j++){
      let v=[...cols[j]];
      for(let k=0;k<j;k++){
        const r=dot(qcols[k],v);
        R[k][j]=r;
        v=v.map((x,i)=>x-r*qcols[k][i]);
      }
      const norm=Math.sqrt(dot(v,v));
      if(norm<1e-12){
        const e=Array(m).fill(0); e[j]=1;
        qcols.push(e);
        R[j][j]=0;
      }else{
        const q=v.map(x=>x/norm);
        qcols.push(q);
        R[j][j]=norm;
      }
    }
    for(let i=0;i<m;i++) for(let j=0;j<m;j++) Q[i][j]=qcols[j][i];
    return {Q,R};
  }

  function matMul(A,B){
    const n=A.length,m=B[0].length,p=B.length;
    return Array.from({length:n},(_,i)=>Array.from({length:m},(_,j)=>{
      let s=0;for(let k=0;k<p;k++) s+=A[i][k]*B[k][j];return s;
    }));
  }

  for(let it=0;it<500;it++){
    const {Q,R}=qrDecompose(M);
    M=matMul(R,Q);
  }

  // For this educational tool, use diagonal of quasi-upper-triangular matrix.
  return M.map((r,i)=>({re:r[i],im:0}));
}

function analyzeFixedPoint(){
  const x=analysisInputVector();
  const initial=app.history.length?app.history[app.history.length-1].h:app.h;
  const result=findFixedPoint(initial,x);

  if(!result.converged){
    $("fixedPointText").textContent=`Search did not fully converge. Residual = ${result.residual.toExponential(2)}`;
    $("stabilityText").textContent="Stability not classified.";
    $("eigenText").textContent="—";
    app.fixedPoint=null;
    drawTrajectory();
    return;
  }

  app.fixedPoint=result.point;
  const J=jacobianAt(result.point,x);
  const eig=eigenvaluesQR(J);
  const reals=eig.map(z=>z.re);
  const tol=1e-5;
  const allNeg=reals.every(r=>r<-tol);
  const allPos=reals.every(r=>r>tol);
  const mixed=reals.some(r=>r>tol)&&reals.some(r=>r<-tol);

  let stability="Marginal / inconclusive";
  if(allNeg) stability="Stable fixed point";
  else if(mixed) stability="Saddle / unstable";
  else if(allPos || reals.some(r=>r>tol)) stability="Unstable fixed point";

  $("fixedPointText").textContent=
    "["+result.point.map(v=>v.toFixed(4)).join(", ")+"]";

  $("stabilityText").textContent=stability;

  $("eigenText").textContent=
    eig.map((z,i)=>`λ${i+1} ≈ ${z.re.toFixed(4)}`).join("   ");

  drawTrajectory();
}

function updateEverything(){
  $("tauValue").textContent=app.tau.toFixed(1);
  $("dtValue").textContent=app.dt.toFixed(3);
  $("clockValue").textContent=`t = ${app.time.toFixed(2)} s`;
  updateOutputs();
  drawActivityPlot();
  drawTrajectory();
}

function animate(){
  if(app.running){
    for(let i=0;i<4;i++){
      if(app.time>=app.duration) break;
      stepCTRNN();
    }
    updateEverything();
  }
  requestAnimationFrame(animate);
}

// ---------- Events ----------
$("startBtn").addEventListener("click",()=>{
  if(app.time>=app.duration) resetSimulation();
  app.running=true; app.pcaReady=false; app.pcaProjection=null;
});

$("pauseBtn").addEventListener("click",()=>{
  app.running=false; computePCAFromHistory(); updateEverything();
});

$("resetBtn").addEventListener("click",resetSimulation);

$("activationSelect").addEventListener("change",e=>{
  app.activationName=e.target.value; resetSimulation();
});

$("tauSlider").addEventListener("input",e=>{
  app.tau=Number(e.target.value); updateEverything();
});

$("dtSlider").addEventListener("input",e=>{
  app.dt=Number(e.target.value); updateEverything();
});

$("durationInput").addEventListener("change",e=>{
  const v=Number(e.target.value);
  if(Number.isFinite(v)&&v>0) app.duration=v; else e.target.value=app.duration;
  if(app.time>app.duration) resetSimulation();
  updateEverything();
});

$("inputCount").addEventListener("change",e=>{
  app.inputCount=Number(e.target.value); initializeArchitecture();
});

$("outputCount").addEventListener("change",e=>{
  app.outputCount=Number(e.target.value); initializeWeights(); drawNetwork(); resetSimulation();
});

$("minusHidden").addEventListener("click",()=>{
  if(app.hiddenCount<=2) return; app.hiddenCount--; initializeArchitecture();
});

$("plusHidden").addEventListener("click",()=>{
  if(app.hiddenCount>=8) return; app.hiddenCount++; initializeArchitecture();
});

$("weightSlider").addEventListener("input",e=>updateSelectedWeight(Number(e.target.value)));

$("trajectoryMode").addEventListener("change",e=>{
  app.trajectoryMode=e.target.value; drawTrajectory();
});

$("analyzeBtn").addEventListener("click",analyzeFixedPoint);

initializeArchitecture();
animate();
