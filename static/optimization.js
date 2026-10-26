/* Appended optimization section. No handlers or state from sections 0–3 are changed. */
(() => {
 'use strict';
 const el = id => document.getElementById('opt-' + id);
 const names = {qlearning:'Q-Learning', double_qlearning:'Double Q-Learning'};
 const methods = {
  grid:['Grid search','Chaque combinaison est testée. Une grille simple à lire et à reproduire : le meilleur point est choisi parmi les valeurs explorées.'],
  random:['Random search','Les paires sont tirées au hasard dans les plages choisies. Chaque essai reçoit le même budget complet d’apprentissage.'],
  bayesian:['Optimisation bayésienne','Après cinq essais initiaux, un modèle probabiliste utilise les scores précédents pour proposer les prochaines paires, en équilibrant exploration et amélioration attendue.'],
  lhs:['Latin hypercube','La plage de chaque paramètre est divisée en intervalles, chacun échantillonné une fois. Cette couverture stratifiée évite de concentrer tous les essais sur quelques valeurs.'],
  hyperband:['Hyperband','Plusieurs groupes commencent avec des budgets différents. Les candidats prometteurs continuent, les autres s’arrêtent. Seuls les essais au budget complet peuvent gagner.']
 };
 const fields = {method:'method',algorithm:'algorithm',episodes:'episodes',trials:'trials',grid_size:'grid',repeats:'repeats',eta_min:'eta-min',eta_max:'eta-max',gamma_min:'gamma-min',gamma_max:'gamma-max',validation_seeds:'validation',test_seeds:'test',seed:'seed'};
 const fmt = (x,n=2) => x == null || !Number.isFinite(Number(x)) ? '—' : Number(x).toLocaleString('fr-FR',{minimumFractionDigits:n, maximumFractionDigits:n});
 const esc = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 let state = {status:'idle',running:false,trials:[],results:{},history:[]}, timer, fetching=false, stopRequested=false, hydrated=false;
 async function api(path,body) {
  const response = await fetch('/api/optimization/'+path,body===undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const value = await response.json();
  if(!response.ok) throw new Error(value.error || 'La recherche est indisponible.');
  return value;
 }
 function error(message) {el('error').hidden=!message;el('error').textContent=message || '';}
 function configuration() {
  return Object.fromEntries(Object.entries(fields).map(([key,id])=>[key,['method','algorithm'].includes(key)?el(id).value:Number(el(id).value)]));
 }
 function updateSetup(event) {
  const method=el('method').value;
  el('trials').min=method==='bayesian'?'6':'4';
  el('episodes').min=method==='hyperband'?'150':'50';
  if(!event||event.target===el('method')){
   if(Number(el('trials').value)<Number(el('trials').min))el('trials').value=el('trials').min;
   if(Number(el('episodes').value)<Number(el('episodes').min))el('episodes').value=el('episodes').min;
  }
  const p=configuration(),grid=p.method==='grid';
  el('grid-label').hidden=!grid;el('trials-label').hidden=grid;
  el('description').innerHTML=`<b>${methods[p.method][0]}.</b> ${methods[p.method][1]}`;
  el('grid-count').textContent=`${p.grid_size} × ${p.grid_size} = ${p.grid_size**2} paires`;
  const count=grid?p.grid_size**2:p.trials, algorithms=p.algorithm==='both'?2:1;
  el('estimate').textContent=`${count} paires à explorer par agent + une référence par défaut. ${algorithms} agent${algorithms>1?'s':''} × ${p.repeats} entraînement${p.repeats>1?'s':''} indépendant${p.repeats>1?'s':''} par paire. ${p.method==='hyperband'?'Les budgets sont répartis par Hyperband.':fmt((count+1)*p.episodes*p.repeats*algorithms,0)+' épisodes d’apprentissage au total.'} Augmenter le budget rend la comparaison plus coûteuse.`;
 }
 el('form').addEventListener('input', updateSetup);
 el('form').addEventListener('change', updateSetup);
 function liveBest(algorithm) {
  const p=state.params;
  const rows=state.trials.filter(r=>r.algorithm===algorithm && r.episodes===p.episodes && r.status==='complete');
  return rows.length?rows.reduce((a,b)=>a.metrics.score>=b.metrics.score?a:b):null;
 }
 function renderResults() {
  const p=state.params;
  if(!p){el('results').innerHTML='';return;}
  const algorithms=p.algorithm==='both'?Object.keys(names):[p.algorithm];
  el('results').innerHTML=algorithms.map(algorithm=>{
   const result=state.results[algorithm],best=liveBest(algorithm);
   if(!result&&!best)return '';
   const eta=result?result.eta:best.eta,gamma=result?result.gamma:best.gamma,m=result?result.validation:best.metrics;
   let body=`<h3>${names[algorithm]}</h3><span class="opt-tag">${result?(result.is_default?'Paramètres par défaut conservés':'Meilleure paire trouvée'):'Meilleur résultat provisoire'}</span><div class="opt-pair"><span>γ ${fmt(gamma,4)}<small>importance du futur</small></span><span>η ${fmt(eta,5)}<small>vitesse d’apprentissage</small></span></div><p class="note">Validation : <b>${fmt(m.score,3)} ± ${fmt(m.score_std,3)}</b> par requête.</p>`;
   if(result){
    const t=result.test,b=result.baseline_test,g=result.test_gain_pct;
    body+=`<table><thead><tr><th>Test indépendant</th><th>Défaut</th><th>Paire retenue</th></tr></thead><tbody><tr><td>Score ↑</td><td>${fmt(b.score,3)}</td><td><b>${fmt(t.score,3)}</b></td></tr><tr><td>Latence (ms) ↓</td><td>${fmt(b.latency_ms,1)}</td><td>${fmt(t.latency_ms,1)}</td></tr><tr><td>Rejets (%) ↓</td><td>${fmt(b.rejection_rate*100,2)}</td><td>${fmt(t.rejection_rate*100,2)}</td></tr></tbody></table>`;
    body+=`<p class="note ${g>=0?'ok':'ko'}"><b>${result.is_default?'Aucune paire ne dépasse la référence en validation.':g>0?'Coût moyen réduit de '+fmt(g,2)+' % au test.':g<0?'Gain de validation non confirmé : coût accru de '+fmt(-g,2)+' % au test.':'Même coût moyen que la référence au test.'}</b></p>`;
    body+=`<p class="note">Test : ${t.seeds_per_scenario} nouveaux scénarios × 3 niveaux de trafic × ${t.repeats} entraînements. Score : ${fmt(t.score,3)} ± ${fmt(t.score_std,3)} entre entraînements.</p>`;
   }else body+='<p class="note">Le choix reste provisoire jusqu’à la fin de la recherche et du test indépendant.</p>';
   return `<article class="opt-result">${body}</article>`;
  }).join('');
 }
 function render() {
  const p=state.params,active=state.running;
  for(const control of el('form').querySelectorAll('input, select')) control.disabled=active;
  el('start').disabled=active;el('stop').disabled=!active||stopRequested;
  el('export').disabled=active||!state.id;
  el('start').textContent=state.id?'Lancer une nouvelle recherche':'Lancer l’optimisation';
  const percent=Math.round((state.progress||0)*100);
  el('progress-fill').style.width=percent+'%';el('progress').setAttribute('aria-valuenow',percent);
  const c=state.current||{};
  if(active&&c.phase){
   const phase={training:'Entraînement',validation:'Validation sans exploration',test:'Test final indépendant'}[c.phase];
   el('status').textContent=stopRequested?'Arrêt en cours…':`${phase} · ${names[c.algorithm]} · γ ${fmt(c.gamma,4)} / η ${fmt(c.eta,5)}${c.phase==='training'?` · épisode ${c.episode}/${c.target}, répétition ${c.repeat}/${p.repeats}`:''}`;
  }else el('status').textContent=state.message||'Prêt à comparer les paramètres. Référence : γ = 0,9 et η = 0,1.';
  el('run-meta').textContent=p?`${methods[p.method][0]} · ${state.trials.filter(t=>t.label!=='baseline').length} paires évaluées · ${fmt(state.elapsed_s||0,0)} s · ${p.episodes} épisodes au budget complet · graine ${p.seed}${p.repeats===1?' · Une seule répétition : variabilité non mesurée.':''}`:'';
  renderResults();
  el('plots').hidden=!state.trials.length;el('trial-area').hidden=!state.trials.length;
  for(const option of el('view').options)option.disabled=!!p && p.algorithm!=='both'&&option.value!==p.algorithm;
  if(p&&p.algorithm!=='both')el('view').value=p.algorithm;
  const winnerIds=Object.values(state.results).map(r=>r.trial_id);
  el('trial-rows').innerHTML=[...state.trials].reverse().map(t=>`<tr class="${winnerIds.includes(t.id)?'opt-winner':''}"><td><b>${t.label==='baseline'?'Référence':'#'+t.id}</b><span class="opt-agent">${names[t.algorithm]}${t.bracket?' · groupe '+t.bracket:''}</span></td><td>${fmt(t.gamma,4)}</td><td>${fmt(t.eta,5)}</td><td>${t.episodes}</td><td>${fmt(t.metrics.score,3)} ± ${fmt(t.metrics.score_std,3)}</td><td>${fmt(t.metrics.latency_ms,1)}</td><td>${fmt(t.metrics.rejection_rate*100,2)}</td><td>${t.status==='complete'?'Budget complet':t.status==='pruned'?'Arrêt anticipé':'Budget partiel'}</td></tr>`).join('');
  drawCharts();
 }
 function canvas(id) {
  const element=el(id),width=element.clientWidth,height=240;
  if(!width)return null;
  const ratio=window.devicePixelRatio||1;
  element.width=Math.round(width*ratio);element.height=Math.round(height*ratio);
  const ctx=element.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);
  const css=getComputedStyle(document.documentElement);
  return {ctx,width,height,ink:css.getPropertyValue('--ink').trim(),mute:css.getPropertyValue('--mute').trim(),line:css.getPropertyValue('--line').trim(),accent:css.getPropertyValue('--acc').trim()};
 }
 function axes(c,xlabel,ylabel) {
  const {ctx:x,width:w,height:h,mute,line}=c;
  x.font='11px system-ui';x.fillStyle=mute;x.strokeStyle=line;x.lineWidth=1;
  x.beginPath();x.moveTo(54,24);x.lineTo(54,h-38);x.lineTo(w-18,h-38);x.stroke();
  x.fillText(ylabel,8,14);x.textAlign='right';x.fillText(xlabel,w-18,h-4);x.textAlign='left';
 }
 function drawSpace(rows) {
  const c=canvas('space');if(!c)return;
  axes(c,'γ · importance du futur','η · vitesse (échelle log)');
  const {ctx:x,width:w,height:h}=c,p=state.params;
  const loG=Math.min(p.gamma_min,.9),hiG=Math.max(p.gamma_max,.9),loE=Math.min(p.eta_min,.1),hiE=Math.max(p.eta_max,.1);
  const px=g=>54+(g-loG)/(hiG-loG)*(w-72),py=e=>h-38-(Math.log(e/loE)/Math.log(hiE/loE))*(h-62);
  x.fillStyle=c.mute;x.font='10px system-ui';
  [0,.5,1].forEach(t=>{const g=loG+t*(hiG-loG);x.fillText(fmt(g,2),px(g)-9,h-22);const eta=loE*Math.pow(hiE/loE,t);x.fillText(fmt(eta,3),5,py(eta)+3);});
  const full=rows.filter(r=>r.status==='complete'),scores=full.map(r=>r.metrics.score),lo=Math.min(...scores),hi=Math.max(...scores);
  for(const row of rows){
   const complete=row.status==='complete',v=(row.metrics.score-lo)/(hi-lo||1),color=complete?`hsl(${175+v*5}, ${25+v*45}%, ${40+v*12}%)`:c.mute;
   x.beginPath();const a=px(row.gamma),b=py(row.eta);
   if(row.label==='baseline'){x.moveTo(a,b-7);x.lineTo(a+7,b);x.lineTo(a,b+7);x.lineTo(a-7,b);x.closePath();}
   else x.arc(a,b,5,0,Math.PI*2);
   x.strokeStyle=color;x.fillStyle=color;x.lineWidth=1.5;if(complete)x.fill();else x.stroke();
  }
 }
 function drawConvergence(rows) {
  const c=canvas('convergence');if(!c)return;
  axes(c,'Essais au budget complet','Score ↑');
  const full=state.history.filter(r=>r.algorithm===el('view').value&&r.episodes===state.params.episodes);if(!full.length)return;
  const {ctx:x,width:w,height:h}=c,base=rows.find(r=>r.label==='baseline');
  let best=-Infinity;const values=full.map(r=>(best=Math.max(best,r.score)));
  const lo=Math.min(...full.map(r=>r.score))-.01,hi=Math.max(...values)+.01;
  const py=v=>h-38-(v-lo)/(hi-lo)*(h-62),px=i=>54+i/Math.max(1,values.length-1)*(w-72);
  x.fillStyle=c.mute;x.font='10px system-ui';[0,.5,1].forEach(t=>{const v=lo+(hi-lo)*t;x.fillText(fmt(v,2),3,py(v)+3);});
  x.fillText('1',52,h-22);x.fillText(String(values.length),w-25,h-22);
  if(base){x.strokeStyle=c.mute;x.setLineDash([4,4]);x.beginPath();x.moveTo(54,py(base.metrics.score));x.lineTo(w-18,py(base.metrics.score));x.stroke();x.setLineDash([]);}
  x.strokeStyle=c.accent;x.lineWidth=2;x.beginPath();values.forEach((v,i)=>{if(i)x.lineTo(px(i),py(v));else x.moveTo(px(i),py(v));});x.stroke();
  x.fillStyle=c.accent;values.forEach((v,i)=>{x.beginPath();x.arc(px(i),py(v),3,0,Math.PI*2);x.fill();});
 }
 function drawCharts() {
  if(!el('panel').open||!state.params||!state.trials.length)return;
  const rows=state.trials.filter(r=>r.algorithm===el('view').value);
  drawSpace(rows);drawConvergence(rows);
 }
 function schedule(){clearTimeout(timer);if(state.running)timer=setTimeout(poll,1000);}
 async function poll(){
  if(fetching)return;fetching=true;
  try{state=await api('status');
   if(!hydrated){
    if(state.params){for(const [key,id] of Object.entries(fields))el(id).value=state.params[key];updateSetup();}
    hydrated=true;
   }
   if(!state.running)stopRequested=false;render();error('');schedule();}
  catch(e){error('Connexion interrompue : '+e.message+' La recherche peut continuer côté serveur.');timer=setTimeout(poll,3000);}
  finally{fetching=false;}
 }
 el('form').addEventListener('submit',async event=>{
  event.preventDefault();if(state.running)return;
  const p=configuration();
  if(!(p.eta_min>0&&p.eta_min<p.eta_max&&p.eta_max<=1&&p.gamma_min>=0&&p.gamma_min<p.gamma_max&&p.gamma_max<1)){error('Vérifiez les plages : 0 < η min < η max ≤ 1 et 0 ≤ γ min < γ max < 1.');return;}
  el('start').disabled=true;error('');
  try{state=await api('start',p);stopRequested=false;render();schedule();}
  catch(e){error(e.message);el('start').disabled=false;}
 });
 el('stop').addEventListener('click',async()=>{
  stopRequested=true;el('stop').disabled=true;
  try{state=await api('stop',{});render();schedule();}catch(e){error(e.message);stopRequested=false;el('stop').disabled=false;}
 });
 el('export').addEventListener('click',async()=>{
  try{const response=await fetch('/api/optimization/report');if(!response.ok){const j=await response.json();throw new Error(j.error);}
   const url=URL.createObjectURL(await response.blob()),a=document.createElement('a');a.href=url;a.download='optimisation-'+state.id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){error(e.message);}
 });
 el('view').addEventListener('change',drawCharts);
 el('panel').addEventListener('toggle',()=>{if(el('panel').open)requestAnimationFrame(drawCharts);});
 window.addEventListener('resize',drawCharts);
 window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',drawCharts);
 updateSetup();poll();
})();
