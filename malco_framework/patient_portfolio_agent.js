import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import {config, safeRef, sessionDir, readJSON, writeJSON, failure, httpError, jsonObject} from './lib/runtime.js';
const run=promisify(execFile);

export async function GET(request) {
  try {
    const ref=safeRef(new URL(request.url).searchParams.get('ref'));
    const file=path.join(sessionDir(ref),'predictions.json');
    if(!fs.existsSync(file)) throw httpError('Predictions not found',404);
    return Response.json(readJSON(file));
  } catch(e) {return failure(e);}
}
export async function POST(request) {
  let dir;
  try {
    const body=await jsonObject(request);
    if(body.ref && !body.answers) return GET(new Request(`http://localhost/api/submit-answers?ref=${encodeURIComponent(body.ref)}`));
    const cfg=config(), ref=safeRef(body.referenceKey);
    if(!body.answers || typeof body.answers!=='object' || Array.isArray(body.answers)) throw httpError('answers must be an object');
    const values=cfg.schema.features.map((feature,i)=> {
      const value=body.answers[`Q${i+1}`];
      if(value===null || value===undefined || value==='') return '';
      if(value==='A') return 1;
      if(value==='B') return 0;
      if(typeof value==='boolean' || typeof value==='object' || !Number.isFinite(Number(value))) throw httpError(`Q${i+1} must be numeric or missing`);
      return Number(value);
    });
    if(values.every(v=>v==='')) throw httpError('At least one observed feature is required');
    dir=sessionDir(ref,cfg);
    fs.mkdirSync(cfg.sessions,{recursive:true});
    try {fs.mkdirSync(dir);} catch(e) {if(e.code==='EEXIST') throw httpError('Reference key already exists; use a new key',409); throw e;}
    const csv=path.join(dir,'patient.csv');
    fs.writeFileSync(csv,cfg.schema.features.join(',')+'\n'+values.join(',')+'\n','utf8');
    const {stdout}=await run(cfg.python,[path.join(cfg.root,'malco_framework/run_models.py'),'--input',csv,
      '--models-dir',cfg.modelsDir,'--tasks',cfg.tasks.join(','),'--output-dir',dir],
      {encoding:'utf8',timeout:120000,maxBuffer:8*1024*1024,env:{...process.env,MPLBACKEND:'Agg'}});
    const result=JSON.parse(stdout);
    if(result.predictions?.length!==cfg.tasks.length || result.predictions.some(p=>!Number.isFinite(p.value)||p.value<0||p.value>1)) throw new Error('Invalid Python prediction response');
    writeJSON(path.join(dir,'session.json'),{referenceKey:ref,mode:cfg.mode,status:'ready'});
    return Response.json({...result,referenceKey:ref,mode:cfg.mode});
  } catch(e) {
    // Failed sessions keep their evidence and may be retried under a new reference key.
    if(dir && fs.existsSync(dir) && e.status!==409) writeJSON(path.join(dir,'failure.json'),{error:'Inference did not complete'});
    console.error('Portfolio request failed:',e.message);
    return failure(e);
  }
}
