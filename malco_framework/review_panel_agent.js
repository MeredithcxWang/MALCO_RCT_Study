import fs from 'node:fs';
import path from 'node:path';
import OpenAI from 'openai';
import {PROMPTS} from './lib/review_prompt.js';
import {config, requestRef, sessionDir, writeJSON, failure, httpError, lockSession, liveSettings} from './lib/runtime.js';

export function determineVerdict(text) {
  const line=String(text || '').split(/\r?\n/)[0].trim();
  if(/^Requires Doctor Intervention(?:[.:\s]|$)/i.test(line) || /^(需要医生介入|需要介入)$/.test(line)) return 'need_doctor';
  if(/^(Passed|Pass|通过)[.!]?$/i.test(line)) return 'pass';
  return 'unknown';
}
export async function POST(request,context={}) {
  let unlock;
  try {
    const ref=await requestRef(request,context), cfg=config(), dir=sessionDir(ref,cfg);
    unlock=lockSession(dir);
    const file=path.join(dir,'chatlog.txt');
    if(!fs.existsSync(file)) throw httpError('Complete at least one chat response first',409);
    const chatLog=fs.readFileSync(file,'utf8');
    let reviewers, configuration='Demo';
    if(cfg.mode==='demo') reviewers=[1,2,3].map(i=>({key:`demo${i}`,model:'offline-test-double'}));
    else {
      if(process.env.REVIEW_MODELS) {
        try {reviewers=JSON.parse(process.env.REVIEW_MODELS);} catch {throw httpError('REVIEW_MODELS must be valid JSON',503);}
        configuration='Custom';
      } else {
        const isB=Math.random()<.6; configuration=isB?'Configuration_B':'Configuration_A';
        const defaults={Configuration_A:[{key:'deepseek',model:'deepseek-r1'},{key:'o3',model:'o3-2025-04-16'},{key:'claude37',model:'claude-3-7-sonnet-20250219'}],
          Configuration_B:[{key:'gpt5',model:'gpt-5-2025-08-07'},{key:'grok4',model:'grok-4-2025-07-09'},{key:'claudeOpus',model:'claude-opus-4.1-2025-08-05'}]};
        let configurations=defaults;
        if(process.env.REVIEW_CONFIGURATIONS) {try {configurations=JSON.parse(process.env.REVIEW_CONFIGURATIONS);} catch {throw httpError('Invalid REVIEW_CONFIGURATIONS JSON',503);}}
        reviewers=configurations[configuration];
      }
      if(!Array.isArray(reviewers) || reviewers.length!==3 || reviewers.some(r=>!r.key || !r.model) || new Set(reviewers.map(r=>r.key)).size!==3) throw httpError('Configure three distinct reviewer keys and model IDs in REVIEW_MODELS',503);
    }
    const results=await Promise.all(reviewers.map(async reviewer=> {
      try {
        let reply;
        if(cfg.mode==='demo') reply='Passed\n[OFFLINE DEMO] Fixed test response, not a clinical safety assessment.';
        else {
          const settings=liveSettings(reviewer.envPrefix || 'REVIEW');
          const client=new OpenAI({...settings,timeout:60000,maxRetries:0});
          const result=await client.chat.completions.create({model:reviewer.model,temperature:.2,messages:[
            {role:'system',content:'You are a meticulous clinical safety auditor.'},
            {role:'user',content:PROMPTS.buildAuditPrompt(chatLog)}]});
          reply=result.choices?.[0]?.message?.content;
          if(result.choices?.[0]?.finish_reason!=='stop') throw new Error('Reviewer response did not complete normally');
          if(typeof reply!=='string' || !reply.trim()) throw new Error('Empty reviewer response');
        }
        return {key:reviewer.key,model_used:reviewer.model,full_response:reply,verdict:determineVerdict(reply),status:'completed'};
      } catch(e) {
        console.error(`Reviewer ${reviewer.key} failed:`,e.message);
        return {key:reviewer.key,model_used:reviewer.model,verdict:'unknown',status:'error',error:'Reviewer request failed'};
      }
    }));
    const incomplete=results.some(r=>r.status!=='completed' || r.verdict==='unknown');
    const result={referenceKey:ref,mode:cfg.mode,applied_configuration:configuration,review_complete:!incomplete,
      final_decision:results.every(r=>r.verdict==='pass') ? 'Pass' : 'Need Physician Review',
      agent_results:Object.fromEntries(results.map(r=>[r.key,r])),
      notice:cfg.mode==='demo' ? 'OFFLINE TEST ONLY: no clinical safety assessment was performed.' : 'Human supervision remains required.'};
    writeJSON(path.join(dir,'review.json'),result);
    const archive=path.join(cfg.archive,ref);
    fs.mkdirSync(archive,{recursive:true});
    for(const name of ['patient.csv','predictions.json','session.json','chat.json','chatlog.txt','review.json']) {
      const source=path.join(dir,name);
      if(fs.existsSync(source)) fs.copyFileSync(source,path.join(archive,name));
    }
    return Response.json(result,{status:incomplete ? 502 : 200});
  } catch(e) {console.error('Review failed:',e.message);return failure(e);} finally {unlock?.();}
}
