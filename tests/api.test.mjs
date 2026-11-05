import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
process.env.MALCO_ROOT=root;
process.env.MALCO_MODE='demo';
process.env.MALCO_MODELS_DIR=path.join(root,'output/demo/models');
process.env.MALCO_OUTPUT_DIR=path.join(root,'output/api-tests',String(Date.now()));
const portfolio=await import('../malco_framework/patient_portfolio_agent.js');
const chat=await import('../malco_framework/conversational_agent.js');
const review=await import('../malco_framework/review_panel_agent.js');
function request(url,body){return new Request('http://localhost'+url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:undefined);}
const payload=JSON.parse(fs.readFileSync(path.join(root,'output/demo/input/patient_request.json'),'utf8'));

test('complete offline API workflow uses real Python models and structured SHAP',async()=>{
  let r=await portfolio.POST(request('/api/submit-answers',payload));
  assert.equal(r.status,200,await r.clone().text());
  const predictions=(await r.json()).predictions;
  assert.equal(predictions.length,8);
  assert.equal(predictions[0].shap.contributions.length,41);
  r=await chat.GET(request('/api/chat?ref=demo001'));
  assert.equal(r.status,200);
  const messages=(await r.json()).messages;
  assert.ok(messages[1].content.includes('shap_value'));
  r=await chat.POST(request('/api/chat?ref=demo001',{message:'Synthetic test message'}));
  assert.equal(r.status,200);
  const events=(await r.text()).trim().split('\n\n').map(s=>JSON.parse(s.slice(6)));
  assert.equal(events[0].mode,'demo');
  assert.equal(events.at(-1).type,'done');
  const other=path.join(process.env.MALCO_OUTPUT_DIR,'sessions','demo001extra');
  fs.mkdirSync(other);fs.writeFileSync(path.join(other,'marker'),'untouched');
  r=await review.POST(request('/api/review?ref=demo001',{}));
  assert.equal(r.status,200);
  assert.equal((await r.json()).final_decision,'Pass');
  assert.equal(fs.readFileSync(path.join(other,'marker'),'utf8'),'untouched');
  assert.ok(!fs.existsSync(path.join(process.env.MALCO_OUTPUT_DIR,'archive','demo001extra')));
  r=await chat.POST(request('/api/chat?ref=demo001',{message:'After review'}));
  assert.equal(r.status,409,'Reviewed sessions must not acquire unreviewed messages');
});

test('bad input is rejected and documented verdict text is recognized',async()=>{
  assert.equal(review.determineVerdict('Requires Doctor Intervention\nReason'),'need_doctor');
  assert.equal(review.determineVerdict('not passed'),'unknown');
  assert.equal(review.determineVerdict('Passed'),'pass');
  let r=await portfolio.POST(request('/api/submit-answers',{...payload,referenceKey:'../bad'}));
  assert.equal(r.status,400);
  r=await portfolio.POST(request('/api/submit-answers',{...payload,referenceKey:'invalidnum',answers:{Q2:'not a number'}}));
  assert.equal(r.status,400);
  r=await portfolio.POST(request('/api/submit-answers',payload));assert.equal(r.status,409);
  r=await chat.GET(request('/api/chat'));assert.equal(r.status,400);
  for (const handler of [portfolio.POST,chat.POST]) {
    r=await handler(new Request('http://localhost/api/chat?ref=demo001',{method:'POST',body:'null'}));
    assert.equal(r.status,400);
    r=await handler(new Request('http://localhost/api/chat?ref=demo001',{method:'POST',body:'{'}));
    assert.equal(r.status,400);
  }
});

test('live adapter against local test server vetoes and fails closed',async()=>{
  let fail=false, truncated=false;
  const server=http.createServer(async(req,res)=>{
    let body='';for await(const part of req)body+=part;
    const args=JSON.parse(body);
    res.setHeader('Content-Type','application/json');
    if(fail && args.model==='broken'){res.statusCode=500;res.end(JSON.stringify({error:{message:'test failure'}}));return;}
    res.end(JSON.stringify({choices:[{finish_reason:truncated?'length':'stop',message:{content:args.model==='veto'?'Requires Doctor Intervention\nSynthetic reason':'Passed'}}]}));
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  process.env.MALCO_MODE='live';
  process.env.REVIEW_BASE_URL=`http://127.0.0.1:${server.address().port}/v1`;
  process.env.REVIEW_API_KEY='test-only-placeholder';
  try{
    process.env.REVIEW_MODELS=JSON.stringify([{key:'a',model:'pass'},{key:'b',model:'veto'},{key:'c',model:'pass'}]);
    let r=await review.POST(request('/api/review?ref=demo001',{}));
    assert.equal(r.status,200);assert.equal((await r.json()).final_decision,'Need Physician Review');
    fail=true;
    process.env.REVIEW_MODELS=JSON.stringify([{key:'a',model:'pass'},{key:'b',model:'broken'},{key:'c',model:'pass'}]);
    r=await review.POST(request('/api/review?ref=demo001',{}));
    assert.equal(r.status,502);
    const result=await r.json();assert.equal(result.review_complete,false);assert.equal(result.final_decision,'Need Physician Review');
    fail=false;truncated=true;
    r=await review.POST(request('/api/review?ref=demo001',{}));
    assert.equal(r.status,502,'Truncated reviews must never count as completed');
  } finally {process.env.MALCO_MODE='demo';await new Promise(resolve=>server.close(resolve));}
});

test('live chat only persists a response with an explicit normal completion',async()=>{
  let finish='length';
  const server=http.createServer(async(req,res)=>{
    for await(const part of req){};
    res.setHeader('Content-Type','text/event-stream');
    const chunk={choices:[{delta:{content:'Synthetic streamed response'}}]};
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    if(finish)res.write(`data: ${JSON.stringify({choices:[{delta:{},finish_reason:finish}]})}\n\n`);
    res.end('data: [DONE]\n\n');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const dir=path.join(process.env.MALCO_OUTPUT_DIR,'sessions','streamtest');
  fs.mkdirSync(dir);
  const initial=[{role:'system',content:'Test only'}];
  fs.writeFileSync(path.join(dir,'chat.json'),JSON.stringify(initial));
  process.env.MALCO_MODE='live';process.env.CHAT_MODEL='local-test';
  process.env.CHAT_BASE_URL=`http://127.0.0.1:${server.address().port}/v1`;
  process.env.CHAT_API_KEY='test-only-placeholder';
  try {
    for(finish of ['length','content_filter',null,'stop']) {
      const r=await chat.POST(request('/api/chat?ref=streamtest',{message:'test'}));
      const events=(await r.text()).trim().split('\n\n').map(s=>JSON.parse(s.slice(6)));
      assert.equal(events.at(-1).type,finish==='stop'?'done':'error');
      const saved=JSON.parse(fs.readFileSync(path.join(dir,'chat.json'),'utf8'));
      assert.equal(saved.length,finish==='stop'?3:1);
      assert.ok(!fs.existsSync(path.join(dir,'.lock')));
    }
  } finally {process.env.MALCO_MODE='demo';await new Promise(resolve=>server.close(resolve));}
});
