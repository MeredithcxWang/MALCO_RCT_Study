import fs from 'node:fs';
import path from 'node:path';

export function config() {
  const root = path.resolve(process.env.MALCO_ROOT || path.join(process.cwd(), '..'));
  const mode = process.env.MALCO_MODE || 'demo';
  if (!['demo', 'live'].includes(mode)) throw new Error('MALCO_MODE must be demo or live');
  const schema = JSON.parse(fs.readFileSync(path.join(root, 'data/schema.json'), 'utf8'));
  const output = path.resolve(process.env.MALCO_OUTPUT_DIR || path.join(root, 'output/demo'));
  const venvPython = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  return {root, mode, schema, output, modelsDir: process.env.MALCO_MODELS_DIR || path.join(output, 'models'),
    sessions: path.join(output, 'sessions'), archive: path.join(output, 'archive'),
    python: process.env.PYTHON_EXECUTABLE || (fs.existsSync(venvPython) ? venvPython : 'python'),
    tasks: (process.env.MALCO_TASKS || Object.keys(schema.tasks).join(',')).split(',').map(s=>s.trim()).filter(Boolean)};
}
export function httpError(message, status=400) { return Object.assign(new Error(message), {status}); }
export async function jsonObject(request) {
  let body;
  try {body=await request.json();} catch {throw httpError('Request body must be valid JSON');}
  if(!body || typeof body!=='object' || Array.isArray(body)) throw httpError('Request body must be a JSON object');
  return body;
}
export function safeRef(ref) {
  if (typeof ref !== 'string' || !/^[A-Za-z0-9_-]{1,128}$/.test(ref)) throw httpError('Invalid or missing reference key');
  return ref;
}
export async function requestRef(request, context={}) {
  const params = await context.params;
  return safeRef(params?.ref || new URL(request.url).searchParams.get('ref'));
}
export function sessionDir(ref, cfg=config()) { return path.join(cfg.sessions, safeRef(ref)); }
export function readJSON(file) { return JSON.parse(fs.readFileSync(file,'utf8')); }
export function writeJSON(file, value) {
  const temp = file + '.tmp';
  fs.writeFileSync(temp, JSON.stringify(value, null, 2), 'utf8');
  fs.renameSync(temp, file);
}
export function failure(error) {
  const status=error.status || 500;
  return Response.json({error:status===500 ? 'Operation failed; inspect the server log.' : error.message},{status});
}
export function lockSession(dir) {
  if (!fs.existsSync(dir)) throw httpError('Session not found',404);
  const lock=path.join(dir,'.lock');
  try { fs.mkdirSync(lock); } catch (e) { if(e.code==='EEXIST') throw httpError('Session is busy',409); throw e; }
  return ()=>{ if(fs.existsSync(lock)) fs.rmdirSync(lock); };
}
export function liveSettings(prefix) {
  const baseURL=process.env[`${prefix}_BASE_URL`] || process.env.OPENAI_BASE_URL;
  const apiKey=process.env[`${prefix}_API_KEY`] || process.env.OPENAI_API_KEY;
  if (!baseURL || !apiKey || /your_|replace|here/i.test(apiKey)) throw httpError(`Set ${prefix}_BASE_URL and ${prefix}_API_KEY for live mode`,503);
  let url; try { url=new URL(baseURL); } catch { throw httpError('Invalid LLM base URL',503); }
  if (!['https:','http:'].includes(url.protocol)) throw httpError('LLM base URL must use HTTP(S)',503);
  return {baseURL,apiKey};
}
