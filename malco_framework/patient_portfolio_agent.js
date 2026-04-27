import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

// ============================================================================
// Constants & Mappings
// ============================================================================
const ML_RUN_DIR = path.join(process.cwd(), 'ML', 'run');
const MASTER_DATA_FILE = path.join(process.cwd(), 'data.csv');

// Maps questionnaire items (Q1–Q41) to XGBoost model feature names
const FEATURE_MAP = {
  Q1: 'use.Sex',      Q2: 'use.age',             Q3: 'use.Alcohol',       Q4: 'use.Smoke',
  Q5: 'use.HT',       Q6: 'use.DM',              Q7: 'use.obesity',       Q8: 'use.CAD',
  Q9: 'use.PVD',      Q10: 'use.AF',             Q11: 'use.DL',           Q12: 'use.Anemia',
  Q13: 'use.cirrhosis', Q14: 'use.parkinson',    Q15: 'use.rheumatism',   Q16: 'use.ACEI.base',
  Q17: 'use.ARB.base', Q18: 'use.Bblocker.base', Q19: 'use.ASA.base',     Q20: 'use.CCB.base',
  Q21: 'use.duretics.base', Q22: 'Hemoglobin.base.value', Q23: 'HbA1c.base.value',
  Q24: 'Glucose.base.value', Q25: 'Creatinie.base.value', Q26: 'Lymph.base.value',
  Q27: 'Neurophil.base.value', Q28: 'calcu.TSAT.all', Q29: 'TIBC.base.result',
  Q30: 'Iron.base.Result', Q31: 'FerritinugL', Q32: 'CRP.value', Q33: 'Iron.saturation.value',
  Q34: 'use.TG.base.result', Q35: 'mean.BMI', Q36: 'use.ALT.base.result',
  Q37: 'LDL.1y.Numeric.Result', Q38: 'statin_base', Q39: 'Albumin_Result',
  Q40: 'AST', Q41: 'eGFR'
};

const SAFE_KEY_PATTERN = /^[a-zA-Z0-9_-]{1,128}$/;

function validateReferenceKey(key) {
  if (!key || typeof key !== 'string' || !SAFE_KEY_PATTERN.test(key)) {
    throw new Error(`Invalid reference key: "${key}"`);
  }
  return key;
}

function sanitizeCSVValue(value) {
  const s = String(value);
  if (/^[=+\-@\t\r]/.test(s)) {
    return "'" + s;
  }
  return s;
}

/**
 * Reads and parses the prediction results file for a given reference key.
 */
function readPredictionResults(referenceKey) {
  const resultsFile = path.join(ML_RUN_DIR, `${referenceKey}_results.txt`);
  if (!fs.existsSync(resultsFile)) {
    return null;
  }

  const resultData = fs.readFileSync(resultsFile, 'utf8');
  return resultData.trim().split('\n').map(line => {
    const [model, p, risk] = line.split(',');
    return { model, value: parseFloat(p), risk };
  });
}

/**
 * Appends patient questionnaire data to the master record file.
 */
function appendToMasterCSV(name, referenceKey, answers) {
  const BOM = '\uFEFF';
  const D = ',';
  const quote = s => `"${sanitizeCSVValue(s).replace(/"/g, '""')}"`;

  const allPossibleQuestions = Array.from({ length: 41 }, (_, i) => `Q${i + 1}`);
  const incomingKeys = Array.from(new Set([...Object.keys(answers), ...allPossibleQuestions]));

  // Initialize file with headers if it does not exist
  if (!fs.existsSync(MASTER_DATA_FILE)) {
    const headers = ['Name', 'Reference Key', ...incomingKeys];
    fs.writeFileSync(MASTER_DATA_FILE, BOM + headers.map(quote).join(D) + '\n', 'utf8');
  } else {
    // Merge any new columns into the existing header
    const lines = fs.readFileSync(MASTER_DATA_FILE, 'utf8').split('\n');
    const oldKeys = lines[0].replace(/^\uFEFF/, '').split(D).map(h => h.replace(/^"|"$/g, '').trim());
    const newKeys = incomingKeys.filter(k => !oldKeys.includes(k));

    if (newKeys.length) {
      const merged = Array.from(new Set(oldKeys.concat(newKeys)));
      const newLines = lines.map((line, idx) => {
        if (idx === 0) return BOM + merged.map(quote).join(D);
        if (!line) return '';
        return line + D.repeat(newKeys.length);
      });
      fs.writeFileSync(MASTER_DATA_FILE, newLines.join('\n'), 'utf8');
    }
  }

  // Read the current header and append the patient's row
  const headerNow = fs.readFileSync(MASTER_DATA_FILE, 'utf8')
    .split('\n')[0].replace(/^\uFEFF/, '').split(D)
    .map(h => h.replace(/^"|"$/g, '').trim());

  const row = headerNow.map(h => {
    if (h === 'Name') return name;
    if (h === 'Reference Key') return referenceKey;
    return answers[h] ?? '';
  });

  fs.appendFileSync(MASTER_DATA_FILE, row.map(quote).join(D) + '\n', 'utf8');
}

/**
 * Generates a single-row CSV file for ML model inference.
 */
function generatePatientInferenceCSV(referenceKey, answers) {
  if (!fs.existsSync(ML_RUN_DIR)) {
    fs.mkdirSync(ML_RUN_DIR, { recursive: true });
  }

  const runHeaders = Object.values(FEATURE_MAP);
  const runValues = runHeaders.map(featureName => {
    const questionKey = Object.keys(FEATURE_MAP).find(q => FEATURE_MAP[q] === featureName);
    if (!questionKey) return '';

    const v = answers[questionKey];
    if (v === undefined || v === '') return '';
    if (v === 'A') return '1';
    if (v === 'B') return '0';
    return v;
  });

  const quote = s => `"${String(s).replace(/"/g, '""')}"`;
  const runCsv = runHeaders.map(quote).join(',') + '\n' + runValues.map(quote).join(',') + '\n';

  const runFile = path.join(ML_RUN_DIR, `${referenceKey}.csv`);
  fs.writeFileSync(runFile, runCsv, 'utf8');
  return runFile;
}


// Retrieve existing prediction results by reference key
export async function GET(request) {
  try {
    const url = new URL(request.url);
    const ref = url.searchParams.get('ref');

    if (!ref) {
      return new Response(JSON.stringify({ error: "Missing reference key" }), { status: 400 });
    }

    const safeRef = validateReferenceKey(ref);
    const predictions = readPredictionResults(safeRef);
    if (!predictions) {
      return new Response(JSON.stringify({ error: "Results not found" }), { status: 404 });
    }

    return new Response(JSON.stringify({ predictions }), { status: 200 });
  } catch (err) {
    console.error('[/api/submit-answers] GET error:', err);
    return new Response(JSON.stringify({ error: err.message }), { status: 500 });
  }
}

// Process questionnaire submission, persist data, and run prediction models
export async function POST(request) {
  try {
    const body = await request.json();

    // Polling request from the results page (ref only, no answers)
    if (body.ref && !body.answers) {
      const safeRef = validateReferenceKey(body.ref);
      const predictions = readPredictionResults(safeRef);
      if (!predictions) {
        return new Response(JSON.stringify({ error: "Results not found" }), { status: 404 });
      }
      return new Response(JSON.stringify({ predictions }), { status: 200 });
    }

    // Full questionnaire submission
    const { name, referenceKey, answers } = body;
    if (!answers || typeof answers !== 'object') {
      return new Response(JSON.stringify({ error: "Missing or invalid answers" }), { status: 400 });
    }

    const safeRef = validateReferenceKey(referenceKey);

    // Persist patient data to the master CSV
    appendToMasterCSV(name, safeRef, answers);

    // Generate inference CSV with model features
    const runFile = generatePatientInferenceCSV(safeRef, answers);

    // Run Python XGBoost inference
    const pyScript = path.join(process.cwd(), 'ML', 'run_models.py');
    const { stdout } = await execFileAsync('python', [pyScript, runFile, safeRef], {
      encoding: 'utf8',
      timeout: 30_000, // 30-second timeout to prevent hung processes
    });

    // Parse Python stdout into structured predictions
    const lines = stdout.trim().split('\n');
    const predictions = lines.map(line => {
      const [model, p, risk] = line.split(',');
      return { model, value: parseFloat(p), risk };
    });

    return new Response(JSON.stringify({ predictions }), { status: 200 });

  } catch (err) {
    console.error('[/api/submit-answers] POST error:', err);
    return new Response(JSON.stringify({ error: err.message }), { status: 500 });
  }
}
