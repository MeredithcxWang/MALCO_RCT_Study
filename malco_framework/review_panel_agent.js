import fs from 'fs';
import path from 'path';


const API_KEY = process.env.OPENAI_API_KEY || '';
if (!API_KEY) {
  console.warn('[Review] OPENAI_API_KEY is not set. Audit calls will fail.');
}

const BASE_URL = process.env.OPENAI_BASE_URL || '.web';

// Dual-review matrix configurations (Configuration A & B)
const CONFIG_A = [
  { key: 'deepseek', model: 'deepseek-r1' },
  { key: 'o3', model: 'o3-2025-04-16' },
  { key: 'claude3.7', model: 'claude-3-7-sonnet-20250219' }
];

const CONFIG_B = [
  { key: 'gpt5', model: 'gpt-5-2025-08-07' },
  { key: 'grok4', model: 'grok-4-2025-07-09' },
  { key: 'claudeOpus', model: 'claude-opus-4.1-2025-08-05' }
];

// Path traversal prevention
const SAFE_KEY_PATTERN = /^[a-zA-Z0-9_-]{1,128}$/;
function validateRef(ref) {
  if (!ref || typeof ref !== 'string' || !SAFE_KEY_PATTERN.test(ref)) {
    throw new Error(`Invalid reference key`);
  }
  return ref;
}

// Review Panel Prompt


const PROMPTS = {
  buildAuditPrompt: (chatLog) => {
    return (
      `You are a senior physician. Please review the following chat log between a patient and the AI to identify any erroneous information, false statements, or situations requiring intervention by a human doctor.
Please note that the patient has already been explicitly informed that they are conversing with an AI and advised to consult a physician if they have any doubts. Therefore, your review must focus strictly on the accuracy and appropriateness of the information, with a paramount emphasis on alerting against misinformation. Omissions of minor reminders are secondary; your absolute priority is to detect incorrect medical information. \n` +
      `Your review should include, but is not limited to, the following criteria:
1. Whether the provided information contains medical errors or inaccuracies;
2. Whether the information is potentially misleading or could cause patient misunderstanding;
3. Whether the given advice is comprehensive and clear, and whether critical health management information has been omitted;
4. Whether the AI provides advice that exceeds its capabilities or addresses issues requiring further evaluation by medical professionals (e.g., prescribing or recommending specific new medications). \n"
` +
      `When generating your output, your response must begin strictly with either 'Passed' or 'Requires Doctor Intervention'.
If doctor intervention is required, you must immediately follow with a clear and explicit justification for your decision.
` +
      `"=== Chat Log Begins ===\n${chatLog}\n === Chat Log Ends === \n Please provide your review results. \n"
` +
      `=== 聊天记录开始 ===\n${chatLog}\n=== 聊天记录结束 ===\n\n请给出审查结果。`
    );
  }
};

// Helper Functions

/**
 * Calls a specified LLM to perform the audit task.
 */
async function auditWithModel(modelName, prompt) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60_000);

  try {
    const resp = await fetch(`${BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: modelName,
        messages: [
          { role: 'system', content: 'You are a meticulous clinical safety auditor.' },
          { role: 'user', content: prompt }
        ],
        temperature: 0.2
      }),
      signal: controller.signal
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`Model ${modelName} returned ${resp.status}: ${txt}`);
    }
    const data = await resp.json();
    return data.choices[0].message.content;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Parses the model's first-line verdict into a normalized enum.
 */
function determineVerdict(line) {
  if (/需要(医生)?介入|need.?doctor|need.?physician/i.test(line)) {
    return 'need_doctor';
  } else if (/通过|pass/i.test(line)) {
    return 'pass';
  }
  return 'unknown';
}

// Route Handler

export async function POST(request, context) {
  const { params } = context;

  try {
    const ref = validateRef(params.ref);
    const runDir = path.join(process.cwd(), 'ML', 'run');
    const historyDir = path.join(process.cwd(), 'ML', 'history_patients', ref);
    const chatLogPath = path.join(runDir, `${ref}_chatlog.txt`);

    if (!fs.existsSync(chatLogPath)) {
      // Do not expose internal paths to the client
      return new Response(
        JSON.stringify({ error: 'Chat log not found for this reference key' }),
        { status: 404, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const chatLog = fs.readFileSync(chatLogPath, 'utf8');
    const prompt = PROMPTS.buildAuditPrompt(chatLog);

    if (!fs.existsSync(historyDir)) {
      fs.mkdirSync(historyDir, { recursive: true });
    }

    // Weighted random configuration selection (40% Config A, 60% Config B)
    const isConfigB = Math.random() < 0.6;
    const selectedConfig = isConfigB ? CONFIG_B : CONFIG_A;
    const configLabel = isConfigB ? 'Configuration_B' : 'Configuration_A';

    // Concurrent model audit calls
    const results = {};
    await Promise.all(
      selectedConfig.map(async ({ key, model }) => {
        const reply = await auditWithModel(model, prompt);
        const auditFile = path.join(historyDir, `${ref}_${key}_audit.txt`);
        fs.writeFileSync(auditFile, reply, 'utf8');

        const firstLine = reply.split(/\r?\n/)[0].trim();
        results[key] = {
          model_used: model,
          full_response: reply,
          verdict: determineVerdict(firstLine)
        };
      })
    );

    // One-Vote Veto Protocol: any "need_doctor" or "unknown" triggers review
    const isVetoed = Object.values(results).some(
      r => r.verdict === 'need_doctor' || r.verdict === 'unknown'
    );
    const finalDecision = isVetoed ? 'Need Physician Review' : 'Pass';

    // Archive all run files for this patient to history directory
    const files = fs.readdirSync(runDir);
    for (const file of files) {
      if (file.startsWith(ref)) {
        const src = path.join(runDir, file);
        const dst = path.join(historyDir, file);
        // Skip if already moved (handles concurrent race)
        if (fs.existsSync(src)) {
          fs.renameSync(src, dst);
        }
      }
    }

    return new Response(JSON.stringify({
      applied_configuration: configLabel,
      final_decision: finalDecision,
      agent_results: results
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (err) {
    console.error('[POST /api/chat/review] error:', err);
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
