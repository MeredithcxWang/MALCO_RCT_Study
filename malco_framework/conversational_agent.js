import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';


// Read API key from environment variables
const API_KEY = process.env.OPENAI_API_KEY || '';
if (!API_KEY) {
  console.warn('[MALCO] OPENAI_API_KEY is not set. LLM calls will fail.');
}

const BASE_URL = process.env.OPENAI_BASE_URL || '.web';
const MODEL_NAME = 'gemini-2.5-pro';

const openai = new OpenAI({ apiKey: API_KEY, baseURL: BASE_URL });

const RUN_DIR = path.join(process.cwd(), 'ML', 'run');

const SAFE_KEY_PATTERN = /^[a-zA-Z0-9_-]{1,128}$/;

function validateRef(ref) {
  if (!ref || typeof ref !== 'string' || !SAFE_KEY_PATTERN.test(ref)) {
    throw new Error(`Invalid reference key: "${ref}"`);
  }
  return ref;
}


// MALCO Prompt Library


const MALCO_PROMPTS = {
  // System role: defines the agent's medical background, safety boundaries, and communication style
  SYSTEM_IDENTITY: `You are an intelligent health assistant specifically designed for patients with heart failure. Your responsibility is to provide accurate, scientific, and easily understandable health advice and communication based on the patient's health data and prediction results. Your responses must be strictly grounded in knowledge trained and fine-tuned on clinical data, integrated with the patient's individual characteristics and risk prediction analysis. You should consistently maintain rigorous medical terminology while avoiding unverified or inappropriate advice for individual patients. Any health recommendations must be based on validated clinical guidelines and evidence-based medicine. During communication, you must ensure information privacy and security, refraining from providing potentially misleading information or sensitive personal data. If you encounter issues that are uncertain or beyond your professional expertise, you should advise the patient to consult professional medical personnel. Your goal is to help patients better understand their health status through interactive dialogue, promote effective self-management, reduce the risk of adverse health events, and improve the patient's quality of life.`,

  // Task instruction: defines how to process ML risk predictions
  INITIAL_TASK: `"The following are the specific details and baseline characteristics of the patient:

[Schema automatically populates from CSV, e.g.,]

[Feature 1]: [Value 1]

[Feature 2]: [Value 2]
...
[Feature N]: [Value N]"
"Acting as a senior physician, please complete the following initial tasks based on the detailed patient data and predictive analysis provided above:
1. Analysis and Recommendations: For each prediction result (excluding mortality), deeply analyze the key influencing factors in conjunction with their SHAP decision plots, and provide specific, actionable improvement suggestions.
2. Risk Communication: Conduct an implicit analysis of the mortality prediction. Avoid using sensitive words such as 'death.' Instead, focus on the insights from the decision plot to guide the patient toward proactive self-management.
3. Comprehensive Summary: Provide a comprehensive summary of the patient's health status, identify the health issues requiring the most immediate attention, and systematically propose integrated management strategies regarding lifestyle, diet, medication, and monitoring.
Ensure your language is professional yet accessible, avoiding overly complex medical jargon and repetitive content. At the end of your response, please append the standard disclaimer: 'This response is for reference only. If you have any concerns, please be sure to consult a physician for secondary verification.'
When the patient asks new questions subsequently, please provide concise, focused answers based on the existing analysis and dialogue context, without regenerating the complete report above.
Please begin your analysis."
`,
};

// Helper Functions

/**
 * Builds the initial user message containing CSV baseline data and prediction results.
 */
function buildInitialUserPrompt(ref) {
  const csvPath = path.join(RUN_DIR, `${ref}.csv`);
  const txtPath = path.join(RUN_DIR, `${ref}_results.txt`);

  let dataContent = '【患者基线数据】\n';
  if (fs.existsSync(csvPath)) {
    const lines = fs.readFileSync(csvPath, 'utf8').trim().split('\n');
    if (lines.length >= 2) {
      const headers = lines[0].split(',');
      const values = lines[1].split(',');
      headers.forEach((h, i) => {
        dataContent += `${h.replace(/"/g, '')}: ${values[i].replace(/"/g, '')}\n`;
      });
    }
  }

  let predictionContent = '\n【机器学习预测结果与 SHAP 分析】\n';
  if (fs.existsSync(txtPath)) {
    const results = fs.readFileSync(txtPath, 'utf8').split('\n').filter(l => l);
    results.forEach(line => {
      const [model, p, risk] = line.split(',');
      predictionContent += `- ${model} 风险概率: ${p} (等级: ${risk})\n`;
      // Reference SHAP force plot so the agent can cite feature contributions
      predictionContent += `  [SHAP Force Plot Source: ${ref}_${model}_force.png]\n`;
    });
  }

  return `${dataContent}${predictionContent}\n${MALCO_PROMPTS.INITIAL_TASK}`;
}

// Route Handlers

// Load or initialize chat history for a given reference key
export async function GET(request, { params }) {
  try {
    const ref = validateRef(params.ref);
    const chatFile = path.join(RUN_DIR, `${ref}_chat.json`);

    if (!fs.existsSync(RUN_DIR)) fs.mkdirSync(RUN_DIR, { recursive: true });

    // Return existing history if available
    if (fs.existsSync(chatFile)) {
      const history = JSON.parse(fs.readFileSync(chatFile, 'utf8'));
      return new Response(JSON.stringify({ messages: history }), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Initialize conversation with system prompt and patient context
    const history = [
      { role: 'system', content: MALCO_PROMPTS.SYSTEM_IDENTITY },
      { role: 'user', content: buildInitialUserPrompt(ref) },
    ];

    fs.writeFileSync(chatFile, JSON.stringify(history, null, 2), 'utf8');

    return new Response(JSON.stringify({ messages: history }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (err) {
    console.error('[MALCO GET] error:', err);
    return new Response(JSON.stringify({ error: err.message }), { status: 500 });
  }
}

// Append user message, call LLM, and stream the response
export async function POST(request, { params }) {
  try {
    const ref = validateRef(params.ref);
    const { message } = await request.json();
    const chatFile = path.join(RUN_DIR, `${ref}_chat.json`);
    const logFile = path.join(RUN_DIR, `${ref}_chatlog.txt`);

    // Load existing session
    if (!fs.existsSync(chatFile)) {
      return new Response(JSON.stringify({ error: 'Session missing' }), { status: 404 });
    }
    let history = JSON.parse(fs.readFileSync(chatFile, 'utf8'));

    // Append user message and log it
    history.push({ role: 'user', content: message });
    fs.appendFileSync(logFile, `USER: ${message}\n\n`, 'utf8');

    // Call LLM with low temperature for reliable medical advice
    const response = await openai.chat.completions.create({
      model: MODEL_NAME,
      messages: history,
      temperature: 0.2,
      stream: true,
    });

    // Stream response chunks to the client
    const encoder = new TextEncoder();
    const readable = new ReadableStream({
      async start(controller) {
        let fullResponse = '';
        const logChunks = []; // Buffer log writes to avoid per-chunk sync I/O

        try {
          for await (const chunk of response) {
            const content = chunk.choices[0]?.delta?.content || '';
            if (content) {
              fullResponse += content;
              logChunks.push(content);
              controller.enqueue(encoder.encode(content));
            }
          }
        } catch (streamErr) {
          // If the LLM stream fails mid-way, close gracefully
          console.error('[MALCO stream] error:', streamErr);
          controller.error(streamErr);
          return;
        }

        // Persist complete conversation and flush log in one write
        history.push({ role: 'assistant', content: fullResponse });
        fs.writeFileSync(chatFile, JSON.stringify(history, null, 2), 'utf8');
        fs.appendFileSync(logFile, `ASSISTANT: ${logChunks.join('')}\n\n`, 'utf8');
        controller.close();
      },
    });

    return new Response(readable, {
      headers: { 'Content-Type': 'text/event-stream' },
    });
  } catch (err) {
    console.error('[MALCO POST] error:', err);
    return new Response(JSON.stringify({ error: err.message }), { status: 500 });
  }
}
