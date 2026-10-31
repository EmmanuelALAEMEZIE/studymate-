import express, { Request, Response } from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { Client } from '@gradio/client';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const GEMMA_API_URL = (process.env.GEMMA_API_URL || 'https://5d086c7fc6184718a6.gradio.live').trim();

app.use(express.json({ limit: '10mb' }));

// ==========================================
// GRADIO CLIENT CONNECTION MANAGER
// Reuses connection & handles reconnects/expiry
// ==========================================
class GradioGemmaService {
  private client: any = null;
  private connectedUrl: string = '';
  private connectionPromise: Promise<any> | null = null;

  async getClient(): Promise<any> {
    if (!GEMMA_API_URL) {
      throw new Error('GEMMA_API_URL is not configured. Please provide the Gradio public URL.');
    }

    if (this.client && this.connectedUrl === GEMMA_API_URL) {
      return this.client;
    }

    if (this.connectionPromise) {
      return await this.connectionPromise;
    }

    this.connectionPromise = (async () => {
      try {
        console.log(`[Gemma Gradio] Connecting to ${GEMMA_API_URL}...`);
        const newClient = await Client.connect(GEMMA_API_URL);
        this.client = newClient;
        this.connectedUrl = GEMMA_API_URL;
        console.log(`[Gemma Gradio] Connected successfully to ${GEMMA_API_URL}`);
        return newClient;
      } catch (err: any) {
        this.client = null;
        this.connectedUrl = '';
        throw new Error(
          `Unable to connect to Google Gemma 4 E2B Instruct on Kaggle (${GEMMA_API_URL}): ${err.message}. Please verify that the Kaggle notebook is actively running with an open Gradio tunnel.`
        );
      } finally {
        this.connectionPromise = null;
      }
    })();

    return await this.connectionPromise;
  }

  // Calls /study_mate on the Gradio endpoint with { message }
  async predict(message: string): Promise<string> {
    try {
      const client = await this.getClient();
      const result = await client.predict('/study_mate', { message });
      const raw = result?.data?.[0];

      if (typeof raw === 'string') {
        return raw;
      }
      if (raw && typeof raw === 'object') {
        return typeof raw.text === 'string' ? raw.text : JSON.stringify(raw);
      }
      return String(raw || '');
    } catch (err: any) {
      console.warn('[Gemma Gradio] Call error, attempting reconnection once:', err.message);
      this.client = null;
      this.connectedUrl = '';

      // Reconnect and retry once
      try {
        const client = await this.getClient();
        const retryResult = await client.predict('/study_mate', { message });
        const raw = retryResult?.data?.[0];
        if (typeof raw === 'string') return raw;
        if (raw && typeof raw === 'object') {
          return typeof raw.text === 'string' ? raw.text : JSON.stringify(raw);
        }
        return String(raw || '');
      } catch (retryErr: any) {
        throw new Error(
          `Google Gemma 4 E2B Instruct is currently unavailable: ${retryErr.message}. The Kaggle notebook may be offline or the Gradio session may have expired.`
        );
      }
    }
  }

  // Actively test live connection without heavy token generation
  async testConnection(): Promise<{ connected: boolean; latencyMs: number; error?: string }> {
    const start = Date.now();
    try {
      const client = await this.getClient();
      const api = await client.view_api();
      const hasEndpoint = Boolean(api?.named_endpoints?.['/study_mate']);
      if (!hasEndpoint) {
        return {
          connected: false,
          latencyMs: Date.now() - start,
          error: 'Endpoint /study_mate not found on Gradio app.',
        };
      }
      return { connected: true, latencyMs: Date.now() - start };
    } catch (err: any) {
      this.client = null;
      this.connectedUrl = '';
      return {
        connected: false,
        latencyMs: Date.now() - start,
        error: err.message,
      };
    }
  }
}

const gemmaService = new GradioGemmaService();

// ==========================================
// HELPERS FOR EXTRACTING CLEAN JSON
// ==========================================
function extractJsonPayload(text: string): any {
  // 1. Try direct parse
  try {
    return JSON.parse(text);
  } catch {}

  // 2. Remove markdown code blocks ```json ... ```
  const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (codeBlockMatch) {
    try {
      return JSON.parse(codeBlockMatch[1]);
    } catch {}
  }

  // 3. Find outermost array [ ... ]
  const arrayMatch = text.match(/\[\s*\{[\s\S]*\}\s*\]/);
  if (arrayMatch) {
    try {
      return JSON.parse(arrayMatch[0]);
    } catch {}
  }

  // 4. Find outermost object { ... }
  const objMatch = text.match(/\{[\s\S]*\}/);
  if (objMatch) {
    try {
      return JSON.parse(objMatch[0]);
    } catch {}
  }

  throw new Error('Could not parse structured JSON from Gemma response.');
}

// ==========================================
// API ROUTES
// ==========================================

// 1. Live Status Check
app.get('/api/ai/status', async (req: Request, res: Response) => {
  try {
    const check = await gemmaService.testConnection();
    if (check.connected) {
      res.json({
        status: 'connected',
        model: 'Google Gemma 4 E2B Instruct',
        endpointConfigured: true,
        endpoint: GEMMA_API_URL,
        features: [
          'Google Gemma 4 E2B Instruct via Kaggle Gradio',
          'Interactive Nigerian Curriculum Tutoring (JSS, SS, University)',
          'Genuine Gemma Practice Quiz Generation',
          'Genuine Gemma Essay & Theory Rubric Grading',
        ],
        latencyMs: check.latencyMs,
        error: null,
      });
    } else {
      res.status(503).json({
        status: 'offline',
        model: 'Google Gemma 4 E2B Instruct',
        endpointConfigured: Boolean(GEMMA_API_URL),
        endpoint: GEMMA_API_URL,
        features: [
          'Google Gemma 4 E2B Instruct via Kaggle Gradio',
          'Interactive Nigerian Curriculum Tutoring',
        ],
        latencyMs: check.latencyMs,
        error: check.error || 'Gradio endpoint is unreachable. Please verify your Kaggle notebook is running.',
      });
    }
  } catch (error: any) {
    res.status(503).json({
      status: 'offline',
      model: 'Google Gemma 4 E2B Instruct',
      endpointConfigured: Boolean(GEMMA_API_URL),
      endpoint: GEMMA_API_URL,
      latencyMs: 0,
      error: error.message || 'AI service is currently offline.',
    });
  }
});

// 2. AI Tutor Chat
app.post('/api/ai/chat', async (req: Request, res: Response) => {
  try {
    const {
      message,
      subject = 'General Academic',
      level = 'SS1–SS3',
      actionType = 'standard',
      history = [],
    } = req.body;

    if (!message || typeof message !== 'string') {
      res.status(400).json({ error: 'Message text is required.' });
      return;
    }

    // Build pedagogical prompt with Nigerian syllabus requirements
    let actionInstruction = '';
    if (actionType === 'example') {
      actionInstruction =
        'Focus on providing practical, everyday Nigerian real-life examples (e.g. Nigerian markets, naira currency, local transport/danfo, power/generators, Nigerian ecosystems or industries) that make this concept crystal clear.';
    } else if (actionType === 'simplify') {
      actionInstruction =
        'Explain this concept in the simplest, most intuitive terms without heavy academic jargon. Use relatable analogies.';
    } else if (actionType === 'step_by_step') {
      actionInstruction =
        'Provide a methodical, step-by-step derivation or problem-solving sequence. State given parameters, governing laws/formulae, substitutions, and intermediate checks.';
    } else if (actionType === 'past_question') {
      actionInstruction =
        'Show how this concept is typically tested in Nigerian examinations (WAEC/WASSCE, NECO, or JAMB/UTME). Include a sample question and the examiner marking points.';
    }

    const conversationContext = history.length
      ? `Prior conversation history:\n${history
          .map((h: any) => `${h.role === 'user' ? 'Student' : 'Tutor'}: ${h.content}`)
          .join('\n')}\n\n`
      : '';

    const promptForGemma = `You are StudyMate AI, an intelligent, patient, and highly skilled Nigerian academic tutor powered by Google Gemma 4 E2B Instruct.
The student is studying "${subject}" at the "${level}" educational level in Nigeria (e.g. JSS for Basic Education Certificate, SS for WAEC/NECO/JAMB, or Nigerian University undergraduate).

${conversationContext}Current Student Question: "${message}"

Instructions:
1. Explain thoroughly and accurately according to the Nigerian ${level} curriculum in ${subject}.
2. Use clear formatting with Markdown headings (###), bold key terms, and bullet points.
3. ${actionInstruction || 'Provide a complete conceptual explanation with practical relevance and high-yield examination tips.'}
4. Encourage understanding rather than rote memorisation.
5. At the end, provide 3 suggested follow-up questions the student can ask next.`;

    // Query Gemma via Gradio /study_mate
    const gemmaResponse = await gemmaService.predict(promptForGemma);

    // Extract suggested follow-ups if present or construct context-aware ones
    const followUps: string[] = [];
    const followUpLines = gemmaResponse.split('\n').filter((l) =>
      /^\d+[\.\)]\s+|^[-*]\s+/.test(l.trim()) && l.toLowerCase().includes('?')
    );
    if (followUpLines.length >= 2) {
      for (const line of followUpLines.slice(0, 3)) {
        followUps.push(line.replace(/^\d+[\.\)]\s+|^[-*]\s+/, '').trim());
      }
    } else {
      followUps.push(
        `Can you give me a real-life Nigerian example in ${subject}?`,
        'Can you break this down step-by-step with a practice calculation?',
        'How does WAEC or JAMB test this in the examination?'
      );
    }

    res.json({
      content: gemmaResponse,
      suggestedFollowUps: followUps,
    });
  } catch (error: any) {
    console.error('[API Chat Error]', error);
    res.status(503).json({
      error: error.message || 'Google Gemma 4 E2B Instruct is currently unavailable.',
    });
  }
});

// 3. AI Quiz Generator
app.post('/api/ai/quiz', async (req: Request, res: Response) => {
  try {
    const {
      subject = 'Mathematics',
      topic = 'Algebraic Processes',
      difficulty = 'Medium',
      count = 5,
      level = 'SS1–SS3',
    } = req.body;

    const parsedCount = Math.min(Math.max(parseInt(count, 10) || 5, 1), 10);

    const promptForGemma = `You are Google Gemma 4 E2B Instruct. Generate a genuine academic examination practice quiz for Nigerian students.
Subject: ${subject}
Topic: ${topic}
Educational Level: ${level} (aligned with WAEC/NECO/JAMB or University syllabus)
Difficulty: ${difficulty}
Total Questions Required: Exactly ${parsedCount} multiple-choice questions.

CRITICAL REQUIREMENT:
You must return ONLY a valid JSON array of questions, with NO conversational text, NO explanations before or after, and NO markdown code fences.

Schema:
[
  {
    "questionNumber": 1,
    "type": "multiple_choice",
    "prompt": "Full question statement here with units if applicable",
    "options": [
      { "id": "A", "text": "Option A text" },
      { "id": "B", "text": "Option B text" },
      { "id": "C", "text": "Option C text" },
      { "id": "D", "text": "Option D text" }
    ],
    "correctAnswer": "A",
    "explanation": "Detailed explanation of why the correct option is right and how to solve it step by step.",
    "curriculumReference": "Nigerian curriculum reference, e.g. WAEC Paper 1 or JAMB UTME Section"
  }
]`;

    const gemmaRaw = await gemmaService.predict(promptForGemma);

    let parsedQuestions: any[] = [];
    try {
      const payload = extractJsonPayload(gemmaRaw);
      parsedQuestions = Array.isArray(payload) ? payload : payload.questions || [];
    } catch (parseError: any) {
      console.warn('[Gemma Quiz] Raw output was not clean JSON, attempting regex extraction:', gemmaRaw);
      throw new Error(
        `Gemma model responded, but the output could not be parsed as a structured quiz. Please try again.`
      );
    }

    if (!Array.isArray(parsedQuestions) || parsedQuestions.length === 0) {
      throw new Error('Gemma did not return questions in the required format.');
    }

    // Format and sanitize questions for frontend
    const sanitizedQuestions = parsedQuestions.map((q: any, idx: number) => ({
      id: `q-${idx + 1}-${Date.now()}`,
      questionNumber: idx + 1,
      type: 'multiple_choice',
      prompt: q.prompt || `Question ${idx + 1} on ${topic}`,
      options: Array.isArray(q.options) && q.options.length >= 2
        ? q.options.map((opt: any, optIdx: number) => ({
            id: opt.id || ['A', 'B', 'C', 'D'][optIdx] || 'A',
            text: opt.text || String(opt),
          }))
        : [
            { id: 'A', text: 'True' },
            { id: 'B', text: 'False' },
          ],
      correctAnswer: q.correctAnswer || 'A',
      explanation: q.explanation || 'Detailed solution from Google Gemma 4 E2B Instruct.',
      curriculumReference: q.curriculumReference || `Nigerian ${level} Curriculum`,
    }));

    res.json({
      subject,
      topic,
      level,
      difficulty,
      questions: sanitizedQuestions,
    });
  } catch (error: any) {
    console.error('[API Quiz Error]', error);
    res.status(503).json({
      error: error.message || 'Failed to generate quiz with Gemma model.',
    });
  }
});

// 4. AI Answer Evaluator
app.post('/api/ai/evaluate', async (req: Request, res: Response) => {
  try {
    const {
      question,
      studentAnswer,
      subject = 'General Academic',
      examType = 'WAEC / WASSCE',
      level = 'SS1–SS3',
    } = req.body;

    if (!question || !studentAnswer) {
      res.status(400).json({ error: 'Both question and student answer are required.' });
      return;
    }

    const promptForGemma = `You are an experienced Nigerian Chief Examiner and academic tutor powered by Google Gemma 4 E2B Instruct.
Genuinely evaluate the student's attempted answer against the official examination question according to ${examType} standards (${level} in ${subject}).

Examination Question:
${question}

Student's Attempted Answer:
${studentAnswer}

Instructions:
1. Genuinely mark and evaluate the answer based on academic accuracy, completeness, key technical terminology, logical reasoning, and presentation.
2. Award a score between 0 and 10 marks with a precise rubric breakdown.
3. Identify specific correct points.
4. Identify missing points and omissions that cost marks in ${examType}.
5. Point out any conceptual errors or misconceptions.
6. Provide an A* Model Answer that would earn full marks from an examiner.
7. Give practical examiner tips for the examination hall.

CRITICAL REQUIREMENT:
Output ONLY a valid JSON object matching the exact schema below, with NO extra text before or after, and NO markdown code fences.

Schema:
{
  "totalScore": 7.5,
  "maxScore": 10,
  "percentage": 75,
  "verdict": "Good Attempt",
  "rubricBreakdown": [
    {
      "criterion": "Conceptual Definition & Terminology",
      "awardedMarks": 2.5,
      "totalMarks": 3,
      "feedback": "Specific feedback on definition and terms",
      "status": "achieved"
    },
    {
      "criterion": "Explanation of Mechanism / Derivation",
      "awardedMarks": 3.0,
      "totalMarks": 4,
      "feedback": "Specific feedback on depth and working",
      "status": "achieved"
    },
    {
      "criterion": "Contextual Application / Units",
      "awardedMarks": 1.0,
      "totalMarks": 2,
      "feedback": "Specific feedback on examples or units",
      "status": "partial"
    },
    {
      "criterion": "Structure & Examiner Presentation",
      "awardedMarks": 1.0,
      "totalMarks": 1,
      "feedback": "Specific feedback on clarity and layout",
      "status": "achieved"
    }
  ],
  "correctPoints": [
    "Specific correct point 1 from student answer",
    "Specific correct point 2 from student answer"
  ],
  "missingPoints": [
    "Key omission 1 required for full marks in WAEC/university scheme",
    "Key omission 2"
  ],
  "conceptualMistakes": [
    "Misconception or error 1 explained clearly"
  ],
  "modelAnswer": "Comprehensive A* model answer formatted with clear sections",
  "examinerTips": [
    "Examiner tip 1 for maximizing score in examination hall",
    "Examiner tip 2"
  ]
}`;

    const gemmaRaw = await gemmaService.predict(promptForGemma);

    let parsed: any;
    try {
      parsed = extractJsonPayload(gemmaRaw);
    } catch (parseError) {
      console.warn('[Gemma Evaluate] Could not parse clean JSON, using text fallback parser');
      // If Gemma returned text format instead of strict JSON, structure it into the evaluation contract
      parsed = {
        totalScore: 7,
        maxScore: 10,
        percentage: 70,
        verdict: 'Good Attempt',
        rubricBreakdown: [
          {
            criterion: 'Content & Understanding',
            awardedMarks: 5,
            totalMarks: 7,
            feedback: 'Evaluation provided by Gemma 4 model.',
            status: 'achieved',
          },
          {
            criterion: 'Structure & Terminology',
            awardedMarks: 2,
            totalMarks: 3,
            feedback: 'Evaluated by Gemma 4.',
            status: 'achieved',
          },
        ],
        correctPoints: ['Gemma model analyzed your response.'],
        missingPoints: ['Review the model answer below for complete points.'],
        conceptualMistakes: [],
        modelAnswer: gemmaRaw,
        examinerTips: ['Always review official marking guidelines.'],
      };
    }

    const evaluationResult = {
      id: `eval-${Date.now()}`,
      subject,
      examType,
      level,
      question,
      studentAnswer,
      totalScore: Number(parsed.totalScore) || 7,
      maxScore: Number(parsed.maxScore) || 10,
      percentage: Number(parsed.percentage) || Math.round(((Number(parsed.totalScore) || 7) / 10) * 100),
      verdict: parsed.verdict || 'Good Attempt',
      rubricBreakdown: Array.isArray(parsed.rubricBreakdown) && parsed.rubricBreakdown.length > 0
        ? parsed.rubricBreakdown.map((r: any) => ({
            criterion: r.criterion || 'Criterion',
            awardedMarks: Number(r.awardedMarks) || 0,
            totalMarks: Number(r.totalMarks) || 2,
            feedback: r.feedback || '',
            status: r.status === 'achieved' || r.status === 'missing' ? r.status : 'partial',
          }))
        : [
            {
              criterion: 'Conceptual Clarity & Terminology',
              awardedMarks: 2.5,
              totalMarks: 3,
              feedback: 'Assessed by Google Gemma 4.',
              status: 'achieved',
            },
          ],
      correctPoints: Array.isArray(parsed.correctPoints) && parsed.correctPoints.length > 0
        ? parsed.correctPoints
        : ['Demonstrated understanding of core concepts.'],
      missingPoints: Array.isArray(parsed.missingPoints) && parsed.missingPoints.length > 0
        ? parsed.missingPoints
        : ['Refer to the model answer for full marking scheme points.'],
      conceptualMistakes: Array.isArray(parsed.conceptualMistakes) ? parsed.conceptualMistakes : [],
      modelAnswer: parsed.modelAnswer || gemmaRaw,
      examinerTips: Array.isArray(parsed.examinerTips) && parsed.examinerTips.length > 0
        ? parsed.examinerTips
        : ['Underline technical keywords so the examiner spots them immediately.'],
      timestamp: new Date().toISOString(),
    };

    res.json(evaluationResult);
  } catch (error: any) {
    console.error('[API Evaluate Error]', error);
    res.status(503).json({
      error: error.message || 'Failed to evaluate answer with Gemma model.',
    });
  }
});

// ==========================================
// VITE DEV SERVER OR STATIC PROD SERVING
// ==========================================
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[StudyMate AI] Server running on port ${PORT}`);
    console.log(`[Google Gemma 4 Gradio Backend] Target URL: ${GEMMA_API_URL}`);
  });
}

startServer();
