import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Set up large JSON payload limits for webcam base64 uploads
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

// Helper to get or lazily check for Gemini client
let aiInstance: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is not set. Please add it in the Secrets panel.");
  }
  if (!aiInstance) {
    aiInstance = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      }
    });
  }
  return aiInstance;
}

// Helper to evaluate user responses for strict domain compliance, context relevance, and silent grammar corrections
async function evaluateUserResponse(
  userText: string,
  selectedDomain: string,
  specificQuestion: string
): Promise<{ isRelevant: boolean; correctedText: string }> {
  try {
    const ai = getGeminiClient();
    const prompt = `
      You are a strict, disciplined content evaluator and grammar expert.
      Your task is to:
      1. Identify and check if the user's response is directly related to the active domain: "${selectedDomain}".
      2. If a specific question or topic was asked: "${specificQuestion || "None"}", check if the response is directly related to solving or answering that question or topic.
      3. If the user's response is irrelevant, off-topic, gibberish, or does NOT address the active domain/question, mark "isRelevant" as false.
      4. If the response is relevant:
         - Silently correct any grammatical, spelling, and vocabulary errors.
         - Enhance its clarity, structural flow, and professionalism.
         - Maintain a highly natural, fluent English speaker output.
         - Keep the semantic meaning and intent identical.
      
      Respond strictly with a valid JSON object matching this schema:
      {
        "isRelevant": boolean,
        "correctedText": string // The corrected and polished version of the user's text. If not relevant, leave this empty.
      }

      User Response to evaluate:
      "${userText}"
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            isRelevant: { type: Type.BOOLEAN },
            correctedText: { type: Type.STRING }
          },
          required: ["isRelevant", "correctedText"]
        }
      }
    });

    const parsed = JSON.parse(response.text || "{}");
    return {
      isRelevant: typeof parsed.isRelevant === "boolean" ? parsed.isRelevant : true,
      correctedText: parsed.correctedText || userText
    };
  } catch (error) {
    console.error("Relevance check system error:", error);
    // Safe fallback so service is robust
    return { isRelevant: true, correctedText: userText };
  }
}

// 1. Speach Analysis Endpoint
// Evaluates pacing, grammar, vocabulary, sentiment, and provides actionable improvements.
app.post("/api/analyze-speech", async (req, res) => {
  try {
    const { text, duration, role, context, selectedDomain, questionText } = req.body;
    if (!text || typeof text !== "string") {
      return res.status(400).json({ error: "Speech text is empty or invalid" });
    }

    // Strict Domain & Relevance check with silent grammar correction
    const evaluation = await evaluateUserResponse(text, selectedDomain || "Technical", questionText || "");
    if (!evaluation.isRelevant) {
      return res.json({
        isRelevant: false,
        error: "Your response is not related to the selected topic. Please speak only about the asked question and stay within the domain."
      });
    }

    const processedText = evaluation.correctedText || text;

    const ai = getGeminiClient();
    const cleanDuration = Number(duration) || 30; // seconds
    const minutes = cleanDuration / 60;
    const wordCount = processedText.split(/\s+/).filter(Boolean).length;
    const computedWpm = Math.round(wordCount / (minutes || 1));

    // Leverage Gemini-3.5-flash to analyze the transcript
    const prompt = `
      You are an expert Speech Coach. Analyze the following transcribed text from a public speaking session.
      The user is speaking in the role of: "${role || "General Practice"}" ${context ? `and the context is: "${context}"` : ""}.
      They spoke about ${wordCount} words over a period of ${cleanDuration} seconds (Calculated WPM: ${computedWpm}).

      Review the transcript carefully and perform these speech coach operations:
      1. Score the clarity level from 0 to 100.
      2. Analyze the pacing (WPM). Assess if it is "Too Slow" (< 100 WPM), "Optimal" (100 - 150 WPM), or "Too Fast" (> 150 WPM).
      3. Identify the number of filer words used (count instances of "um", "uh", "like", "you know", "actually", "basically", etc.) as present in the transcribed text, or note general hesitation markers.
      4. Note 2 to 3 grammar or articulation improvements with before-and-after examples.
      5. Determine the voice tone and emotional sentiment (e.g. Confident, Nervous, Hesitant, Enthusiastic, Monotone).
      6. Provide 3 specific, positive highlights from their speech.
      7. Provide 3 actionable recommendations to improve performance next time.

      Transcribed text:
      "${processedText}"

      Respond strictly with a valid JSON object matching this schema structure:
      {
        "clarityScore": number, // out of 100
        "pacingWpm": number,
        "pacingAssessment": "Too Slow" | "Optimal" | "Too Fast",
        "fillerWordCount": number,
        "fillerWordsFound": string[], // array of unique filler words identified
        "toneSentiment": string, // e.g. "Confident & Articulate"
        "grammarFeedback": [
          { "original": string, "suggested": string, "why": string }
        ],
        "highlights": string[], // 3 items
        "recommendations": string[] // 3 items
      }
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            clarityScore: { type: Type.NUMBER },
            pacingWpm: { type: Type.NUMBER },
            pacingAssessment: { type: Type.STRING },
            fillerWordCount: { type: Type.NUMBER },
            fillerWordsFound: { type: Type.ARRAY, items: { type: Type.STRING } },
            toneSentiment: { type: Type.STRING },
            grammarFeedback: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  original: { type: Type.STRING },
                  suggested: { type: Type.STRING },
                  why: { type: Type.STRING }
                },
                required: ["original", "suggested", "why"]
              }
            },
            highlights: { type: Type.ARRAY, items: { type: Type.STRING } },
            recommendations: { type: Type.ARRAY, items: { type: Type.STRING } }
          },
          required: [
            "clarityScore",
            "pacingWpm",
            "pacingAssessment",
            "fillerWordCount",
            "fillerWordsFound",
            "toneSentiment",
            "grammarFeedback",
            "highlights",
            "recommendations"
          ]
        }
      }
    });

    const resultText = response.text || "{}";
    res.json(JSON.parse(resultText));
  } catch (error: any) {
    console.error("Speech analysis error:", error);
    res.status(500).json({ error: error.message || "Failed to analyze speech" });
  }
});

// 2. Emotion Webcam Analysis Endpoint
// Instructs Gemini to evaluate body language, gaze contact, facial ticks and vocal/visual alignment from a base64 shot.
app.post("/api/analyze-face", async (req, res) => {
  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) {
      return res.status(400).json({ error: "Webcam image is required for visual assessment" });
    }

    const ai = getGeminiClient();

    // Clean base64 string
    const match = imageBase64.match(/^data:(image\/\w+);base64,(.+)$/);
    if (!match) {
      return res.status(400).json({ error: "Invalid image format. Expected visual base64." });
    }

    const mimeType = match[1];
    const data = match[2];

    const imagePart = {
      inlineData: {
        mimeType,
        data,
      },
    };

    const promptText = `
      You are an expert communication coach analyzing a speaker's presentation composure.
      Evaluate the speaker's facial expression, posture, eye contact, and estimated emotional feedback from this camera snapshot taken during standard presentation practice.

      Return a response strictly in JSON format with the following keys:
      {
        "primaryEmotion": string, // e.g. "Confident", "Anxious", "Focused", "Neutral", "Uncertain"
        "emotionExplanation": string, // short description of why this was inferred
        "eyeContactLevel": "Direct" | "Inattentive" | "Wandering" | "Poor",
        "eyeContactFeedback": string, // 1-2 sentences of professional advice
        "postureFeedback": string, // feedback on head tilt, shoulder comfort, frame focus
        "presenterScore": number, // out of 100
        "microexpressionNotes": string // notes about mouth tightness, brow tension, or positive smiling engagement
      }
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: [imagePart, { text: promptText }],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            primaryEmotion: { type: Type.STRING },
            emotionExplanation: { type: Type.STRING },
            eyeContactLevel: { type: Type.STRING },
            eyeContactFeedback: { type: Type.STRING },
            postureFeedback: { type: Type.STRING },
            presenterScore: { type: Type.NUMBER },
            microexpressionNotes: { type: Type.STRING }
          },
          required: [
            "primaryEmotion",
            "emotionExplanation",
            "eyeContactLevel",
            "eyeContactFeedback",
            "postureFeedback",
            "presenterScore",
            "microexpressionNotes"
          ]
        }
      }
    });

    const resultText = response.text || "{}";
    res.json(JSON.parse(resultText));
  } catch (error: any) {
    console.error("Visual webcam analysis error:", error);
    res.status(500).json({ error: error.message || "Failed to analyze presenter's image" });
  }
});

// 3. Resume-based Interview Mode Endpoint
// Feeds resume and optional target Job Description context into Gemini, generating custom interviewer cues.
app.post("/api/interview/chat", async (req, res) => {
  try {
    const { role, resume, jobDescription, messages, interviewerPersona, isHiddenJudgeMode, selectedDomain } = req.body;
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: "Interview history messages are required" });
    }

    // Identify standard domain and perform strict relevance & grammar check on user replies
    if (messages.length > 0) {
      const userMsg = messages[messages.length - 1];
      if (userMsg && userMsg.role === "user") {
        const lastQuestion = messages[messages.length - 2]?.text || "";
        const evaluation = await evaluateUserResponse(userMsg.text, selectedDomain || "Technical", lastQuestion);
        if (!evaluation.isRelevant) {
          return res.json({
            isRelevant: false,
            error: "Your response is not related to the selected topic. Please speak only about the asked question and stay within the domain."
          });
        }
        // Silently correct grammatical structure and replace input before feeding to core prompt
        userMsg.text = evaluation.correctedText || userMsg.text;
      }
    }

    const ai = getGeminiClient();

    const systemInstruction = `
      You are an elite, highly professional Corporate Interviewer and recruiter.
      The mock interview domain is STRICTLY: "${selectedDomain || "Technical"}". All interactions, questions, and responses MUST STRICTLY belong and adhere to this domain.
      The user is practicing a mock interview for the role of: "${role || "General Candidate"}".
      ${resume ? `Candidate's Resume/Experience matches:\n"""\n${resume}\n"""\n` : "The candidate did not upload details, assume standard competitive background."}
      ${jobDescription ? `The target Job Description matches:\n"""\n${jobDescription}\n"""\n` : ""}

      Interviewer Persona: "${interviewerPersona || "Friendly HR"}"
      Adopt the profile of this interviewer throughout questions, tone, and difficulty:
      - "Friendly HR": Warm, polite, conversational, encouraging, asks introductory, emotional intellect, and core culture-fit questions. Low difficulty. (Example: "Tell me about yourself.")
      - "Strict Hiring Manager": Impatient, challenging, direct, highly detail-focused, drill down on metrics, asks high-pressure situational queries. High difficulty. (Example: "Why was there a drop in that metric and what did you personally do?")
      - "FAANG Technical Recruiter": Fast-paced, highly metrics/scale-oriented, focuses on high scalability, systemic thinking, agile execution, technical design benchmarks. Medium-High difficulty. (Example: "How would you design a scalable system for ten million concurrent users?")
      - "Startup Founder": Ambitious, highly energetic, scrappy/multi-functional questions centering around ownership, "Why should I bet my company on you?", urgency, autonomy. Extremely High difficulty. (Example: "Why should I bet my company on you? What will you ship in week one?")
      - "TED Talk Audience": Focuses on storytelling, visual analogies, inspiring public appeal, clarity, charisma, narrative pacing. Medium difficulty. (Example: "How would you explain your primary research to a room of high school students?")

      Hidden Judge Mode Enabled: ${isHiddenJudgeMode ? "YES" : "NO"}
      If Hidden Judge Mode is enabled:
      You behave like a direct, sharp, opinionated Hackathon panel judge or Venture Capitalist investor evaluating a high-stakes product/tech presentation. Ask questions about team product feasibility, technical innovation depth, monetization/security, and pitch confidence. Keep questions highly punchy and direct. (Example: "Why should your team win? Make it clear in ten seconds.")

      You are conducting a dynamic, interactive mock interview. 
      Your goals are:
      1. Ask challenging, high-signal, relevant, or situational STAR questions that fit the selected persona.
      2. If there are previous messages, analyze the candidate's last answer closely. Evaluate if they mentioned Situation, Task, Action, and Result (the STAR framework).
      3. Perform speech evaluation and split the user's last answer into 3 to 6 logical segments representing Strong, Average or Weak points (stutters, filler words, or empty claims) to build a visual highlighted heatmap.
      4. Suggest highly professional improved rephrasings.
      5. Provide Hidden Judge Scores (out of 10) for Innovation, Communication, Technical Depth, and Confidence if the user is practicing.
      6. Formulate a challenging dynamic FOLLOW-UP question that relates to what they just said in their prior message, rather than a boilerplate static question.

      Respond strictly with a JSON object matching this schema:
      {
        "lastAnswerFeedback": {
          "score": number, // out of 10 (0 if first turn)
          "strengths": string,
          "gaps": string,
          "suggestedPhrasing": string,
          "liveIndicator": "🟢 Good pace" | "🟡 Too many fillers" | "🔴 Long pause / hesitation",
          "starEvaluation": {
            "situation": string, // feedback on their Situation statement (e.g. "Excellent setting of CTR context" or "Not provided")
            "task": string,      // feedback on Task statement
            "action": string,    // feedback on Action statement
            "result": string     // feedback on Result / metrics statement
          },
          "heatmapChunks": [
            { "text": string, "category": "strong" | "average" | "weak", "explanation": string }
          ],
          "judgeScores": {
            "innovation": number, // 1-10
            "communication": number, // 1-10
            "technicalDepth": number, // 1-10
            "confidence": number, // 1-10
            "judgeCritique": string
          }
        } | null, // null if this is the start of the interview (history is empty)
        "nextQuestion": string, // the dynamic and challenging subsequent question they must answer
        "interviewStage": string // e.g. "Introduction", "STAR Assessment", "Technical Deep-dive", "Hidden Judge Evaluation", "Conclusion"
      }
    `;

    // Process chat formats for Gemini instruction representation.
    // The messages from client look like: { role: "user" | "assistant", text: "..." }
    const formattedHistory = messages.map(msg => {
      const actor = msg.role === "user" ? "Candidate" : "Interviewer";
      return `${actor}: ${msg.text}`;
    }).join("\n");

    const promptText = `
      Here is the interview history context:
      ${formattedHistory || "[START OF INTERVIEW. No previous messages. Greet the user or pose an initial question based on their selected persona and target resume/job description.]"}

      Synthesize feedback on their last reply if any, divide their response into strong/weak/average heatmap chunks, score all vectors, and formulate a challenging subsequent follow-up question.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: promptText,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            lastAnswerFeedback: {
              type: Type.OBJECT,
              nullable: true,
              properties: {
                score: { type: Type.NUMBER },
                strengths: { type: Type.STRING },
                gaps: { type: Type.STRING },
                suggestedPhrasing: { type: Type.STRING },
                liveIndicator: { type: Type.STRING },
                starEvaluation: {
                  type: Type.OBJECT,
                  properties: {
                    situation: { type: Type.STRING },
                    task: { type: Type.STRING },
                    action: { type: Type.STRING },
                    result: { type: Type.STRING }
                  },
                  required: ["situation", "task", "action", "result"]
                },
                heatmapChunks: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      text: { type: Type.STRING },
                      category: { type: Type.STRING },
                      explanation: { type: Type.STRING }
                    },
                    required: ["text", "category", "explanation"]
                  }
                },
                judgeScores: {
                  type: Type.OBJECT,
                  properties: {
                    innovation: { type: Type.NUMBER },
                    communication: { type: Type.NUMBER },
                    technicalDepth: { type: Type.NUMBER },
                    confidence: { type: Type.NUMBER },
                    judgeCritique: { type: Type.STRING }
                  },
                  required: ["innovation", "communication", "technicalDepth", "confidence", "judgeCritique"]
                }
              },
              required: ["score", "strengths", "gaps", "suggestedPhrasing", "liveIndicator", "starEvaluation", "heatmapChunks", "judgeScores"]
            },
            nextQuestion: { type: Type.STRING },
            interviewStage: { type: Type.STRING }
          },
          required: ["lastAnswerFeedback", "nextQuestion", "interviewStage"]
        }
      }
    });

    const resultText = response.text || "{}";
    res.json(JSON.parse(resultText));
  } catch (error: any) {
    console.error("Resume interview error:", error);
    res.status(500).json({ error: error.message || "Interview engine error" });
  }
});

// 4. Debate Mode Endpoint
// Simulates an articulate debate opponent with user side counter-arguments.
app.post("/api/debate/chat", async (req, res) => {
  try {
    const { topic, userSide, messages, selectedDomain } = req.body; // userSide: "pro" | "con"
    if (!topic) {
      return res.status(400).json({ error: "Debate topic is required" });
    }
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: "Debate transcripts are required" });
    }

    // Verify debate speaker response is fully aligned with domain and question topic
    if (messages.length > 0) {
      const userMsg = messages[messages.length - 1];
      if (userMsg && userMsg.role === "user") {
        const evaluation = await evaluateUserResponse(userMsg.text, selectedDomain || "Technical", topic);
        if (!evaluation.isRelevant) {
          return res.json({
            isRelevant: false,
            error: "Your response is not related to the selected topic. Please speak only about the asked question and stay within the domain."
          });
        }
        // Silently execute grammar correction before analyzing point
        userMsg.text = evaluation.correctedText || userMsg.text;
      }
    }

    const ai = getGeminiClient();

    const systemInstruction = `
      You are an elite, championship-winning competitive Academic Debater.
      The debate domain is strictly: "${selectedDomain || "Technical"}". All counter arguments and academic forensic evaluations must align with this domain.
      You are engaged in a timed, formal, high-caliber debate about: "${topic}".
      The user is arguing: "${userSide.toUpperCase()}".
      Therefore, YOU MUST ADVOCATE THE REVERSE OPPOSING SIDE: "${userSide === "pro" ? "CON" : "PRO"}".

      Rule of Engagement:
      1. Challenge the user's specific arguments. Spot logical fallacies, demand structural or empirical clarity, and formulate an elegant, compelling, and incredibly persuasive counter-argument.
      2. Provide immediate constructive forensic debate commentary of their last speech (clarity index, logical resilience score out of 10, rhetorical fallacies identified, and specific suggestions to boost their counter-strike).
      3. Keep your counter-speech concise (150-200 words), highly intellectual, structured, and finish with a direct debate challenge.
      
      Respond strictly in JSON:
      {
        "lastSpeechFeedback": {
          "score": number, // out of 10
          "strength": string, // brief sentence
          "fallaciesIdentified": string[], // fallacies you spotted e.g. "Strawman", "Slippery Slope", or "None"
          "forensicSuggestion": string // rhetorical tip
        } | null, // null only if first turn
        "aiCounterArgument": string, // your professional collegiate counter speech
        "debateRound": number // incremented round
      }
    `;

    const formattedHistory = messages.map(msg => {
      const actor = msg.role === "user" ? "User Debater" : "AI Debater";
      return `${actor}: ${msg.text}`;
    }).join("\n");

    const promptText = `
      Here is the debate record:
      ${formattedHistory || "[START OF DEBATE. Issue your opening assertion countering the user's stance on this topic and establish the rules. Keep it competitive!]"}

      Respond with formal debate rebuttal and feedback.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: promptText,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            lastSpeechFeedback: {
              type: Type.OBJECT,
              nullable: true,
              properties: {
                score: { type: Type.NUMBER },
                strength: { type: Type.STRING },
                fallaciesIdentified: { type: Type.ARRAY, items: { type: Type.STRING } },
                forensicSuggestion: { type: Type.STRING }
              },
              required: ["score", "strength", "fallaciesIdentified", "forensicSuggestion"]
            },
            aiCounterArgument: { type: Type.STRING },
            debateRound: { type: Type.NUMBER }
          },
          required: ["lastSpeechFeedback", "aiCounterArgument", "debateRound"]
        }
      }
    });

    const resultText = response.text || "{}";
    res.json(JSON.parse(resultText));
  } catch (error: any) {
    console.error("Debate server error:", error);
    res.status(500).json({ error: error.message || "Debate server error" });
  }
});

// Serve frontend assets
async function startServer() {
  // Vite Integration in Development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // Serve Static files in Production
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`SmartSpeak AI backend listening on port ${PORT}`);
  });
}

startServer();
