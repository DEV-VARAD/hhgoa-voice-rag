/* global process */

import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import { SarvamAIClient } from "sarvamai";

dotenv.config();

const app = express();

// Keep the API usable both through Vite's dev proxy and from a separately
// hosted frontend. The proxy is preferred in development, but these headers
// prevent the browser from hiding a useful server response as "Failed to fetch".
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", process.env.CLIENT_ORIGIN || "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.sendStatus(204);
  }

  next();
});

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024,
  },
});

if (!process.env.SARVAM_API_KEY) {
  throw new Error("SARVAM_API_KEY is missing from .env");
}

const client = new SarvamAIClient({
  apiSubscriptionKey: process.env.SARVAM_API_KEY,
});

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    service: "VoiceRAG STT",
  });
});

app.post("/api/transcribe", upload.single("audio"), async (req, res) => {
  const startedAt = performance.now();

  try {
    if (!req.file) {
      return res.status(400).json({
        error: "No audio file received.",
      });
    }

    const languageCode = ["en-IN", "hi-IN"].includes(req.body.targetLanguage)
      ? req.body.targetLanguage
      : "hi-IN";

    console.log(`Transcribing audio in ${languageCode} with same-language mode.`);

    const audioFile = new File(
      [req.file.buffer],
      req.file.originalname || "recording.webm",
      {
        type: req.file.mimetype || "audio/webm",
      }
    );

    const response = await client.speechToText.transcribe({
      file: audioFile,
      model: "saaras:v3",
      // Verbatim keeps the spoken language and script; it never requests
      // Sarvam's speech-to-English translation mode.
      mode: "verbatim",
      language_code: languageCode,
    });

    if (languageCode === "hi-IN" && !/[\u0900-\u097F]/.test(response.transcript || "")) {
      return res.status(422).json({
        error: "Hindi was selected, but Hindi script was not returned. Please record again.",
      });
    }

    const latencyMs = Math.round(performance.now() - startedAt);

    return res.json({
      transcript: response.transcript,
      language_code: languageCode,
      stt_latency_ms: latencyMs,
    });
  } catch (error) {
    console.error("Sarvam STT error:", error);

    return res.status(500).json({
      error: "Transcription failed.",
      details: error?.message || "Unknown Sarvam API error",
    });
  }
});

app.use((error, _req, res, _next) => {
  void _next;

  if (error instanceof multer.MulterError) {
    return res.status(400).json({ error: error.message });
  }

  console.error("Unexpected server error:", error);
  return res.status(500).json({ error: "Unexpected server error." });
});

app.listen(process.env.PORT || 3001, () => {
  console.log(
    `VoiceRAG server running on http://localhost:${process.env.PORT || 3001}`
  );
});
