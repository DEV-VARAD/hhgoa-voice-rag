import { useEffect, useRef, useState } from "react";
import { Mic, Square, Loader2 } from "lucide-react";

function VoiceRecorder({
  onTranscript,
  disabled,
  language,
  onLanguageChange,
}) {
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      setError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      streamRef.current = stream;
      chunksRef.current = [];

      const mimeType = MediaRecorder.isTypeSupported(
        "audio/webm;codecs=opus"
      )
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const mediaRecorder = new MediaRecorder(stream, { mimeType });

      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, {
          type: mimeType,
        });

        stream.getTracks().forEach((track) => track.stop());

        setRecording(false);
        setProcessing(true);

        try {
          const formData = new FormData();

          formData.append(
            "audio",
            audioBlob,
            "recording.webm"
          );
          formData.append("targetLanguage", language);

          // In development Vite proxies /api to the Express server. This avoids
          // a cross-origin browser request and also works when both are deployed
          // behind the same domain.
          const response = await fetch("/api/transcribe", {
            method: "POST",
            body: formData,
          });

          const data = await response.json().catch(() => ({}));

          console.log("Sarvam STT response:", data);

          if (!response.ok) {
            throw new Error(
              data.details ||
              data.error ||
              "Transcription failed."
            );
          }

          if (!data.transcript?.trim()) {
            throw new Error(
              "No speech was detected. Please try again."
            );
          }

          // Send REAL Sarvam transcript to App.jsx
          onTranscript(data.transcript, data.language_code || language);
        } catch (err) {
          console.error("Transcription error:", err);
          setError(err.message || "Could not transcribe your audio.");
        } finally {
          setProcessing(false);
        }
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      console.error("Microphone error:", err);
      setError(
        "Microphone access is required to ask a question."
      );
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  };

  return (
    <div className="recorder-card">
      <div className={`mic-orbit ${recording ? "recording" : ""}`}>
        <div className="mic-circle">
          {processing ? (
            <Loader2 size={34} className="spin" />
          ) : (
            <Mic size={34} />
          )}
        </div>
      </div>

      <h2>
        {recording
          ? "Listening..."
          : processing
          ? "Transcribing..."
          : "Tap to ask anything"}
      </h2>

      <p>
        {recording
          ? "Speak your question, then stop recording."
          : processing
          ? "Sarvam is converting your voice into text."
          : "Ask your question using your voice."}
      </p>

      <div className="language-switch" aria-label="Transcript language">
        <button
          className={language === "en-IN" ? "active" : ""}
          disabled={processing}
          onClick={() => onLanguageChange("en-IN")}
          type="button"
        >
          English
        </button>
        <button
          className={language === "hi-IN" ? "active" : ""}
          disabled={processing}
          onClick={() => onLanguageChange("hi-IN")}
          type="button"
        >
          हिन्दी
        </button>
      </div>

      {error && <div className="voice-error">{error}</div>}

      {!recording ? (
        <button
          className="record-btn"
          onClick={startRecording}
          disabled={disabled || processing}
        >
          <Mic size={19} />
          {processing ? "Transcribing..." : "Start speaking"}
        </button>
      ) : (
        <button className="stop-btn" onClick={stopRecording}>
          <Square size={17} fill="currentColor" />
          Stop recording
        </button>
      )}
    </div>
  );
}

export default VoiceRecorder;
