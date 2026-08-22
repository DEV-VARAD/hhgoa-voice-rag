import { useEffect, useState } from "react";
import { Link, Mic, Moon, RotateCcw, Sun, X } from "lucide-react";

import VoiceRecorder from "./components/VoiceRecorder";
import TranscriptCard from "./components/TranscriptCard";
import AnswerCard from "./components/AnswerCard";
import PipelineStatus from "./components/PipelineStatus";
import LatencyMetrics from "./components/LatencyMetrics";
import { askRAG } from "./services/api";
import meImage from "./assets/me.jpg";
import sarveshImage from "./assets/sarvesh.jpg";

function App() {
  const [showIntro, setShowIntro] = useState(true);
  const [openProfile, setOpenProfile] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState("hi-IN");
  const [transcript, setTranscript] = useState("");
  const [result, setResult] = useState(null);
  const [stage, setStage] = useState("idle");
  const [error, setError] = useState("");
  const [theme, setTheme] = useState("dark");

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  useEffect(() => {
    const timer = window.setTimeout(() => setShowIntro(false), 1300);
    return () => window.clearTimeout(timer);
  }, []);

  const handleTranscript = async (text) => {
    setError("");
    setResult(null);

    if (!text?.trim()) return;

    try {
      setTranscript(text);
      setStage("retrieving");

      const data = await askRAG(text, (newStage) => setStage(newStage));

      setResult(data);
      setStage("complete");
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong.");
      setStage("error");
    }
  };

  const handleLanguageChange = (language) => {
    if (language === targetLanguage) return;

    setTargetLanguage(language);
    setError("");
  };

  const reset = () => {
    setTranscript("");
    setResult(null);
    setStage("idle");
    setError("");
  };

  return (
    <main className={`app ${theme === "light" ? "light-theme" : ""}`}>
      <section
        className={`intro-curtain ${
          showIntro ? "" : "intro-curtain--open"
        }`}
        aria-hidden={!showIntro}
      >
        <div className="curtain-panel curtain-panel--left">
          <div className="palm palm--top" />
          <div className="palm palm--bottom" />
        </div>

        <div className="curtain-panel curtain-panel--right">
          <div className="palm palm--top" />
          <div className="palm palm--bottom" />
        </div>

        <div className="intro-content">
          <p className="intro-kicker">HH GOA · 2026</p>
          <p className="intro-title">HACKER</p>
          <p className="intro-goa">गोवा</p>
          <p className="intro-title intro-title--bottom">HOUSE</p>
        </div>

        <p className="intro-credit">VOICE RAG · GOA, INDIA</p>
      </section>

      <div className="leaf-shadow leaf-shadow--left" />
      <div className="leaf-shadow leaf-shadow--right" />

      <nav className="navbar">
        <div className="brand">
          <span className="nav-location">GOA · INDIA</span>

          {(transcript || result) && (
            <button className="reset-btn" onClick={reset}>
              <RotateCcw size={16} />
              New question
            </button>
          )}

          <button
            className="theme-toggle-btn"
            onClick={toggleTheme}
            aria-label={`Switch to ${
              theme === "dark" ? "light" : "dark"
            } theme`}
            type="button"
          >
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          </button>

          <div
            className="profile-menu-group"
            onMouseLeave={() => setOpenProfile(null)}
          >
            <div
              className="profile-wrapper"
              onMouseEnter={() => setOpenProfile("you")}
            >
              <button
                className={`profile-trigger ${
                  openProfile === "you" ? "active" : ""
                }`}
                onClick={() =>
                  setOpenProfile(
                    openProfile === "you" ? null : "you"
                  )
                }
                aria-label="Open Varad's social links"
                aria-expanded={openProfile === "you"}
                type="button"
              >
                <img src={meImage} alt="Me" className="profile-img" />
                <span>Varad</span>
              </button>

              {openProfile === "you" && (
                <div className="profile-dropdown">
                  <strong>Varad's links</strong>

                  <a
                    href="https://www.linkedin.com/in/varad-yadav/"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <Link size={15} />
                    LinkedIn
                  </a>

                  <a
                    href="https://github.com/DEV-VARAD"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <Link size={15} />
                    GitHub
                  </a>

                  <a
                    href="https://x.com/varadkf"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <X size={15} />
                    X
                  </a>
                </div>
              )}
            </div>

            <div
              className="profile-wrapper"
              onMouseEnter={() => setOpenProfile("friend")}
            >
              <button
                className={`profile-trigger profile-trigger--friend ${
                  openProfile === "friend" ? "active" : ""
                }`}
                onClick={() =>
                  setOpenProfile(
                    openProfile === "friend" ? null : "friend"
                  )
                }
                aria-label="Open Sarvesh's social links"
                aria-expanded={openProfile === "friend"}
                type="button"
              >
                <img
                  src={sarveshImage}
                  alt="Sarvesh"
                  className="profile-img"
                />
                <span>Sarvesh</span>
              </button>

              {openProfile === "friend" && (
                <div className="profile-dropdown">
                  <strong>Sarvesh's links</strong>

                  <a
                    href="https://www.linkedin.com/in/sarvesh-patil-75060721a"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <Link size={15} />
                    LinkedIn
                  </a>

                  <a
                    href="https://github.com/Sarvesh-107"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <Link size={15} />
                    GitHub
                  </a>

                  <a
                    href="https://x.com/XENOGAM59155308"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <X size={15} />
                    X
                  </a>
                </div>
              )}
            </div>
          </div>
        </div>
      </nav>

      <section className="hero">
        <div className="badge">
          <span className="live-dot" />
          HH GOA 2026 · VOICE ENABLED RAG
        </div>

        <h1 className="hero-brand">GOTCHU</h1>

        <p>Your personal assistant, ready to help.</p>

        {!transcript && !result && (
          <div className="suggestions-container">
            <p className="suggestions-title">Try asking:</p>

            <div className="suggestions-grid">
              <button
                className="suggestion-pill"
                onClick={() =>
                  handleTranscript("What is Werner syndrome")
                }
              >
                <span className="lang-tag lang-tag--en">
                  English
                </span>
                <span className="suggestion-text">
                  What is Werner syndrome (English)
                </span>
              </button>

              <button
                className="suggestion-pill"
                onClick={() =>
                  handleTranscript("मैनहट्टन परियोजना क्या है")
                }
              >
                <span className="lang-tag lang-tag--hi">
                  हिंदी
                </span>
                <span className="suggestion-text">
                  मैनहट्टन परियोजना क्या है
                </span>
              </button>

              <button
                className="suggestion-pill"
                onClick={() =>
                  handleTranscript("How do I reset my password")
                }
              >
                <span className="lang-tag lang-tag--en">
                  English
                </span>
                <span className="suggestion-text">
                  How do I reset my password
                </span>
              </button>

              <button
                className="suggestion-pill"
                onClick={() =>
                  handleTranscript("फ्लूम किस दिशा में बहता है")
                }
              >
                <span className="lang-tag lang-tag--hi">
                  हिंदी
                </span>
                <span className="suggestion-text">
                  फ्लूम किस दिशा में बहता है
                </span>
              </button>

              <button
                className="suggestion-pill"
                onClick={() =>
                  handleTranscript("what is the capital of france")
                }
              >
                <span className="lang-tag lang-tag--en">
                  English
                </span>
                <span className="suggestion-text">
                  what is the capital of france
                </span>
              </button>

              <button
                className="suggestion-pill"
                onClick={() =>
                  handleTranscript("tell me a joke")
                }
              >
                <span className="lang-tag lang-tag--en">
                  English
                </span>
                <span className="suggestion-text">
                  tell me a joke
                </span>
              </button>

              <button
                className="suggestion-pill"
                onClick={() =>
                  handleTranscript(
                    "गोल्ड पैमाने पर सोने की कठोरता कितनी होती है"
                  )
                }
              >
                <span className="lang-tag lang-tag--hi">
                  हिंदी
                </span>
                <span className="suggestion-text">
                  गोल्ड पैमाने पर सोने की कठोरता कितनी होती है
                </span>
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="workspace">
        <div className="voice-section">
          <VoiceRecorder
            onTranscript={handleTranscript}
            disabled={
              stage !== "idle" &&
              stage !== "complete" &&
              stage !== "error"
            }
            language={targetLanguage}
            onLanguageChange={handleLanguageChange}
          />

          {error && <div className="error-card">{error}</div>}
        </div>

        <PipelineStatus stage={stage} />

        {transcript && (
          <TranscriptCard transcript={transcript} />
        )}

        {result && (
          <>
            <AnswerCard result={result} />
            <LatencyMetrics latency={result.latency_ms} />
          </>
        )}
      </section>

      <footer>
        <Mic size={14} />
        Voice → STT → Retrieval → Grounded Generation
      </footer>
    </main>
  );
}

export default App;
