export async function askRAG(question, onStage) {
  onStage("retrieving");

  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: question }),
  });

  onStage("generating");

  if (!response.ok) {
    throw new Error(`RAG backend error: ${response.status}`);
  }

  const data = await response.json();

  return {
    answer: data.answer,
    grounded: data.grounded,
    confidence: null,
    sources: (data.sources || []).map((s) => ({
      text: s.text,
      language: s.language,
      score: s.score,
    })),
    latency_ms: {
      stt: null,
      retrieval: data.latency_ms?.retrieval ?? null,
      generation: data.latency_ms?.generation ?? null,
      total: data.latency_ms?.total ?? null,
    },
  };
}