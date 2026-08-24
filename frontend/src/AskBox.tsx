import { useState } from "react";

const API_BASE = "http://127.0.0.1:8001";

type Source = {
  n: number;
  chunk_id: number;
  title: string;
  source_url: string | null;
  excerpt: string;
  cited: boolean;
};

type AskResponse = {
  answer: string;
  sources: Source[];
};

export default function AskBox() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section style={{ marginTop: "2rem" }}>
      <h2>Ask a question</h2>
      <form onSubmit={handleAsk} style={{ display: "flex", gap: "0.5rem" }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. What were the main objectives of the Apollo 11 mission?"
          style={{ flex: 1, padding: "0.5rem" }}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Asking…" : "Ask"}
        </button>
      </form>

      {error && <p style={{ color: "crimson" }}>Failed to get an answer: {error}</p>}

      {result && (
        <div style={{ marginTop: "1rem" }}>
          <p>{result.answer}</p>
          {result.sources.length > 0 && (
            <>
              <strong>Evidence:</strong>
              <ol style={{ listStyle: "none", padding: 0 }}>
                {result.sources.map((s) => (
                  <li
                    key={s.chunk_id}
                    style={{
                      margin: "0.75rem 0",
                      padding: "0.5rem 0.75rem",
                      // Cited passages are the ones the answer actually
                      // referenced; the rest were retrieved but unused.
                      borderLeft: `4px solid ${s.cited ? "#2563eb" : "#d1d5db"}`,
                      background: s.cited ? "#f0f5ff" : "transparent",
                    }}
                  >
                    <div>
                      <strong>[{s.n}]</strong>{" "}
                      <a href={s.source_url ?? "#"} target="_blank" rel="noreferrer">
                        {s.title}
                      </a>{" "}
                      <small style={{ color: "#6b7280" }}>
                        {s.cited ? "· cited in answer" : "· retrieved, not cited"}
                      </small>
                    </div>
                    <pre
                      style={{
                        maxHeight: "8rem",
                        overflowY: "auto",
                        whiteSpace: "pre-wrap",
                        margin: "0.5rem 0 0",
                        fontSize: "0.85rem",
                        color: "#374151",
                      }}
                    >
                      {s.excerpt}
                    </pre>
                  </li>
                ))}
              </ol>
            </>
          )}
        </div>
      )}
    </section>
  );
}
