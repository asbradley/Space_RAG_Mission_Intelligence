import { useState } from "react";

const API_BASE = "http://127.0.0.1:8001";

type Source = {
  title: string;
  source_url: string | null;
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
              <strong>Sources:</strong>
              <ul>
                {result.sources.map((s, i) => (
                  <li key={i}>
                    <a href={s.source_url ?? "#"} target="_blank" rel="noreferrer">
                      {s.title}
                    </a>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </section>
  );
}
