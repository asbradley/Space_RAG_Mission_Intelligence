import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

type Document = {
  id: number;
  title: string;
  source_id: string;
  source_url: string | null;
  ingested_at: string;
};

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/documents`)
      .then((res) => {
        if (!res.ok) throw new Error(`API returned ${res.status}`);
        return res.json();
      })
      .then(setDocuments)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <main style={{ fontFamily: "sans-serif", padding: "2rem" }}>
      <h1>Space RAG</h1>
      <p>Documents ingested from NASA's Technical Reports Server (NTRS).</p>

      {error && <p style={{ color: "crimson" }}>Failed to load documents: {error}</p>}

      {!error && documents.length === 0 && (
        <p>No documents ingested yet — run the Phase 1 ingestion script.</p>
      )}

      <ul>
        {documents.map((doc) => (
          <li key={doc.id}>
            <a href={doc.source_url ?? "#"} target="_blank" rel="noreferrer">
              {doc.title}
            </a>{" "}
            <small>({new Date(doc.ingested_at).toLocaleString()})</small>
          </li>
        ))}
      </ul>
    </main>
  );
}
