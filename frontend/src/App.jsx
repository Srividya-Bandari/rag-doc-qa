import { useState } from "react";

import ChatPanel from "./components/ChatPanel";
import SourceCard from "./components/SourceCard";
import UploadPanel from "./components/UploadPanel";

export default function App() {
  const [documentId, setDocumentId] = useState(null);
  const [sources, setSources] = useState([]);

  function handleUploadSuccess(nextDocumentId) {
    setDocumentId(nextDocumentId);
    setSources([]);
  }

  return (
    <div className="min-h-screen bg-slate-100">
      {/* Header */}
      <header className="bg-slate-800 px-6 py-4 shadow">
        <h1 className="text-lg font-semibold text-white tracking-tight">
          RAG Document Q&amp;A
        </h1>
        <p className="text-xs text-slate-400 mt-0.5">
          Upload a PDF — ask questions — get cited answers
        </p>
      </header>

      <main className="mx-auto max-w-6xl p-6 space-y-6">
        {/* Upload + Chat row */}
        <div className="grid gap-5 md:grid-cols-2">
          <UploadPanel onUploadSuccess={handleUploadSuccess} />
          <ChatPanel
            key={documentId ?? "no-document"}
            documentId={documentId}
            onNewSources={setSources}
          />
        </div>

        {/* Sources row */}
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
            Sources
          </h2>
          <SourceCard sources={sources} />
        </section>
      </main>
    </div>
  );
}

