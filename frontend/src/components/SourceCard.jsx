import { useState } from "react";

function SourceItem({ source }) {
  const [open, setOpen] = useState(false);

  return (
    <article className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
      {/* Badge + toggle */}
      <div className="flex items-center justify-between">
        <span className="inline-flex items-center rounded-full bg-slate-800 px-2.5 py-0.5 text-xs font-medium text-white">
          {source.source_id} | page {source.page_number}
        </span>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="text-xs text-slate-400 hover:text-slate-600"
        >
          {open ? "Hide excerpt ▲" : "Show excerpt ▼"}
        </button>
      </div>

      {/* Collapsible chunk text */}
      {open && (
        <p className="mt-2 text-xs leading-relaxed text-slate-600 line-clamp-none">
          {source.chunk_text}
        </p>
      )}
    </article>
  );
}

export default function SourceCard({ sources }) {
  if (!sources || sources.length === 0) {
    return (
      <p className="text-sm text-slate-400">
        Sources will appear here after you ask a question.
      </p>
    );
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {sources.map((src) => (
        <SourceItem key={src.source_id} source={src} />
      ))}
    </div>
  );
}


