import { useRef, useState } from "react";

import { uploadPDF } from "../api";

const STATUS = { IDLE: "idle", UPLOADING: "uploading", DONE: "done", ERROR: "error" };

export default function UploadPanel({ onUploadSuccess }) {
  const [status, setStatus] = useState(STATUS.IDLE);
  const [filename, setFilename] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  async function handleFile(file) {
    if (!file || file.type !== "application/pdf") {
      setErrorMsg("Please select a valid PDF file.");
      setStatus(STATUS.ERROR);
      return;
    }
    setFilename(file.name);
    setStatus(STATUS.UPLOADING);
    setErrorMsg(null);
    try {
      const { document_id } = await uploadPDF(file);
      setStatus(STATUS.DONE);
      onUploadSuccess(document_id);
    } catch (err) {
      setStatus(STATUS.ERROR);
      setErrorMsg(err.message);
    }
  }

  function onInputChange(e) {
    handleFile(e.target.files[0]);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }

  return (
    <section className="rounded-lg bg-white p-5 shadow">
      <h1 className="text-xl font-semibold text-slate-800">Upload PDF</h1>
      <p className="mt-1 text-sm text-slate-500">
        Upload a document to start asking questions.
      </p>

      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={[
          "mt-4 flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-10 transition-colors",
          dragging
            ? "border-slate-500 bg-slate-100"
            : "border-slate-300 hover:border-slate-400 hover:bg-slate-50",
        ].join(" ")}
      >
        <svg className="mb-2 h-8 w-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 12V4m0 0L8 8m4-4 4 4" />
        </svg>
        <p className="text-sm text-slate-500">
          <span className="font-medium text-slate-700">Click to upload</span> or drag and drop
        </p>
        <p className="mt-1 text-xs text-slate-400">PDF only</p>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={onInputChange}
        />
      </div>

      {/* Status feedback */}
      {status === STATUS.UPLOADING && (
        <p className="mt-3 flex items-center gap-2 text-sm text-slate-600">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-400 border-t-transparent" />
          Uploading <span className="font-medium">{filename}</span>…
        </p>
      )}
      {status === STATUS.DONE && (
        <p className="mt-3 text-sm text-emerald-600">
          ✓ <span className="font-medium">{filename}</span> indexed and ready.
        </p>
      )}
      {status === STATUS.ERROR && (
        <p className="mt-3 text-sm text-red-600">✗ {errorMsg}</p>
      )}
    </section>
  );
}


