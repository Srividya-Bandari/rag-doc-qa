const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/**
 * Upload a PDF file.
 * Returns { document_id, status }.
 */
export async function uploadPDF(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE_URL}/api/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Upload failed (${res.status})`);
  }
  return res.json();
}

/**
 * Ask a question about an uploaded document.
 * Returns { answer, sources }.
 */
export async function askQuestion(documentId, question, chatHistory = []) {
  const res = await fetch(`${BASE_URL}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_id: documentId,
      question,
      chat_history: chatHistory,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

function parseEventBlock(block) {
  const lines = block.split("\n");
  let event = "message";
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }

  const rawData = dataLines.join("\n");
  return {
    event,
    payload: rawData ? JSON.parse(rawData) : {},
  };
}

export async function askQuestionStream(
  documentId,
  question,
  chatHistory = [],
  { onChunk, onDone } = {},
) {
  const res = await fetch(`${BASE_URL}/api/ask/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      document_id: documentId,
      question,
      chat_history: chatHistory,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Request failed (${res.status})`);
  }
  if (!res.body) {
    throw new Error("Streaming is not available in this browser.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      if (!block.trim()) {
        continue;
      }
      const { event, payload } = parseEventBlock(block);
      if (event === "chunk") {
        onChunk?.(payload.delta ?? "");
      } else if (event === "done") {
        finalPayload = payload;
        onDone?.(payload);
      } else if (event === "error") {
        throw new Error(payload.detail ?? "Streaming request failed.");
      }
    }

    if (done) {
      break;
    }
  }

  return finalPayload;
}
