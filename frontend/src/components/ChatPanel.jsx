import { useEffect, useRef, useState } from "react";

import { askQuestionStream } from "../api";

function createMessage(role, text = "", extra = {}) {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    text,
    ...extra,
  };
}

export default function ChatPanel({ documentId, onNewSources }) {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  // Scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    setMessages([]);
    setQuestion("");
    onNewSources([]);
  }, [documentId, onNewSources]);

  async function handleAsk() {
    const q = question.trim();
    if (!q || !documentId || loading) return;

    const userMessage = createMessage("user", q);
    const assistantMessage = createMessage("assistant", "", { streaming: true });
    const chatHistory = messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .map((message) => ({
        role: message.role,
        content: message.text,
      }));

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setQuestion("");
    setLoading(true);

    try {
      await askQuestionStream(documentId, q, chatHistory, {
        onChunk(delta) {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantMessage.id
                ? {
                    ...message,
                    text: message.text + delta,
                  }
                : message,
            ),
          );
        },
        onDone(payload) {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantMessage.id
                ? {
                    ...message,
                    text: payload.answer,
                    streaming: false,
                  }
                : message,
            ),
          );
          onNewSources(payload.sources ?? []);
        },
      });
    } catch (err) {
      setMessages((prev) => [
        ...prev.filter((message) => message.id !== assistantMessage.id),
        createMessage("error", err.message),
      ]);
      onNewSources([]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  }

  return (
    <section className="flex flex-col rounded-lg bg-white p-5 shadow" style={{ minHeight: "420px" }}>
      <h2 className="text-xl font-semibold text-slate-800">Ask Questions</h2>

      {/* Message list */}
      <div className="mt-3 flex-1 overflow-y-auto rounded border border-slate-100 bg-slate-50 p-3">
        {messages.length === 0 ? (
          <p className="text-sm text-slate-400">
            {documentId
              ? "Ask a question about your uploaded PDF…"
              : "Upload a PDF first to start asking questions."}
          </p>
        ) : (
          messages.map((msg, i) => (
            <div
              key={msg.id}
              className={[
                "mb-3 max-w-[90%] rounded-lg px-3 py-2 text-sm",
                msg.role === "user"
                  ? "ml-auto bg-slate-800 text-white"
                  : msg.role === "error"
                  ? "bg-red-50 text-red-600"
                  : "bg-white text-slate-700 shadow-sm",
              ].join(" ")}
            >
              {msg.text || (msg.streaming ? "Thinking…" : "")}
            </div>
          ))
        )}
        {loading && (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-600" />
            Thinking…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input row */}
      <div className="mt-3 flex gap-2">
        <textarea
          rows={2}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={documentId ? "Ask a question… (Enter to send)" : "Upload a PDF first"}
          disabled={!documentId || loading}
          className="flex-1 resize-none rounded border border-slate-300 p-2 text-sm disabled:cursor-not-allowed disabled:bg-slate-100 focus:outline-none focus:ring-1 focus:ring-slate-400"
        />
        <button
          type="button"
          onClick={handleAsk}
          disabled={!documentId || loading || !question.trim()}
          className="self-end rounded bg-slate-800 px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40 hover:bg-slate-700"
        >
          Ask
        </button>
      </div>
    </section>
  );
}

