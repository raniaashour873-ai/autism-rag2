import {
  Search,
  FileText,
  ShieldCheck,
} from "lucide-react";

function EmptyChat({ onSuggestion }) {
  const suggestions = [
    "What are the diagnostic considerations for autism in adults?",
    "What does NICE recommend for autism assessment?",
    "What support strategies are recommended for autistic adults?",
  ];

  return (
    <div className="empty-chat">

      <div className="empty-thread-mark">

        <div className="empty-thread-symbol">
          ∞
        </div>

        <span />
        <span />
        <span />

      </div>

      <span className="empty-kicker">
        EVIDENCE-GROUNDED CLINICAL AI
      </span>

      <h2>
        Start with a
        <br />
        clinical question.
      </h2>

      <p>
        Threadline retrieves relevant clinical evidence
        and connects the answer directly to its sources.
      </p>

      <div className="suggestion-grid">

        {suggestions.map((suggestion, index) => (

          <button
            key={suggestion}
            className="suggestion-card"
            onClick={() => onSuggestion?.(suggestion)}
          >

            <span className="suggestion-icon">

              {index === 0 && (
                <Search size={16} />
              )}

              {index === 1 && (
                <FileText size={16} />
              )}

              {index === 2 && (
                <ShieldCheck size={16} />
              )}

            </span>

            <span>
              {suggestion}
            </span>

          </button>

        ))}

      </div>

    </div>
  );
}

export default EmptyChat;