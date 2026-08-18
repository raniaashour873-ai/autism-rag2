import {
  Plus,
  MessageSquare,
  BookOpen,
  ShieldCheck,
  Clock3,
} from "lucide-react";


function ChatSidebar({
  onNewQuestion,
  onEvidence,
  onGuidelines,
  onHistory,
  onSafety,
  messages = [],
}) {

  return (
    <aside className="chat-sidebar">

      {/* NEW QUESTION */}

      <button
        type="button"
        className="new-question-button"
        onClick={onNewQuestion}
      >
        <Plus size={16} />

        <span>
          New question
        </span>
      </button>


      {/* RECENT */}

      <div className="sidebar-section">

        <div className="sidebar-section-label">
          RECENT
        </div>

        {messages.length === 0 ? (

          <div className="sidebar-empty">

            <MessageSquare size={14} />

            <span>
              Your questions will
              appear here.
            </span>

          </div>

        ) : (

          <div className="sidebar-history-list">

            {messages
              .slice(-5)
              .reverse()
              .map((message) => (

                <button
                  type="button"
                  className="sidebar-history-item"
                  key={message.id}
                  title={message.question}
                  onClick={() => {

                    const element =
                      document.getElementById(
                        `message-${message.id}`
                      );

                    if (element) {

                      element.scrollIntoView({
                        behavior: "smooth",
                        block: "center",
                      });

                    }

                  }}
                >

                  <MessageSquare size={13} />

                  <span>
                    {message.question}
                  </span>

                </button>

              ))}

          </div>

        )}

      </div>


      {/* KNOWLEDGE */}

      <div className="sidebar-section">

        <div className="sidebar-section-label">
          KNOWLEDGE
        </div>


        <button
          type="button"
          className="sidebar-item"
          onClick={onEvidence}
        >
          <BookOpen size={14} />

          <span>
            Evidence
          </span>
        </button>


        <button
          type="button"
          className="sidebar-item"
          onClick={onGuidelines}
        >
          <BookOpen size={14} />

          <span>
            Guidelines
          </span>
        </button>

      </div>


      {/* WORKSPACE */}

      <div className="sidebar-section">

        <div className="sidebar-section-label">
          WORKSPACE
        </div>


        <button
          type="button"
          className="sidebar-item"
          onClick={onHistory}
        >
          <Clock3 size={14} />

          <span>
            History
          </span>
        </button>


        <button
          type="button"
          className="sidebar-item"
          onClick={onSafety}
        >
          <ShieldCheck size={14} />

          <span>
            Safety
          </span>
        </button>

      </div>


      {/* FOOTER */}

      <div className="sidebar-footer">

        <div className="sidebar-footer-line">
          RAG SYSTEM
        </div>

        <div className="sidebar-footer-status">

          <span className="workspace-status-dot" />

          Operational

        </div>

      </div>

    </aside>
  );
}


export default ChatSidebar;