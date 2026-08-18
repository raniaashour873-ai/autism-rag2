import {
  X,
  BookOpen,
  ShieldCheck,
  Clock3,
} from "lucide-react";


function UtilityPanel({
  type,
  messages = [],
  onClose,
}) {

  if (!type) {
    return null;
  }


  const config = {

    guidelines: {
      icon: BookOpen,
      title: "Clinical Guidelines",
      kicker: "KNOWLEDGE BASE",
    },

    history: {
      icon: Clock3,
      title: "Question History",
      kicker: "WORKSPACE",
    },

    safety: {
      icon: ShieldCheck,
      title: "Safety & Scope",
      kicker: "CLINICAL SAFETY",
    },

  };


  const current =
    config[type];

  if (!current) {
    return null;
  }


  const Icon =
    current.icon;


  return (

    <div
      className="utility-overlay"
      onMouseDown={(event) => {

        if (
          event.target === event.currentTarget
        ) {
          onClose();
        }

      }}
    >

      <div className="utility-panel">


        {/* HEADER */}

        <header className="utility-header">

          <div>

            <span className="evidence-kicker">
              {current.kicker}
            </span>

            <h2>
              {current.title}
            </h2>

          </div>


          <button
            type="button"
            className="utility-close"
            onClick={onClose}
            aria-label="Close"
          >
            <X size={16} />
          </button>

        </header>


        {/* GUIDELINES */}

        {type === "guidelines" && (

          <div className="utility-content">

            <div className="utility-icon">
              <Icon size={18} />
            </div>

            <h3>
              Evidence-based autism guidance
            </h3>

            <p>
              Threadline retrieves relevant
              clinical evidence from the indexed
              autism sources and uses those
              sources to ground its responses.
            </p>

            <div className="utility-note">

              <strong>
                Traceability
              </strong>

              <span>
                Retrieved evidence is displayed
                with its section, page and
                retrieval distance when available.
              </span>

            </div>

            <div className="utility-note">

              <strong>
                RAG workflow
              </strong>

              <span>
                Question → Retrieval →
                Evidence → Clinical response.
              </span>

            </div>

          </div>

        )}


        {/* HISTORY */}

        {type === "history" && (

          <div className="utility-content">

            {messages.length === 0 ? (

              <div className="utility-empty">

                <Clock3 size={20} />

                <p>
                  No questions have been asked
                  in this session yet.
                </p>

              </div>

            ) : (

              <div className="utility-history">

                {messages
                  .slice()
                  .reverse()
                  .map(
                    (message, index) => (

                      <button
                        type="button"
                        className="utility-history-item"
                        key={message.id}
                        onClick={() => {

                          const element =
                            document.getElementById(
                              `message-${message.id}`
                            );

                          onClose();

                          setTimeout(() => {

                            element?.scrollIntoView({
                              behavior: "smooth",
                              block: "center",
                            });

                          }, 100);

                        }}
                      >

                        <span>
                          {String(
                            messages.length -
                            index
                          ).padStart(2, "0")}
                        </span>

                        <p>
                          {message.question}
                        </p>

                      </button>

                    )
                  )}

              </div>

            )}

          </div>

        )}


        {/* SAFETY */}

        {type === "safety" && (

          <div className="utility-content">

            <div className="utility-icon">
              <Icon size={18} />
            </div>

            <h3>
              Clinical decision support only
            </h3>

            <p>
              Threadline retrieves and synthesizes
              clinical evidence. It does not replace
              professional clinical assessment or
              clinical judgment.
            </p>


            <div className="utility-safety-list">

              <div>

                <strong>
                  ALLOWED
                </strong>

                <span>
                  Evidence supports a grounded response.
                </span>

              </div>


              <div>

                <strong>
                  NEEDS_CAUTION
                </strong>

                <span>
                  Evidence exists, but clinical
                  judgment is required.
                </span>

              </div>


              <div>

                <strong>
                  REFUSE
                </strong>

                <span>
                  The system cannot safely provide
                  a grounded answer.
                </span>

              </div>

            </div>

          </div>

        )}

      </div>

    </div>
  );
}


export default UtilityPanel;