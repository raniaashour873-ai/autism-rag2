import {
  Check,
  Copy,
  AlertTriangle,
  ShieldX,
} from "lucide-react";

import { useState } from "react";

import SafetyStatus from "./SafetyStatus";
import Citation from "../evidence/Citation";


function AssistantAnswer({
  answer,
  safetyLabel = "REFUSE",
  sources = [],
  activeSourceId,
  onCitationClick,
  citationRefs,
}) {

  const [copied, setCopied] =
    useState(false);


  const normalizedLabel =
    String(safetyLabel).toUpperCase();


  const isRefused =
    normalizedLabel === "REFUSE";


  const isCaution =
    normalizedLabel === "NEEDS_CAUTION";


  async function handleCopyAnswer() {

    if (!answer) {
      return;
    }


    try {

      await navigator.clipboard.writeText(
        answer
      );

      setCopied(true);


      setTimeout(() => {
        setCopied(false);
      }, 1800);

    } catch (error) {

      console.error(
        "Copy failed:",
        error
      );

    }

  }


  return (

    <article
      className={[
        "message",
        "assistant-message",
        isRefused
          ? "assistant-refused"
          : "",
        isCaution
          ? "assistant-caution"
          : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >


      {/* HEADER */}

      <div className="assistant-header">

        <div className="assistant-identity">

          <div className="assistant-mark">
            ∞
          </div>

          <div>

            <strong>
              Threadline AI
            </strong>

            <span>
              Clinical evidence synthesis
            </span>

          </div>

        </div>


        <SafetyStatus
          label={normalizedLabel}
        />

      </div>


      {/* ANSWER */}

      <div className="assistant-answer">

        <div className="answer-heading">

          <span className="answer-label">
            CLINICAL RESPONSE
          </span>


          {!isRefused && answer && (

            <button
              type="button"
              className="copy-answer-button"
              onClick={
                handleCopyAnswer
              }
            >

              {copied ? (
                <Check size={12} />
              ) : (
                <Copy size={12} />
              )}

              {copied
                ? "Copied"
                : "Copy"}

            </button>

          )}

        </div>


        {/* REFUSE */}

        {isRefused ? (

          <div className="refusal-content">

            <div className="clinical-boundary">

              <div className="clinical-boundary-mark">
                <ShieldX size={17} />
              </div>

              <div>

                <h3>
                  A safe answer isn't available
                </h3>

                <p>
                  The retrieved evidence does not
                  provide enough support for a safe,
                  grounded response to this question.
                </p>

              </div>

            </div>

          </div>

        ) : (

          <>

            {/* CAUTION */}

            {isCaution && (

              <div className="clinical-caution">

                <div className="clinical-caution-icon">

                  <AlertTriangle size={14} />

                </div>

                <div>

                  <strong>
                    Clinical judgment required
                  </strong>

                  <p>
                    This information is evidence-grounded,
                    but it should not replace individualized
                    clinical assessment or professional judgment.
                  </p>

                </div>

              </div>

            )}


            {/* ANSWER TEXT */}

            <div className="answer-text">

              {answer || (
                "No clinical response was returned."
              )}

            </div>


            {/* CITATIONS */}

            {sources.length > 0 && (

              <div className="answer-citations">

                <span className="citation-heading">
                  EVIDENCE THREADS
                </span>


                <div className="citation-list">

                  {sources.map(
                    (source, index) => (

                      <Citation

                        key={
                          source.id ||
                          index
                        }

                        id={
                          source.id ||
                          index
                        }

                        document={
                          source.document ||
                          "Clinical Evidence"
                        }

                        section={
                          source.section ||
                          "Clinical evidence"
                        }

                        page={
                          source.page
                        }

                        active={
                          source.id ===
                          activeSourceId
                        }

                        onClick={
                          onCitationClick
                        }

                        citationRef={
                          citationRefs
                        }

                      />

                    )
                  )}

                </div>

              </div>

            )}

          </>

        )}

      </div>

    </article>

  );
}


export default AssistantAnswer;