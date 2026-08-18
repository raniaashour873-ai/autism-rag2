import {
  useEffect,
  useRef,
  useState,
} from "react";

import AppShell from "../../components/layout/AppShell";

import ChatHeader from "../../components/chat/ChatHeader";
import EmptyChat from "../../components/chat/EmptyChat";
import QuestionComposer from "../../components/chat/QuestionComposer";
import UserQuestion from "../../components/chat/UserQuestion";
import AssistantAnswer from "../../components/chat/AssistantAnswer";

import EvidencePanel from "../../components/evidence/EvidencePanel";
import EvidenceDetails from "../../components/evidence/EvidenceDetails";

import ThreadConnection from "../../components/threadline/ThreadConnection";

import UtilityPanel from "../../components/layout/UtilityPanel";

import { askClinicalQuestion } from "../../services/api";


function ClinicalAI() {

  const [question, setQuestion] =
    useState("");

  const [messages, setMessages] =
    useState([]);

  const [activeSourceId, setActiveSourceId] =
    useState(null);

  const [selectedEvidence, setSelectedEvidence] =
    useState(null);

  const [utilityPanel, setUtilityPanel] =
    useState(null);

  const [isLoading, setIsLoading] =
    useState(false);

  const [error, setError] =
    useState(null);

  const [lastFailedQuestion, setLastFailedQuestion] =
    useState("");

  const [mobileEvidenceOpen, setMobileEvidenceOpen] =
    useState(false);


  const chatScrollRef =
    useRef(null);

  const citationRefs =
    useRef({});

  const evidenceRefs =
    useRef({});


  // =========================================================
  // AUTO SCROLL
  // =========================================================

  useEffect(() => {

    const container =
      chatScrollRef.current;

    if (!container) {
      return;
    }

    requestAnimationFrame(() => {

      container.scrollTo({
        top: container.scrollHeight,
        behavior: "smooth",
      });

    });

  }, [
    messages.length,
    isLoading,
    error,
  ]);


  // =========================================================
  // ESC CLOSE
  // =========================================================

  useEffect(() => {

    function handleEscape(event) {

      if (event.key !== "Escape") {
        return;
      }

      setUtilityPanel(null);

      setSelectedEvidence(null);

      setMobileEvidenceOpen(false);

    }


    window.addEventListener(
      "keydown",
      handleEscape
    );


    return () => {

      window.removeEventListener(
        "keydown",
        handleEscape
      );

    };

  }, []);


  // =========================================================
  // REFS
  // =========================================================

  function registerCitation(
    id,
    element
  ) {

    if (!id) {
      return;
    }

    if (!element) {

      delete citationRefs.current[id];

      return;
    }

    citationRefs.current[id] =
      element;
  }


  function registerEvidence(
    id,
    element
  ) {

    if (!id) {
      return;
    }

    if (!element) {

      delete evidenceRefs.current[id];

      return;
    }

    evidenceRefs.current[id] =
      element;
  }


  // =========================================================
  // NAVIGATION
  // =========================================================

  function goToEvidence() {

    setUtilityPanel(null);

    setMobileEvidenceOpen(true);


    setTimeout(() => {

      const panel =
        document.querySelector(
          ".evidence-panel"
        );


      if (panel) {

        panel.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
        });

      }


      panel?.classList.add(
        "evidence-panel-highlight"
      );


      setTimeout(() => {

        panel?.classList.remove(
          "evidence-panel-highlight"
        );

      }, 1200);

    }, 80);

  }


  function goToGuidelines() {

    setMobileEvidenceOpen(false);

    setUtilityPanel(
      "guidelines"
    );

  }


  function goToHistory() {

    setMobileEvidenceOpen(false);

    setUtilityPanel(
      "history"
    );

  }


  function goToSafety() {

    setMobileEvidenceOpen(false);

    setUtilityPanel(
      "safety"
    );

  }


  // =========================================================
  // SUGGESTIONS
  // =========================================================

  function handleSuggestion(text) {

    setQuestion(text);

    setError(null);

  }


  // =========================================================
  // SEND
  // =========================================================

  async function sendQuestion(
    questionText
  ) {

    const cleanQuestion =
      questionText.trim();


    if (!cleanQuestion) {
      return;
    }


    if (isLoading) {
      return;
    }


    setError(null);

    setIsLoading(true);

    setActiveSourceId(null);

    setSelectedEvidence(null);


    try {

      const result =
        await askClinicalQuestion(
          cleanQuestion,
          5
        );


      if (
        !result ||
        typeof result !== "object"
      ) {

        throw new Error(
          "Invalid response from Clinical AI."
        );

      }


      const rawSources =
        Array.isArray(result.sources)
          ? result.sources
          : [];


      const sources =
        rawSources.map(
          (source, index) => ({

            ...source,

            id:
              `source-${Date.now()}-${index}`,

            index:
              index + 1,

            document:
              source.document ||
              "Clinical Evidence",

          })
        );


      const message = {

        id:
          `message-${Date.now()}-${Math.random()
            .toString(36)
            .slice(2, 8)}`,

        question:
          cleanQuestion,

        answer:
          typeof result.answer === "string"
            ? result.answer
            : "",

        safetyLabel:
          result.safety_label ||
          "REFUSE",

        status:
          result.status ||
          "refused",

        sources,

      };


      setMessages(
        previous => [
          ...previous,
          message,
        ]
      );


      setQuestion("");

      setLastFailedQuestion("");


    } catch (requestError) {

      console.error(
        "Clinical AI request failed:",
        requestError
      );


      setLastFailedQuestion(
        cleanQuestion
      );


      setError(
        requestError?.message ||
        "Unable to connect to Clinical AI."
      );


    } finally {

      setIsLoading(false);

    }

  }


  function handleSubmit() {

    sendQuestion(question);

  }


  function handleRetry() {

    if (!lastFailedQuestion) {
      return;
    }

    sendQuestion(
      lastFailedQuestion
    );

  }


  // =========================================================
  // SOURCE
  // =========================================================

  function handleSourceSelect(
    sourceId
  ) {

    setActiveSourceId(
      sourceId
    );

  }


  function handleOpenSource(
    source
  ) {

    if (!source) {
      return;
    }

    setSelectedEvidence(
      source
    );

    setActiveSourceId(
      source.id
    );

  }


  function handleCloseEvidence() {

    setSelectedEvidence(
      null
    );

  }


  // =========================================================
  // NEW QUESTION
  // =========================================================

  function handleNewQuestion() {

    setMessages([]);

    setQuestion("");

    setError(null);

    setActiveSourceId(null);

    setSelectedEvidence(null);

    setUtilityPanel(null);

    setMobileEvidenceOpen(false);

    setLastFailedQuestion("");

    citationRefs.current = {};

    evidenceRefs.current = {};


    setTimeout(() => {

      document
        .querySelector(
          ".composer textarea"
        )
        ?.focus();

    }, 100);

  }


  // =========================================================
  // LATEST
  // =========================================================

  const latestMessage =
    messages.length
      ? messages[
          messages.length - 1
        ]
      : null;


  // =========================================================
  // UI
  // =========================================================

  return (

    <AppShell

      onNewQuestion={
        handleNewQuestion
      }

      onEvidence={
        goToEvidence
      }

      onGuidelines={
        goToGuidelines
      }

      onHistory={
        goToHistory
      }

      onSafety={
        goToSafety
      }

      messages={
        messages
      }

    >

      <div className="clinical-workspace">


        {/* =================================================
            CHAT
            ================================================= */}

        <section
          className="clinical-chat"
          id="clinical-ai"
        >

          <ChatHeader />


          <div
            className="chat-scroll"
            ref={chatScrollRef}
          >

            {messages.length === 0 ? (

              <EmptyChat
                onSuggestion={
                  handleSuggestion
                }
              />

            ) : (

              <div className="conversation">

                {messages.map(
                  (message, index) => (

                    <div
                      id={
                        `message-${message.id}`
                      }

                      className="conversation-turn"

                      key={
                        message.id ||
                        index
                      }
                    >

                      <UserQuestion
                        question={
                          message.question
                        }
                      />


                      <AssistantAnswer

                        answer={
                          message.answer
                        }

                        safetyLabel={
                          message.safetyLabel
                        }

                        sources={
                          message.sources
                        }

                        activeSourceId={
                          activeSourceId
                        }

                        onCitationClick={
                          handleSourceSelect
                        }

                        citationRefs={
                          registerCitation
                        }

                      />

                    </div>

                  )
                )}

              </div>

            )}


            {/* LOADING */}

            {isLoading && (

              <div className="rag-loading">

                <div className="rag-loading-line">

                  <div className="rag-loading-dot" />

                  <div className="rag-loading-dot" />

                  <div className="rag-loading-dot" />

                </div>


                <div>

                  <span className="rag-loading-title">
                    THREADLINE IS SEARCHING
                  </span>

                  <span className="rag-loading-text">
                    Retrieving relevant clinical evidence...
                  </span>

                </div>

              </div>

            )}


            {/* ERROR */}

            {error && (

              <div className="api-error">

                <div className="api-error-title">
                  Connection interrupted
                </div>


                <p>
                  {error}
                </p>


                <div className="api-error-actions">

                  <button
                    type="button"
                    onClick={
                      handleRetry
                    }
                  >
                    Try again
                  </button>


                  <button
                    type="button"
                    onClick={() => {

                      setError(null);

                      setLastFailedQuestion("");

                    }}
                  >
                    Dismiss
                  </button>

                </div>

              </div>

            )}

          </div>


          <QuestionComposer

            value={question}

            onChange={setQuestion}

            onSubmit={handleSubmit}

            disabled={isLoading}

          />

        </section>


        {/* =================================================
            EVIDENCE
            ================================================= */}

        <div
          id="evidence"
          className={
            mobileEvidenceOpen
              ? "evidence-mobile-open"
              : ""
          }
        >

          <EvidencePanel

            sources={
              latestMessage?.sources ||
              []
            }

            activeSourceId={
              activeSourceId
            }

            onSourceSelect={
              handleSourceSelect
            }

            onOpenSource={
              handleOpenSource
            }

            evidenceRefs={
              registerEvidence
            }

          />

        </div>


        {/* =================================================
            THREADLINE
            ================================================= */}

        <ThreadConnection

          activeSourceId={
            activeSourceId
          }

          citationRefs={
            citationRefs
          }

          evidenceRefs={
            evidenceRefs
          }

        />


        {/* =================================================
            EVIDENCE DETAILS
            ================================================= */}

        {selectedEvidence && (

          <EvidenceDetails

            source={
              selectedEvidence
            }

            onClose={
              handleCloseEvidence
            }

          />

        )}


        {/* =================================================
            UTILITY PANEL
            ================================================= */}

        {utilityPanel && (

          <UtilityPanel

            type={
              utilityPanel
            }

            messages={
              messages
            }

            onClose={() =>
              setUtilityPanel(null)
            }

          />

        )}

      </div>

    </AppShell>

  );
}


export default ClinicalAI;