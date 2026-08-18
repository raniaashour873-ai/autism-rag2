import {
  BookOpen,
  Sparkles,
} from "lucide-react";

import EvidenceItem from "./EvidenceItem";


function EvidencePanel({
  sources = [],
  activeSourceId,
  onSourceSelect,
  onOpenSource,
  evidenceRefs,
}) {

  return (

    <aside className="evidence-panel">


      <div className="evidence-panel-header">

        <div>

          <span className="evidence-kicker">
            RETRIEVAL
          </span>

          <h2>
            Evidence
          </h2>

        </div>


        <BookOpen size={18} />

      </div>


      <div className="evidence-summary">

        <span>
          {sources.length} sources retrieved
        </span>


        <span className="retrieval-indicator">

          <Sparkles size={10} />

          TRACEABLE

        </span>

      </div>


      {sources.length === 0 ? (

        <div className="evidence-panel-empty">

          <div className="evidence-panel-symbol">
            ∞
          </div>

          <h3>
            Evidence will appear here.
          </h3>

          <p>
            Ask a clinical question to see
            the sources retrieved by the
            RAG system.
          </p>

        </div>

      ) : (

        <div className="evidence-items">

          {sources.map(
            (source, index) => (

              <EvidenceItem

                key={
                  source.id ||
                  index
                }

                source={source}

                index={index}

                active={
                  source.id ===
                  activeSourceId
                }

                onClick={
                  onSourceSelect
                }

                onOpen={
                  onOpenSource
                }

                evidenceRef={
                  evidenceRefs
                }

              />

            )
          )}

        </div>

      )}

    </aside>

  );
}


export default EvidencePanel;