import {
  X,
  FileText,
  ExternalLink,
} from "lucide-react";


function EvidenceDetails({
  source,
  onClose,
}) {

  if (!source) {
    return null;
  }


  return (

    <div className="evidence-details-overlay">

      <div className="evidence-details">


        <div className="evidence-details-header">

          <div>

            <span className="evidence-kicker">
              SOURCE DETAILS
            </span>

            <h2>
              Evidence {source.index}
            </h2>

          </div>


          <button
            type="button"
            className="evidence-details-close"
            onClick={onClose}
            aria-label="Close evidence"
          >
            <X size={16} />
          </button>

        </div>


        <div className="evidence-details-document">

          <div className="evidence-file-icon">
            <FileText size={16} />
          </div>

          <div>

            <strong>
              {source.document ||
                "Clinical Evidence"}
            </strong>

            <span>
              Retrieved by Threadline RAG
            </span>

          </div>

        </div>


        <div className="evidence-details-grid">


          <div className="evidence-detail">

            <span>
              SECTION
            </span>

            <strong>
              {source.section ||
                "Clinical evidence"}
            </strong>

          </div>


          <div className="evidence-detail">

            <span>
              PAGE
            </span>

            <strong>
              {source.page ?? "—"}
            </strong>

          </div>


          <div className="evidence-detail">

            <span>
              DISTANCE
            </span>

            <strong>
              {typeof source.distance === "number"
                ? source.distance.toFixed(3)
                : "—"}
            </strong>

          </div>

        </div>


        <div className="evidence-detail-note">

          <span className="evidence-kicker">
            RETRIEVAL NOTE
          </span>

          <p>
            This source was retrieved by the
            clinical RAG pipeline as supporting
            evidence for the current response.
          </p>

        </div>


        <button
          type="button"
          className="evidence-open"
          onClick={() => {

            /*
             * The current API does not yet return
             * a public source URL.
             *
             * We intentionally don't invent one.
             */

          }}
        >

          Open source

          <ExternalLink size={13} />

        </button>

      </div>

    </div>

  );
}


export default EvidenceDetails;