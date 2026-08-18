import {
  ExternalLink,
  FileText,
} from "lucide-react";


function EvidenceItem({
  source,
  index,
  active = false,
  onClick,
  onOpen,
  evidenceRef,
}) {

  return (

    <article
      ref={(element) => {
        evidenceRef?.(source.id, element);
      }}

      className={`evidence-item ${
        active
          ? "evidence-item-active"
          : ""
      }`}

      onClick={() =>
        onClick?.(source.id)
      }
    >


      <div className="evidence-item-top">

        <span className="evidence-number">
          {String(index + 1).padStart(2, "0")}
        </span>

        <span className="evidence-type">
          RETRIEVED
        </span>

      </div>


      <div className="evidence-title-row">

        <div className="evidence-file-icon">

          <FileText size={14} />

        </div>


        <h3>
          {source.section ||
            "Clinical evidence"}
        </h3>

      </div>


      <div className="evidence-document">

        {source.document ||
          "Clinical Evidence"}

      </div>


      <div className="evidence-meta">

        <span>
          Page {source.page ?? "—"}
        </span>


        <span className="distance-value">

          distance{" "}

          {typeof source.distance === "number"
            ? source.distance.toFixed(3)
            : "—"}

        </span>

      </div>


      <div className="evidence-relevance">

        <div className="relevance-track">

          <div
            className="relevance-fill"
            style={{
              width:
                typeof source.distance ===
                "number"
                  ? `${Math.max(
                      8,
                      Math.min(
                        100,
                        (1 -
                          source.distance) *
                          100
                      )
                    )}%`
                  : "8%",
            }}
          />

        </div>

      </div>


      <button
        type="button"
        className="evidence-open"
        onClick={(event) => {

          event.stopPropagation();

          onOpen?.(source);

        }}
      >

        View details

        <ExternalLink size={13} />

      </button>

    </article>

  );
}


export default EvidenceItem;