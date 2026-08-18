function Citation({
  id,
  document = "NICE CG142",
  page = 23,
  section = "Assessment",
  active = false,
  onClick,
  citationRef,
}) {
  return (
    <button
      ref={(element) => {
        citationRef?.(id, element);
      }}
      type="button"
      className={`citation ${
        active ? "citation-active" : ""
      }`}
      onClick={() => onClick?.(id)}
      aria-label={`View ${document}, ${section}, page ${page}`}
    >
      <span className="citation-mark">
        ↗
      </span>

      <span>
        {document}
      </span>

      <span className="citation-separator">
        ·
      </span>

      <span>
        {section}
      </span>

      <span className="citation-separator">
        ·
      </span>

      <span>
        p.{page}
      </span>
    </button>
  );
}

export default Citation;