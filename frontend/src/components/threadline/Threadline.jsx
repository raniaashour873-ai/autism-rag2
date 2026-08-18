function Threadline({
  source = "NICE CG142",
  page = "p.23",
  label = "Evidence source",
}) {
  return (
    <div className="threadline-wrapper">

      <svg
        className="threadline-svg"
        viewBox="0 0 500 300"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >

        <path
          className="threadline-path"
          d="
            M250 40
            C165 40 95 85 95 150
            C95 215 165 260 250 260
            C335 260 405 215 405 150
            C405 85 335 40 250 40

            M250 40
            C335 40 405 85 405 150
            C405 215 335 260 250 260
            C165 260 95 215 95 150
            C95 85 165 40 250 40
          "
        />

        <circle
          className="threadline-end"
          cx="250"
          cy="40"
          r="5"
        />

        <circle
          className="threadline-source-dot"
          cx="250"
          cy="260"
          r="6"
        />

        <line
          className="threadline-stem"
          x1="250"
          y1="260"
          x2="250"
          y2="288"
        />

      </svg>

      <div className="threadline-source">

        <span className="threadline-source-label">
          {label}
        </span>

        <strong>
          {source}
        </strong>

        <span>
          {page}
        </span>

      </div>

    </div>
  );
}

export default Threadline;