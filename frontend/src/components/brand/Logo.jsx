function Logo({ light = false }) {
  return (
    <a
      href="/"
      className={`threadline-logo ${
        light ? "threadline-logo-light" : ""
      }`}
      aria-label="Threadline home"
    >
      <span className="threadline-logo-mark">
        ∞
      </span>

      <span className="threadline-logo-name">
        Threadline
      </span>
    </a>
  );
}

export default Logo;