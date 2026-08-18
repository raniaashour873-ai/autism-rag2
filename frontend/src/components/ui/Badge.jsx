function Badge({
  children,
  variant = "sage",
}) {
  return (
    <span
      className={`tl-badge tl-badge-${variant}`}
    >
      <span className="tl-badge-dot" />

      {children}
    </span>
  );
}

export default Badge;