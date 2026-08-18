function Button({
  children,
  variant = "primary",
  href,
  onClick,
}) {
  const className = `tl-button tl-button-${variant}`;

  if (href) {
    return (
      <a
        href={href}
        className={className}
      >
        {children}
      </a>
    );
  }

  return (
    <button
      className={className}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

export default Button;