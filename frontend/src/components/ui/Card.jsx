function Card({
  children,
  className = "",
}) {
  return (
    <div
      className={`tl-card ${className}`}
    >
      {children}
    </div>
  );
}

export default Card;