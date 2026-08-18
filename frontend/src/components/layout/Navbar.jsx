import Logo from "../brand/Logo";

function Navbar() {
  return (
    <header className="site-navbar">

      <div className="navbar-inner">

        <Logo />

        <nav className="navbar-links">

          <a href="#platform">
            Platform
          </a>

          <a href="#evidence">
            Evidence
          </a>

          <a href="#guidelines">
            Guidelines
          </a>

          <a href="#safety">
            Safety
          </a>

        </nav>

        <a
          href="/app"
          className="navbar-cta"
        >
          Open Clinical AI
          <span>↗</span>
        </a>

      </div>

    </header>
  );
}

export default Navbar;
