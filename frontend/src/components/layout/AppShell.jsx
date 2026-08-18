import ChatSidebar from "./ChatSidebar";


function AppShell({
  children,
  onNewQuestion,
  onEvidence,
  onGuidelines,
  onHistory,
  onSafety,
  messages = [],
}) {

  return (

    <div className="app-shell">

      {/* TOP BAR */}

      <header className="workspace-topbar">

        <div className="workspace-brand">

          <button
            type="button"
            className="workspace-brand-mark-button"
            onClick={onNewQuestion}
            aria-label="Threadline home"
          >
            <span className="workspace-brand-mark">
              ∞
            </span>
          </button>

          <div>

            <strong>
              THREADLINE
            </strong>

            <span>
              Clinical Evidence Intelligence
            </span>

          </div>

        </div>


        {/* TOP NAV */}

        <nav className="workspace-nav">

          <button
            type="button"
            className="workspace-nav-active"
            onClick={onNewQuestion}
          >
            Clinical AI
          </button>


          <button
            type="button"
            onClick={onEvidence}
          >
            Evidence
          </button>


          <button
            type="button"
            onClick={onGuidelines}
          >
            Guidelines
          </button>

        </nav>


        {/* STATUS */}

        <div className="workspace-status">

          <span className="workspace-status-dot" />

          System ready

        </div>

      </header>


      {/* BODY */}

      <div className="workspace-body">

        <ChatSidebar

          onNewQuestion={
            onNewQuestion
          }

          onEvidence={
            onEvidence
          }

          onGuidelines={
            onGuidelines
          }

          onHistory={
            onHistory
          }

          onSafety={
            onSafety
          }

          messages={
            messages
          }

        />


        <main className="workspace-main">

          {children}

        </main>

      </div>

    </div>

  );
}


export default AppShell;