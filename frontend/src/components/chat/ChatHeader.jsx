import Badge from "../ui/Badge";

function ChatHeader() {
  return (
    <header className="chat-header">

      <div>

        <div className="chat-header-meta">

          <Badge>
            CLINICAL AI
          </Badge>

          <span className="chat-header-divider">
            /
          </span>

          <span>
            AUTISM SPECTRUM DISORDER
          </span>

        </div>

        <h1>
          Clinical evidence assistant
        </h1>

        <p>
          Ask a clinical question and trace every
          answer back to its supporting evidence.
        </p>

      </div>

    </header>
  );
}

export default ChatHeader;