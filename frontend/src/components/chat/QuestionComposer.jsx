import {
  ArrowUp,
  Paperclip,
} from "lucide-react";


function QuestionComposer({
  value,
  onChange,
  onSubmit,
  disabled = false,
}) {


  function handleKeyDown(event) {

    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {

      event.preventDefault();

      if (!disabled) {
        onSubmit();
      }

    }

  }


  return (

    <div
      className={[
        "composer",
        disabled
          ? "composer-loading"
          : "",
      ].join(" ")}
    >

      <textarea

        value={value}

        onChange={(event) =>
          onChange(event.target.value)
        }

        onKeyDown={handleKeyDown}

        disabled={disabled}

        rows={3}

        placeholder={
          disabled
            ? "Searching clinical evidence..."
            : "Ask a clinical question..."
        }

        aria-label="Clinical question"

      />


      <div className="composer-footer">

        <button
          type="button"
          className="composer-attach"
          disabled={disabled}
          aria-label="Attach file"
        >

          <Paperclip size={15} />

        </button>


        <span className="composer-hint">

          {disabled
            ? "Retrieving evidence..."
            : "Enter to ask"}

        </span>


        <button
          type="button"
          className="composer-send"
          onClick={onSubmit}
          disabled={
            disabled ||
            !value.trim()
          }
          aria-label="Send question"
        >

          <ArrowUp size={17} />

        </button>

      </div>

    </div>

  );
}


export default QuestionComposer;