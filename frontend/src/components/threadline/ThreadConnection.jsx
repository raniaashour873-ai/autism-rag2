import { useEffect, useState } from "react";

function ThreadConnection({
  activeSourceId,
  citationRefs,
  evidenceRefs,
}) {
  const [path, setPath] = useState(null);

  useEffect(() => {
    if (!activeSourceId) {
      setPath(null);
      return;
    }

    const citationElement =
      citationRefs?.current?.[activeSourceId];

    const evidenceElement =
      evidenceRefs?.current?.[activeSourceId];

    if (!citationElement || !evidenceElement) {
      setPath(null);
      return;
    }

    const updatePath = () => {
      const citationRect =
        citationElement.getBoundingClientRect();

      const evidenceRect =
        evidenceElement.getBoundingClientRect();

      const workspace =
        citationElement.closest(
          ".clinical-workspace"
        );

      if (!workspace) {
        return;
      }

      const workspaceRect =
        workspace.getBoundingClientRect();

      const startX =
        citationRect.right -
        workspaceRect.left;

      const startY =
        citationRect.top +
        citationRect.height / 2 -
        workspaceRect.top;

      const endX =
        evidenceRect.left -
        workspaceRect.left;

      const endY =
        evidenceRect.top +
        evidenceRect.height / 2 -
        workspaceRect.top;

      const distance =
        Math.abs(endX - startX);

      const curve =
        Math.max(50, distance * 0.45);

      const d = `
        M ${startX} ${startY}
        C
        ${startX + curve} ${startY},
        ${endX - curve} ${endY},
        ${endX} ${endY}
      `;

      setPath({
        d,
        startX,
        startY,
        endX,
        endY,
      });
    };

    updatePath();

    window.addEventListener(
      "resize",
      updatePath
    );

    window.addEventListener(
      "scroll",
      updatePath,
      true
    );

    return () => {
      window.removeEventListener(
        "resize",
        updatePath
      );

      window.removeEventListener(
        "scroll",
        updatePath,
        true
      );
    };
  }, [
    activeSourceId,
    citationRefs,
    evidenceRefs,
  ]);

  if (!path) {
    return null;
  }

  return (
    <svg
      className="thread-connection"
      aria-hidden="true"
    >
      <path
        className="thread-connection-shadow"
        d={path.d}
      />

      <path
        className="thread-connection-path"
        d={path.d}
      />

      <circle
        className="thread-connection-start"
        cx={path.startX}
        cy={path.startY}
        r="4"
      />

      <circle
        className="thread-connection-end"
        cx={path.endX}
        cy={path.endY}
        r="4"
      />
    </svg>
  );
}

export default ThreadConnection;