import {
  Check,
  AlertTriangle,
  ShieldX,
} from "lucide-react";


function SafetyStatus({
  label = "ALLOWED",
}) {

  const normalizedLabel =
    String(label).toUpperCase();


  const config = {

    ALLOWED: {
      icon: Check,
      title: "Evidence grounded",
      text: "Supported by retrieved clinical evidence.",
      className: "safety-allowed",
    },

    NEEDS_CAUTION: {
      icon: AlertTriangle,
      title: "Clinical caution",
      text: "This response should be interpreted with clinical judgment.",
      className: "safety-caution",
    },

    REFUSE: {
      icon: ShieldX,
      title: "Safety boundary",
      text: "The system cannot safely provide this answer.",
      className: "safety-refuse",
    },

  };


  const current =
    config[normalizedLabel] ||
    config.REFUSE;


  const Icon =
    current.icon;


  return (

    <div
      className={`safety-status ${current.className}`}
      title={current.text}
    >

      <span className="safety-status-icon">

        <Icon size={13} />

      </span>


      <span className="safety-status-content">

        <span className="safety-status-title">
          {current.title}
        </span>

        <span className="safety-status-label">
          {normalizedLabel}
        </span>

      </span>

    </div>

  );
}


export default SafetyStatus;