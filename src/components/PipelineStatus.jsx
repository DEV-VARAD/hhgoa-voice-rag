import { Mic, Search, Sparkles, Check } from "lucide-react";

function PipelineStatus({ stage }) {
  const steps = [
    { id: "voice", label: "Voice input", icon: Mic },
    { id: "retrieving", label: "Retrieval", icon: Search },
    { id: "generating", label: "Generation", icon: Sparkles },
    { id: "complete", label: "Answer", icon: Check },
  ];

  const stageOrder = {
    idle: 0,
    voice: 1,
    retrieving: 2,
    generating: 3,
    complete: 4,
  };

  const current = stageOrder[stage] || 0;

  return (
    <div className="pipeline">
      {steps.map((step, index) => {
        const Icon = step.icon;
        const stepNumber = index + 1;

        const active =
          (step.id === "voice" && current === 1) ||
          (step.id === "retrieving" && current === 2) ||
          (step.id === "generating" && current === 3) ||
          (step.id === "complete" && current === 4);

        const done = current > stepNumber;

        return (
          <div
            className={`pipeline-step ${
              active ? "active" : ""
            } ${done ? "done" : ""}`}
            key={step.id}
          >
            <div className="step-icon">
              <Icon size={17} />
            </div>
            <span>{step.label}</span>
          </div>
        );
      })}
    </div>
  );
}

export default PipelineStatus;