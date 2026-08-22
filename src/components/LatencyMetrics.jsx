import { Gauge } from "lucide-react";

function LatencyMetrics({ latency = {} }) {
  const entries = [
    ["STT", latency.stt],
    ["Retrieval", latency.retrieval],
    ["Generation", latency.generation],
    ["Total", latency.total],
  ];

  return (
    <div className="latency-card">
      <div className="card-label"><Gauge size={17} />PIPELINE LATENCY</div>
      <div className="latency-grid">
        {entries.map(([label, value]) => (
          <div className="latency-item" key={label}>
            <span>{label}</span>
            <strong>{value ?? "—"}<small> ms</small></strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default LatencyMetrics;
