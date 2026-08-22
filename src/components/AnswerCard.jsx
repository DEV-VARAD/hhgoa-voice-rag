import { Sparkles, ShieldCheck, BookOpen } from "lucide-react";

function AnswerCard({ result }) {
  return (
    <div className="answer-card">
      <div className="answer-header">
        <div className="card-label">
          <Sparkles size={17} />
          GROUNDED ANSWER
        </div>

        {result.grounded && (
          <div className="grounded-badge">
            <ShieldCheck size={15} />
            Grounded
          </div>
        )}
      </div>

      <div className="answer-text">
        {result.answer}
      </div>

      {result.sources?.length > 0 && (
        <div className="sources">
          <div className="sources-title">
            <BookOpen size={16} />
            Retrieved context
          </div>

          <div className="source-list">
            {result.sources.map((source, index) => (
              <div className="source-item" key={index}>
                <span className="source-number">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <p>{source.text || source}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default AnswerCard;