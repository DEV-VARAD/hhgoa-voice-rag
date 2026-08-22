import { FileAudio } from "lucide-react";

function TranscriptCard({ transcript }) {
  return (
    <div className="content-card transcript-card">
      <div className="card-label"><FileAudio size={17} />TRANSCRIPTION</div>
      <p className="transcript-text">“{transcript}”</p>
    </div>
  );
}

export default TranscriptCard;
