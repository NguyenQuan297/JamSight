import { useState, useRef } from "react";
import { Upload, Video, X } from "lucide-react";

interface VideoCaptureProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export default function VideoCapture({ onFileSelect, disabled }: VideoCaptureProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  function handleFile(file: File) {
    if (!file.type.startsWith("video/") && !file.type.startsWith("audio/")) {
      return;
    }
    setFileName(file.name);
    if (file.type.startsWith("video/")) {
      setPreview(URL.createObjectURL(file));
    } else {
      setPreview(null);
    }
    onFileSelect(file);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function clear() {
    setPreview(null);
    setFileName(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary">Input</h3>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed
          transition-all cursor-pointer min-h-[200px]
          ${dragOver ? "border-accent-purple bg-accent-purple/5" : "border-border-accent hover:border-accent-purple/50"}
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          ${preview ? "p-2" : "p-8"}`}
      >
        {preview ? (
          <div className="relative w-full">
            <video
              ref={videoRef}
              src={preview}
              className="w-full rounded-lg max-h-[180px] object-cover"
              muted
              playsInline
              onMouseEnter={(e) => (e.target as HTMLVideoElement).play()}
              onMouseLeave={(e) => { const v = e.target as HTMLVideoElement; v.pause(); v.currentTime = 0; }}
            />
            <button
              onClick={(e) => { e.stopPropagation(); clear(); }}
              className="absolute top-2 right-2 p-1 rounded-full bg-bg-primary/80 hover:bg-accent-red/20 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ) : fileName ? (
          <div className="flex flex-col items-center gap-2">
            <Video className="w-10 h-10 text-accent-purple" />
            <span className="text-sm text-text-secondary truncate max-w-[200px]">{fileName}</span>
            <button
              onClick={(e) => { e.stopPropagation(); clear(); }}
              className="text-xs text-accent-red hover:underline"
            >
              Remove
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-text-secondary">
            <div className="p-4 rounded-full bg-bg-surface-2">
              <Upload className="w-8 h-8" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-text-primary">Upload video piano cua ban</p>
              <p className="text-xs mt-1 text-text-muted">MP4 · MOV · max 5 phut</p>
            </div>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="video/*,audio/*"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
          disabled={disabled}
        />
      </div>
    </div>
  );
}
