import type { Genre } from "../types/music";

interface GenreSelectorProps {
  genre: Genre;
  onChange: (genre: Genre) => void;
}

const GENRES: { id: Genre; label: string; color: string }[] = [
  { id: "blues", label: "Blues", color: "bg-accent-purple" },
  { id: "jazz", label: "Jazz", color: "bg-accent-blue" },
  { id: "pop", label: "Pop", color: "bg-accent-green" },
  { id: "rock", label: "Rock", color: "bg-accent-orange" },
  { id: "funk", label: "Funk", color: "bg-accent-yellow" },
];

export default function GenreSelector({ genre, onChange }: GenreSelectorProps) {
  return (
    <div className="flex gap-2">
      {GENRES.map((g) => (
        <button
          key={g.id}
          onClick={() => onChange(g.id)}
          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
            genre === g.id
              ? `${g.color} text-white`
              : "bg-bg-surface text-text-secondary hover:text-text-primary border border-border-subtle"
          }`}
        >
          {g.label}
        </button>
      ))}
    </div>
  );
}
