import { Music } from "lucide-react";
import type { TabView, Genre } from "../types/music";
import GenreSelector from "./GenreSelector";

interface HeaderProps {
  activeTab: TabView;
  onTabChange: (tab: TabView) => void;
  genre: Genre;
  onGenreChange: (genre: Genre) => void;
}

const TABS: { id: TabView; label: string }[] = [
  { id: "analyze", label: "Analyze" },
  { id: "history", label: "History" },
  { id: "journal", label: "Journal" },
];

export default function Header({ activeTab, onTabChange, genre, onGenreChange }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-border-subtle">
      {/* Logo */}
      <div className="flex items-center gap-2">
        <Music className="w-6 h-6 text-accent-purple" />
        <span className="text-xl font-bold">
          <span className="text-accent-purple">Jam</span>
          <span className="text-text-primary">Sight</span>
        </span>
      </div>

      {/* Tabs */}
      <nav className="flex gap-1 bg-bg-surface rounded-lg p-1">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? "bg-bg-surface-2 text-text-primary"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Genre selector */}
      <GenreSelector genre={genre} onChange={onGenreChange} />
    </header>
  );
}
