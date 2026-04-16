"""MCP bridge to ai-jam-sessions (stub — connect to real MCP in production)."""

import logging
from services.context_assembler import get_exemplar

logger = logging.getLogger(__name__)


def fetch_exemplar_from_mcp(genre: str) -> str:
    """Fetch exemplar context from MCP ai-jam-sessions server.

    In production, this calls the MCP server via subprocess/stdio.
    For hackathon, falls back to local exemplar database.
    """
    try:
        # TODO: Connect to real MCP server
        # import subprocess, json
        # result = subprocess.run(
        #     ["npx", "@anthropic/mcp-client", "call", "get_song_by_genre", genre],
        #     capture_output=True, text=True, timeout=10
        # )
        # return json.loads(result.stdout)["content"]
        raise NotImplementedError("MCP bridge not connected")
    except Exception:
        logger.info(f"MCP unavailable, using local exemplar for genre: {genre}")
        return get_exemplar(genre)


def play_song_via_mcp(midi_path: str) -> bool:
    """Play a MIDI file through the MCP ai-jam-sessions cockpit.

    In production, calls the play_song MCP tool.
    """
    try:
        # TODO: Connect to real MCP server
        # import subprocess
        # subprocess.run(
        #     ["npx", "@anthropic/mcp-client", "call", "play_song", midi_path],
        #     timeout=30
        # )
        raise NotImplementedError("MCP playback not connected")
    except Exception:
        logger.info(f"MCP playback unavailable for: {midi_path}")
        return False
