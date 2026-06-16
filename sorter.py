"""File move/rename logic with dry-run mode."""
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def resolve_destination(src: Path, output_root: Path, suggested_folder: str, filename: str) -> Path:
    """Return destination path, appending _1, _2, … on conflict. Never overwrites."""
    dest_dir = output_root / suggested_folder
    dest = dest_dir / filename

    if not dest.exists():
        return dest

    name = Path(filename)
    stem, suffix = name.stem, name.suffix
    counter = 1
    while True:
        candidate = dest_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


class Sorter:
    def __init__(self, config: dict, output_root: "str | Path", dry_run: bool = True):
        self.config = config
        self.output_root = Path(output_root)
        self.dry_run = dry_run
        self.log_path = config.get("sort_log", "sort_log.jsonl")

    def sort_document(self, meta: dict) -> dict:
        src = Path(meta["path"])
        category = meta.get("classification", "")
        confidence = float(meta.get("confidence", 0.0))
        destination = meta.get("destination", "Unsorted") or "Unsorted"

        if not src.exists():
            self._log("missing", str(src), "", category, confidence)
            return {"action": "missing", "source": str(src)}

        if category in ("", "needs_review"):
            self._log("skipped", str(src), "", category, confidence, "needs_review")
            return {"action": "skipped", "source": str(src), "reason": "needs_review"}

        dest = resolve_destination(src, self.output_root, destination, src.name)

        if self.dry_run:
            dest_rel = dest.relative_to(self.output_root)
            console.print(f"  [dim]DRY-RUN[/dim]  [cyan]{src.name}[/cyan]  →  [blue]{dest_rel}[/blue]")
            self._log("would_move", str(src), str(dest), category, confidence)
            return {"action": "would_move", "source": str(src), "destination": str(dest)}

        dest.parent.mkdir(parents=True, exist_ok=True)
        note = ""
        if dest != self.output_root / destination / src.name:
            note = "conflict-resolved"
        shutil.move(str(src), str(dest))
        dest_rel = dest.relative_to(self.output_root)
        console.print(f"  [green]MOVED[/green]  [cyan]{src.name}[/cyan]  →  [blue]{dest_rel}[/blue]")
        self._log("moved", str(src), str(dest), category, confidence, note)
        return {"action": "moved", "source": str(src), "destination": str(dest)}

    def sort_all(self, classified_docs: list) -> list:
        table = Table(title="Sort Preview (dry-run)" if self.dry_run else "Sort Results")
        table.add_column("File", style="cyan", max_width=40)
        table.add_column("Category")
        table.add_column("Confidence")
        table.add_column("Action")

        results = []
        for meta in classified_docs:
            result = self.sort_document(meta)
            results.append(result)

            styles = {"moved": "green", "would_move": "blue", "skipped": "yellow", "missing": "red"}
            action = result["action"]
            table.add_row(
                Path(meta["path"]).name,
                meta.get("classification", ""),
                f"{float(meta.get('confidence', 0)):.2f}",
                f"[{styles.get(action, 'white')}]{action}[/{styles.get(action, 'white')}]",
            )

        console.print(table)

        if self.dry_run:
            console.print("\n[yellow]Dry-run complete — pass [bold]--execute[/bold] to actually move files.[/yellow]")

        return results

    def _log(
        self,
        action: str,
        source: str,
        destination: str,
        category: str,
        confidence: float,
        note: str = "",
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "source": source,
            "destination": destination,
            "category": category,
            "confidence": confidence,
            "note": note,
            "dry_run": self.dry_run,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
