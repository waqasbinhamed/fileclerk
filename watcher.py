"""Watchdog-based folder watcher for automatic re-indexing."""
import logging
import time
from pathlib import Path

from rich.console import Console
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from extractor import SUPPORTED_EXTENSIONS
from indexer import Indexer

console = Console()
logger = logging.getLogger(__name__)


class _DocumentHandler(FileSystemEventHandler):
    def __init__(self, indexer: Indexer, root: Path):
        self.indexer = indexer
        self.root = root

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_index(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe_index(Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory:
            console.print(f"[yellow]Deleted: {event.src_path} — will be removed on next full sync[/yellow]")

    def _maybe_index(self, path: Path) -> None:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        console.print(f"[blue]Change detected, re-indexing: {path.name}[/blue]")
        # Re-scan the root; hash check skips unchanged files
        self.indexer.index_path(self.root, force=False)


def watch(path: str, indexer: Indexer) -> None:
    """Block and watch *path*, auto-indexing on file create/modify. Ctrl-C to stop."""
    root = Path(path)
    handler = _DocumentHandler(indexer, root)
    observer = Observer()
    observer.schedule(handler, str(root), recursive=True)
    observer.start()
    console.print(f"[bold green]Watching {root} for changes… (Ctrl-C to stop)[/bold green]")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
