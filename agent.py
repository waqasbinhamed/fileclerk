"""Main CLI entry point for pi-doc-agent."""
import argparse
import logging
import sys
import time
from pathlib import Path

import psutil
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from classifier import classify_document
from indexer import Indexer
from sorter import Sorter

console = Console()

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _log_ram(step: str) -> None:
    mem = psutil.virtual_memory()
    console.print(f"[dim]RAM after {step}: {mem.percent:.1f}% ({mem.used // 1024**2} MB used)[/dim]")


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_index(args, config: dict) -> None:
    console.rule("[bold green]Index[/bold green]")
    indexer = Indexer(config)
    indexer.index_path(args.path)
    _log_ram("indexing")
    console.print("[bold green]Indexing complete.[/bold green]")


def cmd_sort(args, config: dict) -> None:
    dry_run = not getattr(args, "execute", False)
    console.rule(f"[bold]Sort — {'DRY-RUN' if dry_run else 'EXECUTE'}[/bold]")

    indexer = Indexer(config)

    console.print(f"[green]Step 1/3 — Indexing {args.path}[/green]")
    indexer.index_path(args.path)
    _log_ram("indexing")

    metadatas, documents = indexer.get_all_documents()
    taxonomy = config.get("taxonomy", [])

    to_classify = [
        (m, d)
        for m, d in zip(metadatas, documents)
        if m and not m.get("classification")
    ]

    console.print(f"[green]Step 2/3 — Classifying {len(to_classify)} document(s)[/green]")
    for m, doc in to_classify:
        extraction = {"filename": m["filename"], "text_snippet": doc}
        classification = classify_document(extraction, taxonomy, config)
        indexer.update_classification(m["path"], classification)
        time.sleep(1)  # avoid memory pressure on Pi between Ollama calls

    _log_ram("classification")

    console.print("[green]Step 3/3 — Sorting[/green]")
    classified = indexer.get_all_classified()
    sorter = Sorter(config, args.output, dry_run=dry_run)
    sorter.sort_all(classified)


def cmd_query(args, config: dict) -> None:
    query_string = " ".join(args.query_string)
    console.rule(f'[bold]Query: "{query_string}"[/bold]')

    indexer = Indexer(config)
    results = indexer.query(query_string)

    table = Table(title=f'Top {len(results)} results for "{query_string}"')
    table.add_column("Filename", style="cyan", max_width=35)
    table.add_column("Category")
    table.add_column("Conf.")
    table.add_column("Similarity")
    table.add_column("Path", max_width=50)

    for r in results:
        table.add_row(
            r.get("filename", ""),
            r.get("classification", "—"),
            f"{float(r.get('confidence', 0)):.2f}",
            f"{float(r.get('similarity', 0)):.3f}",
            r.get("path", ""),
        )

    console.print(table)


def cmd_review(args, config: dict) -> None:
    console.rule("[bold yellow]Files Pending Review[/bold yellow]")

    indexer = Indexer(config)
    pending = indexer.get_needs_review()

    if not pending:
        console.print("[green]No documents pending review.[/green]")
        return

    table = Table(title=f"Pending Review ({len(pending)} documents)")
    table.add_column("Filename", style="cyan", max_width=40)
    table.add_column("Type")
    table.add_column("Date Indexed")
    table.add_column("Path", max_width=50)

    for m in pending:
        table.add_row(
            m.get("filename", ""),
            m.get("filetype", ""),
            m.get("date_indexed", "")[:10],
            m.get("path", ""),
        )

    console.print(table)


def cmd_sync(args, config: dict) -> None:
    console.rule("[bold green]Sync[/bold green]")
    indexer = Indexer(config)
    indexer.index_path(args.path)  # hash check built into index_path
    _log_ram("sync")
    console.print("[bold green]Sync complete.[/bold green]")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description="Pi Document Agent — local LLM document indexing, classification, and organisation",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="Index documents into the vector store")
    p.add_argument("--path", required=True, help="Folder to index")

    p = sub.add_parser("sort", help="Index + classify + sort documents")
    p.add_argument("--path", required=True, help="Source documents folder")
    p.add_argument("--output", required=True, help="Output root for organised files")
    p.add_argument("--execute", action="store_true", help="Actually move files (default: dry-run)")

    p = sub.add_parser("query", help="Semantic search over indexed documents")
    p.add_argument("query_string", nargs="+", help="Search query words")

    sub.add_parser("review", help="List documents pending review (low confidence)")

    p = sub.add_parser("sync", help="Re-index changed files only (uses hash check)")
    p.add_argument("--path", required=True, help="Folder to sync")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    dispatch = {
        "index": cmd_index,
        "sort": cmd_sort,
        "query": cmd_query,
        "review": cmd_review,
        "sync": cmd_sync,
    }
    dispatch[args.command](args, config)


if __name__ == "__main__":
    main()
