"""Interactive script to label a random sample of documents for evaluation."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def load_config() -> dict:
    path = Path(__file__).parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Manually label a sample of documents for evaluation")
    parser.add_argument("--n", type=int, default=50, help="Number of documents to label (default: 50)")
    parser.add_argument("--output", default="eval/ground_truth.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    config = load_config()
    from indexer import Indexer

    indexer = Indexer(config)
    metadatas, documents = indexer.get_all_documents()
    pairs = [(m, d) for m, d in zip(metadatas, documents) if m and d]

    if not pairs:
        console.print("[red]Index is empty. Run 'python agent.py index --path <dir>' first.[/red]")
        sys.exit(1)

    sample = random.sample(pairs, min(args.n, len(pairs)))
    taxonomy = config.get("taxonomy", [])
    taxonomy_display = "\n".join(f"  [bold]{i+1}[/bold]. {cat}" for i, cat in enumerate(taxonomy))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labeled = []
    console.print(f"[bold]Labeling {len(sample)} documents.[/bold]  Type [bold]q[/bold] to quit and save, [bold]s[/bold] to skip.\n")

    for i, (meta, doc) in enumerate(sample):
        console.print(f"\n[bold cyan]Document {i + 1} / {len(sample)}[/bold cyan]")
        console.print(
            Panel(
                doc[:600],
                title=f"[bold]{meta.get('filename', 'unknown')}[/bold]",
                subtitle=meta.get("filetype", ""),
            )
        )
        console.print(f"\nCategories:\n{taxonomy_display}")

        category = None
        while True:
            answer = Prompt.ask("\nEnter number, category name, [bold]s[/bold]=skip, [bold]q[/bold]=quit")

            if answer.lower() == "q":
                break
            if answer.lower() == "s":
                break

            if answer.isdigit():
                idx = int(answer) - 1
                if 0 <= idx < len(taxonomy):
                    category = taxonomy[idx]
                    break
                console.print("[red]Number out of range.[/red]")
            elif answer in taxonomy:
                category = answer
                break
            else:
                console.print("[red]Not a valid category. Try the number or exact name.[/red]")

        if answer.lower() == "q":
            break

        if category is not None:
            labeled.append(
                {
                    "path": meta.get("path"),
                    "filename": meta.get("filename"),
                    "true_category": category,
                    "text_snippet": doc[:600],
                }
            )

    with open(output_path, "w") as f:
        for item in labeled:
            f.write(json.dumps(item) + "\n")

    console.print(f"\n[bold green]Saved {len(labeled)} labeled samples to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
