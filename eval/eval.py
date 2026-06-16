"""Evaluate classifier precision, recall, and F1 over a labeled ground truth set."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from rich.console import Console
from rich.table import Table

console = Console()


def load_config() -> dict:
    path = Path(__file__).parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(true_labels: list, pred_labels: list, categories: list) -> dict:
    metrics = {}
    for cat in categories:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cat and p == cat)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cat and p == cat)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cat and p != cat)

        if tp + fp + fn == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[cat] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate classifier against labeled ground truth")
    parser.add_argument("--ground-truth", default="eval/ground_truth.jsonl")
    parser.add_argument("--output", default="eval/results.jsonl")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        console.print(f"[red]Ground truth file not found: {gt_path}[/red]")
        console.print("Run [bold]python eval/label_sample.py[/bold] first.")
        sys.exit(1)

    config = load_config()
    taxonomy = config.get("taxonomy", [])

    from classifier import classify_document

    samples = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    console.print(f"[green]Evaluating {len(samples)} labeled samples...[/green]")

    true_labels, pred_labels, details = [], [], []

    for i, sample in enumerate(samples):
        console.print(f"[dim]  {i + 1}/{len(samples)}: {sample['filename']}[/dim]")
        extraction = {"filename": sample["filename"], "text_snippet": sample["text_snippet"]}
        result = classify_document(extraction, taxonomy, config)

        pred_cat = result.get("category", "Unsorted")
        true_cat = sample["true_category"]

        true_labels.append(true_cat)
        pred_labels.append(pred_cat)
        details.append(
            {
                "path": sample.get("path"),
                "filename": sample["filename"],
                "true_category": true_cat,
                "predicted_category": pred_cat,
                "confidence": result.get("confidence", 0.0),
                "correct": true_cat == pred_cat,
            }
        )
        time.sleep(0.5)

    seen_categories = sorted(set(true_labels))
    metrics = compute_metrics(true_labels, pred_labels, seen_categories)
    accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)

    table = Table(title="Evaluation Results")
    table.add_column("Category", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Support", justify="right")

    for cat, m in sorted(metrics.items()):
        table.add_row(cat, f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}", str(m["support"]))

    console.print(table)
    console.print(f"\n[bold]Overall Accuracy: {accuracy:.2%}  ({sum(d['correct'] for d in details)}/{len(details)})[/bold]")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")

    with open(output_path, "w") as f:
        for item in details:
            f.write(json.dumps(item) + "\n")

    with open(metrics_path, "w") as f:
        json.dump({"accuracy": accuracy, "per_category": metrics}, f, indent=2)

    console.print(f"[green]Results saved to {output_path} and {metrics_path}[/green]")


if __name__ == "__main__":
    main()
