"""CLI: ask the aviation assistant a question."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.index import load_index  # noqa: E402
from src.retrieve import Retriever  # noqa: E402
from src.agents.planner import run  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Aviation question to answer"),
    style: str = typer.Option("cot", help="Prompt style: cot or zero_shot"),
    mode: str = typer.Option("hybrid", help="Retrieval mode: semantic, lexical, hybrid"),
    show_evidence: bool = typer.Option(False, help="Print retrieved chunks"),
) -> None:
    bundle = load_index()
    retriever = Retriever.from_bundle(bundle)
    result = run(question, retriever, prompt_style=style, retrieval_mode=mode)

    if result.out_of_scope_message:
        console.print(Panel(result.out_of_scope_message, title="Out of scope", style="yellow"))
        return

    trace = result.trace
    console.print(Panel(f"[bold]{trace.query}[/bold]", title="Question"))

    if show_evidence:
        table = Table(title="Retrieved evidence")
        table.add_column("rank")
        table.add_column("case_id")
        table.add_column("field")
        table.add_column("score")
        table.add_column("text", overflow="fold")
        for i, r in enumerate(trace.retrieved, 1):
            table.add_row(
                str(i),
                r.chunk.case_id,
                r.chunk.field,
                f"{r.score:.4f}",
                r.chunk.text[:160],
            )
        console.print(table)

    if trace.answer:
        console.print(
            Panel(
                f"{trace.answer.answer}\n\n[dim]Cited:[/dim] "
                f"{', '.join(trace.answer.cited_cases) or '(none)'}\n"
                f"[dim]Why these:[/dim] {trace.answer.rationale}\n"
                f"[dim]Confidence:[/dim] {trace.answer.confidence}",
                title="Answer",
            )
        )
    if trace.verdict:
        console.print(
            Panel(
                f"verdict: {trace.verdict.verdict}\n"
                f"grounding_ok: {trace.verdict.grounding_ok}  "
                f"consistency_ok: {trace.verdict.consistency_ok}\n"
                f"notes: {trace.verdict.notes}",
                title="Validator",
                style="cyan",
            )
        )


if __name__ == "__main__":
    app()
