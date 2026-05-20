from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_pipeline.interface.schemas import (
    GenerationOptions,
    RetrievalOptions,
    SearchFilters,
)
from graphrag_pipeline.interface.service import SearchGenerationService
from graphrag_pipeline.pipeline import GraphRAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphrag", description="GraphRAG prompt/search CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ask = subcommands.add_parser("ask", help="Search the graph/vector index and generate an answer")
    ask.add_argument("query")
    ask.add_argument("--db", default="graphrag.db", help="SQLite database path")
    ask.add_argument("--model", default=None, help="MiniGPT checkpoint path")
    ask.add_argument("--template", default="financial_rag")
    ask.add_argument("--ticker", action="append", default=[])
    ask.add_argument("--entity", action="append", default=[])
    ask.add_argument("--source", action="append", default=[])
    ask.add_argument("--published-after", default=None)
    ask.add_argument("--published-before", default=None)
    ask.add_argument("--min-sentiment", type=float, default=None)
    ask.add_argument("--max-sentiment", type=float, default=None)
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--graph-neighbors", type=int, default=None)
    ask.add_argument("--no-graph", action="store_true")
    ask.add_argument("--no-chunks", action="store_true")
    ask.add_argument("--no-sentiment", action="store_true")
    ask.add_argument("--max-tokens", type=int, default=None)
    ask.add_argument("--temperature", type=float, default=None)
    ask.add_argument("--generation-top-k", type=int, default=None)
    ask.add_argument("--debug", action="store_true")
    ask.add_argument("--json", action="store_true", help="Print full JSON response")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ask":
        return run_ask(args)
    raise ValueError(f"Unsupported command: {args.command}")


def run_ask(args: argparse.Namespace) -> int:
    pipeline = GraphRAGPipeline.local(
        database_path=Path(args.db),
        model_checkpoint_path=args.model,
    )
    service = SearchGenerationService(pipeline)
    response = service.search_and_generate(
        query=args.query,
        template=args.template,
        filters=SearchFilters.from_mapping(
            {
                "tickers": args.ticker,
                "entities": args.entity,
                "sources": args.source,
                "published_after": args.published_after,
                "published_before": args.published_before,
                "min_sentiment": args.min_sentiment,
                "max_sentiment": args.max_sentiment,
            }
        ),
        retrieval=RetrievalOptions(
            vector_top_k=args.top_k,
            graph_neighbors_per_entity=args.graph_neighbors,
            include_graph=not args.no_graph,
            include_chunks=not args.no_chunks,
            include_sentiment=not args.no_sentiment,
        ),
        generation=GenerationOptions(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.generation_top_k,
        ),
        debug=args.debug,
    )
    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
        return 0

    print(response.answer)
    if args.debug:
        print()
        print(json.dumps(response.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
