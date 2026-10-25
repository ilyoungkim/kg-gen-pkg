#!/usr/bin/env python3
"""
Command line interface for kngraph.
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
import importlib.util


def check_and_install_mcp_dependencies():
    """Check if MCP dependencies are installed, and install them if not."""
    # Check if fastmcp is available
    fastmcp_spec = importlib.util.find_spec("fastmcp")
    mcp_spec = importlib.util.find_spec("mcp")

    if fastmcp_spec is None or mcp_spec is None:
        print("MCP dependencies not found. Installing them now...")
        print("This is a one-time setup for MCP functionality.")
        print()

        try:
            # Try to install MCP dependencies
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "fastmcp>=2.10.6",
                    "mcp>=1.12.1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            print("✅ MCP dependencies installed successfully!")
            print()
            return True

        except subprocess.CalledProcessError as e:
            print("❌ Failed to install MCP dependencies:")
            print(f"Error: {e.stderr}")
            print()
            print("Please install manually:")
            print("  pip install fastmcp>=2.10.6 mcp>=1.12.1")
            print("or")
            print("  pip install 'kngraph[mcp]'")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during installation: {e}")
            print()
            print("Please install manually:")
            print("  pip install fastmcp>=2.10.6 mcp>=1.12.1")
            return False

    return True


def run_mcp():
    """Start the MCP server using fastmcp."""
    # Check and install MCP dependencies if needed
    if not check_and_install_mcp_dependencies():
        return 1

    # Get the path to the server.py file
    server_path = Path(__file__).parent.parent.parent / "mcp" / "server.py"

    if not server_path.exists():
        print(f"Error: MCP server file not found at {server_path}")
        return 1

    print("Starting kngraph MCP server...")
    print(f"Server path: {server_path}")
    print("Use Ctrl+C to stop the server")
    print()

    try:
        # Run fastmcp with the server script
        result = subprocess.run(["fastmcp", "run", str(server_path)], check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: fastmcp not found even after installation attempt.")
        print("Please try installing manually:")
        print("  pip install fastmcp>=2.10.6 mcp>=1.12.1")
        return 1
    except KeyboardInterrupt:
        print("\nMCP server stopped.")
        return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="kngraph",
        description="kngraph: Extract knowledge graphs using LLMs from any text",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # MCP subcommand
    mcp_parser = subparsers.add_parser("mcp", help="Start the MCP server")
    mcp_parser.add_argument(
        "--model", help="Model to use (e.g., openai/gpt-4o)", default=None
    )
    mcp_parser.add_argument(
        "--storage-path", help="Path for memory storage file", default=None
    )
    mcp_parser.add_argument(
        "--keep-memory",
        action="store_true",
        help="Keep existing memory instead of clearing it on startup",
    )

    # Analyze subcommand: crawl a website and build a knowledge graph + HTML
    analyze_parser = subparsers.add_parser(
        "analyze", help="Crawl a website and generate a knowledge graph (JSON/HTML)"
    )
    analyze_parser.add_argument("--url", "-u", default=None, help="Target URL to crawl (or TARGET_URL env)")
    analyze_parser.add_argument("--model", "-m", default=None, help="LLM model (or KG_MODEL env)")
    analyze_parser.add_argument("--qa-model", default=None, help="Q&A model (or KG_QA_MODEL env)")
    analyze_parser.add_argument(
        "--locally-ollama", dest="locally_ollama", action="store_true",
        help="Use direct local Ollama calls",
    )
    analyze_parser.add_argument("--renew", action="store_true", help="Ignore cached output and re-crawl")
    analyze_parser.add_argument("--output", "-o", default=None, help="Graph JSON output path")
    analyze_parser.add_argument("--html", default=None, help="Visualization HTML output path")
    analyze_parser.add_argument("--page", type=int, default=None, help="Max pages to collect")
    analyze_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logs")
    analyze_parser.add_argument("--config", default=None, help="config.json path (extraction guideline)")
    analyze_parser.add_argument("--no-browser", action="store_true", help="Do not open browser after generation")
    analyze_parser.add_argument(
        "--browser", action="store_true",
        help="Crawl with a real Chrome over CDP instead of plain HTTP",
    )
    analyze_parser.add_argument(
        "--browser-headless", dest="browser_headless", action="store_true",
        help="Run the crawling browser in headless mode",
    )
    analyze_parser.add_argument("--chrome-path", default=None, help="Chrome executable path")
    analyze_parser.add_argument("--cdp-url", default=None, help="Connect to an existing Chrome via CDP URL")
    analyze_parser.add_argument("--profile-dir", default=None, help="Browser profile directory (default: .browser-profile)")

    # Batch subcommand: analyze every URL listed in urls.yaml
    batch_parser = subparsers.add_parser(
        "batch", help="Analyze every URL in urls.yaml (batch run)"
    )
    batch_parser.add_argument("--urls", default="urls.yaml", help="URL case list (YAML)")
    batch_parser.add_argument("--output-root", default="outputs/batch", help="Batch output root directory")
    batch_parser.add_argument("--model", "-m", default=None, help="LLM model (or KG_MODEL env)")
    batch_parser.add_argument("--config", default=None, help="config.json path (extraction guideline)")
    batch_parser.add_argument("--page", type=int, default=None, help="Max pages to collect per URL")
    batch_parser.add_argument("--host", dest="only_host", default=None, help="Only run URLs of this host")
    batch_parser.add_argument(
        "--case", dest="case_ids", type=int, action="append", default=None,
        help="Only run this case_id (repeatable)",
    )
    batch_parser.add_argument("--renew", action="store_true", help="Ignore cached output and re-crawl")
    batch_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logs")
    batch_parser.add_argument("--dry-run", action="store_true", help="List planned work without running")
    batch_parser.add_argument("--no-report", action="store_true", help="Do not write the batch report")
    batch_parser.add_argument(
        "--merge-output", default=None,
        help="After the batch, merge all per-URL sources.json into one graph.html here",
    )
    batch_parser.add_argument(
        "--browser", action="store_true",
        help="Crawl with a real Chrome over CDP instead of plain HTTP",
    )
    batch_parser.add_argument(
        "--browser-headless", dest="browser_headless", action="store_true",
        help="Run the crawling browser in headless mode",
    )
    batch_parser.add_argument("--chrome-path", default=None, help="Chrome executable path")
    batch_parser.add_argument("--cdp-url", default=None, help="Connect to an existing Chrome via CDP URL")
    batch_parser.add_argument("--profile-dir", default=None, help="Browser profile directory (default: .browser-profile)")

    # Probe subcommand: answer-engine visibility probe for SSG7 membership
    probe_parser = subparsers.add_parser(
        "probe", help="Run the answer-engine visibility probe (brand / discovery Q&A suites)"
    )
    probe_parser.add_argument(
        "--sources", nargs="+", default=["outputs/www.ssg.com/sources.json"],
        help="sources.json file(s) produced by previous runs (repeatable)",
    )
    probe_parser.add_argument(
        "--suite", dest="suites", action="append", default=None,
        help="Suite to run: brand | discovery (repeatable, default both)",
    )
    probe_parser.add_argument("--brand-count", type=int, default=20, help="Brand question count")
    probe_parser.add_argument("--discovery-count", type=int, default=50, help="Discovery question count")
    probe_parser.add_argument(
        "--model", "-m", default=None,
        help="LLM model (default: openai/gpt-oss-120b via OpenRouter)",
    )
    probe_parser.add_argument("--output", default="outputs/probe", help="Report output directory")
    probe_parser.add_argument("--workers", type=int, default=4, help="Parallel question workers")
    probe_parser.add_argument("--max-chunks", type=int, default=40, help="Reference chunks to build")

    # Regenerate subcommand: rebuild outputs from already collected sources
    regenerate_parser = subparsers.add_parser(
        "regenerate", help="Rebuild graph/HTML from already collected sources.json (no re-crawl)"
    )
    regenerate_parser.add_argument(
        "--sources", nargs="+", default=None,
        help="sources.json file(s) to rebuild from (default: search --source-root)",
    )
    regenerate_parser.add_argument(
        "--source-root", default="outputs",
        help="Directory to search for sources.json files (default: outputs)",
    )
    regenerate_parser.add_argument(
        "--output", "-o", default="outputs/regenerated/graph.json",
        help="graph JSON output path",
    )
    regenerate_parser.add_argument("--html", default=None, help="HTML output path")
    regenerate_parser.add_argument("--model", "-m", default=None, help="LLM model (or KG_MODEL env)")
    regenerate_parser.add_argument("--config", default=None, help="config.json path (extraction guideline)")
    regenerate_parser.add_argument(
        "--reuse-graph", action="store_true",
        help="Skip the LLM pass and reuse the existing graph JSON (dashboard only)",
    )
    regenerate_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logs")
    regenerate_parser.add_argument("--no-browser", action="store_true", help="Do not open browser after generation")
    regenerate_parser.add_argument(
        "--probe-report", default=None,
        help="qa_probe_report.json to embed in the report tab (default: outputs/probe/qa_probe_report.json)",
    )

    # Configure subcommand: store API keys and defaults
    configure_parser = subparsers.add_parser(
        "configure", help="Store API keys and defaults in ~/.config/kngraph/.env"
    )
    configure_parser.add_argument("--api-key", default=None, help="OpenAI/KG API key (or KG_API_KEY)")
    configure_parser.add_argument("--openrouter-key", default=None, help="OpenRouter API key (or OPEN_ROUTER_API_KEY)")
    configure_parser.add_argument("--api-base", default=None, help="Custom API base URL (or KG_API_BASE)")
    configure_parser.add_argument("--model", default=None, help="Default analyze model (KG_MODEL)")
    configure_parser.add_argument("--qa-model", default=None, help="Default Q&A model (KG_QA_MODEL)")
    configure_parser.add_argument("--show", action="store_true", help="Show current stored settings")

    # QA server subcommand (fixed port 43870)
    qa_parser = subparsers.add_parser("qa-server", help="Run the Q&A runtime server (fixed port 43870)")
    qa_parser.add_argument("--host", default=None)
    qa_parser.add_argument("--port", type=int, default=None)
    qa_parser.add_argument("--model", default=None)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "mcp":
        # Set environment variables if provided
        if args.model:
            os.environ["KG_MODEL"] = args.model

        # Handle storage path - resolve relative paths relative to where command was called
        storage_path = args.storage_path or "./kg_memory.json"
        # Convert to absolute path relative to current working directory (where user called command)
        abs_storage_path = os.path.abspath(storage_path)
        os.environ["KG_STORAGE_PATH"] = abs_storage_path

        # Clear memory by default unless --keep-memory is specified
        if not args.keep_memory:
            os.environ["KG_CLEAR_MEMORY"] = "true"

        return run_mcp()

    if args.command == "analyze":
        from kngraph.config import load_config_files
        from kngraph.runner import UserFacingError, analyze

        load_config_files()
        if args.model:
            os.environ["KG_MODEL"] = args.model
        if args.qa_model:
            os.environ["KG_QA_MODEL"] = args.qa_model
        try:
            result = analyze(
                url=args.url,
                model=args.model,
                locally_ollama=args.locally_ollama,
                renew=args.renew,
                output=args.output,
                html=args.html,
                page=args.page,
                verbose=args.verbose,
                open_browser=not args.no_browser,
                config_path=args.config,
                browser=args.browser,
                browser_headless=args.browser_headless,
                chrome_path=args.chrome_path,
                cdp_url=args.cdp_url,
                profile_dir=args.profile_dir,
            )
        except UserFacingError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(result)
        return 0

    if args.command == "batch":
        from kngraph.batch import run_batch, write_batch_report
        from kngraph.config import load_config_files

        load_config_files()
        if args.model:
            os.environ["KG_MODEL"] = args.model
        try:
            summary = run_batch(
                urls_path=args.urls,
                output_root=args.output_root,
                model=args.model,
                page=args.page,
                renew=args.renew,
                verbose=args.verbose,
                config_path=args.config,
                only_host=args.only_host,
                only_case_ids=args.case_ids,
                dry_run=args.dry_run,
                browser=args.browser,
                browser_headless=args.browser_headless,
                chrome_path=args.chrome_path,
                cdp_url=args.cdp_url,
                profile_dir=args.profile_dir,
                merge_output=args.merge_output,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if not args.no_report:
            report_paths = write_batch_report(summary, args.output_root)
            summary["report"] = report_paths
        print(summary)
        return 0 if summary.get("failed", 0) == 0 else 1

    if args.command == "probe":
        from kngraph.qa_probe import run_probe, write_report

        source_documents: list[dict] = []
        for raw_path in args.sources:
            sources_path = Path(raw_path)
            if not sources_path.exists():
                print(f"소스 파일을 찾을 수 없습니다: {sources_path}", file=sys.stderr)
                return 1
            documents = json.loads(sources_path.read_text(encoding="utf-8"))
            if not isinstance(documents, list):
                print(f"sources.json 형식이 올바르지 않습니다: {sources_path}", file=sys.stderr)
                return 1
            source_documents.extend(documents)

        seen_urls: set[str] = set()
        deduplicated_documents: list[dict] = []
        for document in source_documents:
            url = str(document.get("url", ""))
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduplicated_documents.append(document)
        source_documents = deduplicated_documents

        suites = tuple(args.suites) if args.suites else ("brand", "discovery")
        try:
            report = run_probe(
                source_documents=source_documents,
                suites=suites,
                brand_count=args.brand_count,
                discovery_count=args.discovery_count,
                model=args.model,
                workers=args.workers,
                max_chunks=args.max_chunks,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        report_paths = write_report(report, args.output)
        report["report"] = report_paths
        print(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))
        print(report_paths)
        return 0

    if args.command == "regenerate":
        from kngraph.regenerate import collect_source_files, regenerate

        source_paths = list(args.sources) if args.sources else []
        if not source_paths:
            source_paths = [str(path) for path in collect_source_files(args.source_root)]
        if not source_paths:
            print(
                f"sources.json 을 찾지 못했습니다: {args.source_root}",
                file=sys.stderr,
            )
            return 1

        print({'source_files': len(source_paths)})
        try:
            result = regenerate(
                sources=source_paths,
                output=args.output,
                html=args.html,
                model=args.model,
                config_path=args.config,
                reuse_graph=args.reuse_graph,
                verbose=args.verbose,
                open_browser=not args.no_browser,
                probe_report=args.probe_report,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(result)
        return 0

    if args.command == "configure":
        from kngraph.config import (
            get_user_config_path,
            mask_api_key,
            read_user_config,
            write_user_config,
        )

        if args.show:
            current = read_user_config()
            print(f"config: {get_user_config_path()}")
            print(f"KG_API_KEY={mask_api_key(current.get('KG_API_KEY'))}")
            print(f"OPEN_ROUTER_API_KEY={mask_api_key(current.get('OPEN_ROUTER_API_KEY'))}")
            print(f"KG_API_BASE={current.get('KG_API_BASE', '(미설정)')}")
            print(f"KG_MODEL={current.get('KG_MODEL', '(미설정)')}")
            print(f"KG_QA_MODEL={current.get('KG_QA_MODEL', '(미설정)')}")
            return 0

        values: dict[str, str] = {}
        api_key = args.api_key or os.getenv("KG_API_KEY") or ""
        if not api_key:
            try:
                api_key = input("OpenAI/KG API key (KG_API_KEY, empty to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 1
        if api_key:
            values["KG_API_KEY"] = api_key
        openrouter_key = args.openrouter_key or os.getenv("OPEN_ROUTER_API_KEY") or ""
        if openrouter_key:
            values["OPEN_ROUTER_API_KEY"] = openrouter_key
        if args.api_base:
            values["KG_API_BASE"] = args.api_base
        if args.model:
            values["KG_MODEL"] = args.model
        if args.qa_model:
            values["KG_QA_MODEL"] = args.qa_model
        if not values:
            print("저장할 값이 없습니다. --api-key / --openrouter-key / --api-base / --model / --qa-model 중 하나를 지정하세요.")
            return 1
        path = write_user_config(values)
        print(f"저장됨: {path} (권한 600)")
        return 0

    if args.command == "qa-server":
        from kngraph.config import DEFAULT_QA_SERVER_HOST, DEFAULT_QA_SERVER_PORT, load_config_files
        from kngraph.qa import main as qa_main

        load_config_files()
        return qa_main([
            "--host", args.host or DEFAULT_QA_SERVER_HOST,
            "--port", str(args.port or DEFAULT_QA_SERVER_PORT),
            *(["--model", args.model] if args.model else []),
        ]) or 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
