#!/usr/bin/env python3
"""
Command line interface for kngraph.
"""

import argparse
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
            )
        except UserFacingError as exc:
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
