"""Command-line interface for Jig."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from jig import __version__
from jig.agent import SchemaAgent
from jig.client import LMStudioClient
from jig.constants import PAIRINGS_DIR
from jig.creator import SchemaCreator
from jig.factory import create_client
from jig.ollama_client import OllamaClient
from jig.repository import PairingRepository
from jig.ui.console import ConsoleUI, print_json_colored
from jig.ui.console import confirm, make_confirm


def cmd_create(args: argparse.Namespace, client) -> None:
    """Handle create command."""
    ConsoleUI.dim(
        f"Purpose: {args.purpose[:60]}{'...' if len(args.purpose) > 60 else ''}"
    )
    ConsoleUI.dim(f"Target: pairings/{args.name}/")
    creator = SchemaCreator(client, confirm_overwrite=make_confirm(skip_all=args.yes))
    result = creator.create(args.purpose, args.name, force=args.force)
    if result:
        repo = PairingRepository()
        target = repo.path(args.name)
        ConsoleUI.success("Created pairing:")
        ConsoleUI.dim(f"   Directory: {target}")
        ConsoleUI.dim(f"   Schema: schema.json | Prompt: prompt.txt | Meta: meta.json")
        ConsoleUI.dim(f"   Description: {result.get('description', 'N/A')}")
        _display_creation_preview(args.name, result)
    elif not args.yes:
        ConsoleUI.info("Skipped (already exists).", dim=True)


def cmd_run(args: argparse.Namespace, client) -> None:
    """Handle run command."""
    from jig.utils import resolve_pairing_path

    if not args.input and not (args.images and len(args.images) > 0):
        ConsoleUI.error("Provide --input text and/or --image path(s)")
        return

    ConsoleUI.dim("Running Agent")
    agent = SchemaAgent(client)
    schema_ref = args.schema
    prompt_ref = args.prompt or args.schema
    schema_path = resolve_pairing_path(schema_ref, "schema")
    prompt_path = resolve_pairing_path(prompt_ref, "prompt")
    ConsoleUI.dim(f"Schema: {schema_path}")
    ConsoleUI.dim(f"Prompt: {prompt_path}")
    ConsoleUI.dim(
        f"Input: {args.input[:50]}{'...' if len(args.input) > 50 else '(from images)' if args.images else ''}"
    )
    if args.images:
        ConsoleUI.dim(f"Images: {len(args.images)} file(s)")
    ConsoleUI.dim(f"Model: {client.ensure_model()}")
    result = agent.run(
        args.input or "(no text)",
        schema_ref,
        prompt_ref=prompt_ref,
        output_path=args.output,
        temperature=args.temperature,
        image_paths=args.images,
    )
    print()
    ConsoleUI.success("Result:")
    print_json_colored(result)
    if args.output:
        ConsoleUI.dim(f"Saved to: {Path(args.output).absolute()}")


def cmd_list(_args: argparse.Namespace, _client) -> None:
    """Handle list command."""
    repo = PairingRepository()
    if not PAIRINGS_DIR.exists():
        ConsoleUI.info(
            "No pairings directory found. Create one with 'create' command.", dim=True
        )
        return
    pairings = list(repo.list_all())
    if not pairings:
        ConsoleUI.info(f"No pairings found in {PAIRINGS_DIR}/", dim=True)
        return
    print(f"\n{'-' * 50}")
    for p in pairings:
        status = "[OK]" if p["schema_exists"] and p["prompt_exists"] else "[--]"
        print(f"\n{status} {p['name']}")
        ConsoleUI.dim(
            f"   Schema: {'schema.json' if p['schema_exists'] else 'missing'}"
        )
        ConsoleUI.dim(f"   Prompt: {'prompt.txt' if p['prompt_exists'] else 'missing'}")
        if p["description"]:
            ConsoleUI.dim(f"   Info: {p['description']}...")
        ConsoleUI.dim(f"   Run: jig run -s {p['name']} -i \"...\"")


def cmd_models(args: argparse.Namespace, client) -> None:
    """List and optionally set available models for the connected backend."""
    models = []
    try:
        if getattr(args, "set_model", None):
            try:
                _set_active_model(client, args.set_model)
            except Exception as exc:
                ConsoleUI.error(str(exc))
                return
        models = client.list_models()
    except Exception as e:
        ConsoleUI.error(str(e))
        return
    if not models:
        ConsoleUI.info(
            "No models found. Load/pull a model in LM Studio or Ollama.", dim=True
        )
        return
    current_model = client.model
    print(f"\n{'-' * 50}")
    ConsoleUI.label(f"Available models ({len(models)})")
    display = []
    for m in models:
        prefix = "→ " if m == current_model else "  "
        display.append(f"{prefix}{m}")
    ConsoleUI.block("\n".join(display))
    ConsoleUI.dim(
        "Use 'model <name>' (interactive) or `jig models --set <name>` to switch."
    )


def cmd_show(args: argparse.Namespace, _client) -> None:
    """Handle show command."""
    repo = PairingRepository()
    dir_path = repo.path(args.name)
    if not dir_path.exists():
        ConsoleUI.error(f"Pairing '{args.name}' not found")
        return
    schema_path = dir_path / "schema.json"
    prompt_path = dir_path / "prompt.txt"
    print(f"\n{'-' * 50}")
    if prompt_path.exists():
        content = prompt_path.read_text()
        print(f"\nSystem Prompt:\n{content[:500]}{'...' if len(content) > 500 else ''}")
    if schema_path.exists():
        import json

        try:
            schema = json.loads(schema_path.read_text())
            print("\nSchema Structure:")
            print_json_colored(schema)
        except json.JSONDecodeError:
            ConsoleUI.error("Invalid JSON in schema file")


def _display_creation_preview(name: str, result: Dict[str, Any]) -> None:
    """Pretty-print the prompt and schema for a freshly created pairing."""
    if not result:
        return
    ConsoleUI.heading(f"Preview · {name}")
    desc = (result.get("description") or "").strip()
    if desc:
        ConsoleUI.dim(desc)
    ConsoleUI.label("System Prompt")
    ConsoleUI.block(result.get("system_prompt", ""))
    print()
    ConsoleUI.label("JSON Schema")
    schema = result.get("response_schema")
    if isinstance(schema, dict):
        print_json_colored(schema)
    else:
        ConsoleUI.info("No JSON schema returned.", dim=True)
    print()


def _set_active_model(client, desired: str, announce: bool = True) -> str:
    """Set and validate the active model on the connected client."""
    desired = (desired or "").strip()
    if not desired:
        raise ValueError("Model name cannot be empty.")
    client.model = desired
    resolved = client.ensure_model()
    if announce:
        ConsoleUI.success(f"Active model set to '{resolved}'")
    return resolved


def interactive_loop(client) -> None:
    """Interactive REPL mode."""
    creator = SchemaCreator(
        client, confirm_overwrite=lambda m: confirm(m, default=False)
    )
    agent = SchemaAgent(client)
    print()
    ConsoleUI.label("Commands")
    command_entries = [
        ("create", "Generate schema + prompt pairing from a description"),
        ("run", "Execute a pairing with text/file/image input"),
        ("list", "Show saved pairings and file status"),
        ("show", "Print schema + prompt for one pairing"),
        ("models", "List models (optionally set default)"),
        ("model", "Shortcut to set the active model"),
        ("help", "Show detailed help and examples"),
        ("quit", "Exit interactive mode"),
    ]
    ConsoleUI.command_list(command_entries)
    ConsoleUI.label("Tip")
    ConsoleUI.dim(
        "Type a description by itself to auto-create; use `run <name> ./input.txt` for files."
    )
    while True:
        try:
            cmd = input("\n> ").strip()
            if not cmd:
                continue
            parts = cmd.split(maxsplit=1)
            command = parts[0].lower()
            rest = parts[1] if len(parts) > 1 else ""

            if command in ("quit", "exit", "q"):
                ConsoleUI.info("Goodbye!")
                break
            if command == "list":
                cmd_list(argparse.Namespace(), client)
            elif command == "models":
                cmd_models(argparse.Namespace(set_model=rest or None), client)
            elif command in ("model", "use", "setmodel"):
                if not rest:
                    ConsoleUI.info("Usage: model <name>")
                    continue
                try:
                    _set_active_model(client, rest)
                except Exception as e:
                    ConsoleUI.error(str(e))
            elif command == "show":
                if rest:
                    cmd_show(argparse.Namespace(name=rest), client)
                else:
                    ConsoleUI.info("Usage: show <pairing_name>")
            elif command == "create":
                if not rest:
                    ConsoleUI.info("Usage: create <description>")
                    continue
                name = input("Name for this pairing [auto]: ").strip()
                if not name:
                    import re

                    name = "_".join(rest.split()[:3]).lower()
                    name = re.sub(r"[^\w]", "", name) or "untitled"
                result = creator.create(rest, name)
                if result:
                    ConsoleUI.success(f"Created pairing '{name}'")
                    _display_creation_preview(name, result)
                else:
                    ConsoleUI.info("Skipped (already exists).", dim=True)
            elif command == "run":
                subparts = rest.split(maxsplit=1)
                if len(subparts) < 2:
                    ConsoleUI.info("Usage: run <pairing_name> <input or file.txt>")
                    continue
                name, user_input = subparts
                try:
                    result = agent.run(user_input, name, name)
                    print_json_colored(result)
                except Exception as e:
                    ConsoleUI.error(str(e))
            elif command == "help":
                ConsoleUI.heading("Jig CLI Help")
                ConsoleUI.label("Commands")
                ConsoleUI.command_list(command_entries)
                ConsoleUI.label("Examples")
                ConsoleUI.block(
                    "\n".join(
                        [
                            "create Extract lab results into JSON",
                            'run lab "HB: 13.5, WBC: 6.3"',
                            "run lab ./lab_results.txt - streams file input",
                            "model zai-org/GLM-4.6",
                            "models --set llama3.1:instruct",
                        ]
                    )
                )
                ConsoleUI.label("Shortcuts")
                ConsoleUI.block(
                    "\n".join(
                        [
                            "Enter plain text with no command to auto-create a pairing",
                            "Use `run <pairing>` with zero input to process images only",
                            "Switch models anytime with `model <name>` (LM Studio or Ollama)",
                        ]
                    )
                )
            else:
                name = input("Name for this pairing [auto]: ").strip()
                if not name:
                    import re

                    name = "_".join(cmd.split()[:3]).lower()
                    name = re.sub(r"[^\w]", "", name) or "untitled"
                result = creator.create(cmd, name)
                if result:
                    ConsoleUI.success(f"Created pairing '{name}'")
                    _display_creation_preview(name, result)
                else:
                    ConsoleUI.info("Skipped (already exists).", dim=True)
        except KeyboardInterrupt:
            ConsoleUI.info("Interrupted")
            break
        except Exception as e:
            ConsoleUI.error(str(e))


def main() -> int:
    """Main entry point."""
    ConsoleUI.banner(__version__)

    parser = argparse.ArgumentParser(
        description="Jig - Structured output for LM Studio & Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jig                                           # Interactive mode
  jig models                                    # List available models
  jig --backend ollama run -s meeting -i "..."  # Use Ollama
  jig create "Extract meeting details" -n meeting
  jig run -s meeting -i "Attendees: John, Jane"
  jig models --set zai-org/GLM-4.6
  jig --gradio
  jig list
        """,
    )

    conn = parser.add_argument_group("Connection")
    conn.add_argument(
        "--backend",
        choices=("lmstudio", "ollama", "auto"),
        default="auto",
        help="Backend: lmstudio, ollama, or auto (try LM Studio then Ollama)",
    )
    conn.add_argument("--host", default="localhost", help="Server host")
    conn.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 1234 LM Studio, 11434 Ollama)",
    )
    conn.add_argument("--model", default=None, help="Model name/ID")
    conn.add_argument(
        "--auto-probe",
        action="store_true",
        help="Auto-scan ports if LM Studio connection fails",
    )
    conn.add_argument("--timeout", type=float, default=3.0, help="Connection timeout")

    parser.add_argument(
        "-I", "--interactive", action="store_true", help="Interactive REPL"
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmations")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio web UI")

    sub = parser.add_subparsers(dest="command")

    create_p = sub.add_parser("create", help="Create schema+prompt pairing")
    create_p.add_argument("purpose", help="Description of AI task")
    create_p.add_argument("-n", "--name", default="generated", help="Pairing name")
    create_p.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing"
    )

    run_p = sub.add_parser("run", help="Run agent with pairing")
    run_p.add_argument(
        "-s", "--schema", required=True, help="Schema file or pairing name"
    )
    run_p.add_argument(
        "-p", "--prompt", help="Prompt file or pairing name (default: same as schema)"
    )
    run_p.add_argument(
        "-i", "--input", default="", help="Input text or path to .txt file"
    )
    run_p.add_argument(
        "--image",
        action="append",
        dest="images",
        default=None,
        metavar="PATH",
        help="Image file path(s) for vision models (can repeat)",
    )
    run_p.add_argument("-o", "--output", help="Output JSON file")
    run_p.add_argument(
        "-t", "--temperature", type=float, default=0.2, help="Temperature"
    )

    sub.add_parser("list", help="List pairings")
    models_p = sub.add_parser(
        "models", help="List or set available models for the connected backend"
    )
    models_p.add_argument(
        "--set", dest="set_model", help="Set active model for this session"
    )
    show_p = sub.add_parser("show", help="Display pairing contents")
    show_p.add_argument("name", help="Pairing name")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        args.interactive = True

    # Commands that don't need backend connection
    offline_commands = ("list", "show")
    needs_client = (
        args.command in ("create", "run", "models") or args.interactive or args.gradio
    )

    if needs_client:
        try:
            client = create_client(
                host=args.host,
                port=args.port,
                model=args.model,
                timeout=args.timeout,
                auto_probe=args.auto_probe,
                backend=args.backend,
            )
            backend_label = (
                "LM Studio" if isinstance(client, LMStudioClient) else "Ollama"
            )
            parsed = urlparse(client.base_url)
            port = parsed.port
            port_hint = ""
            if isinstance(client, LMStudioClient) and port == 1234:
                port_hint = " (default LM Studio port 1234)"
            elif isinstance(client, OllamaClient) and port in (11434, 11435):
                port_hint = f" (default Ollama port {port})"
            ConsoleUI.success(
                f"Connected to {backend_label} at {client.base_url}{port_hint}"
            )
            try:
                active_model = client.ensure_model()
            except Exception:
                active_model = client.model or "(no model selected)"
            ConsoleUI.dim(f"Active model: {active_model}")
        except RuntimeError as e:
            ConsoleUI.error(str(e))
            return 1
    else:
        client = None

    try:
        if args.gradio:
            from jig.ui.gradio_app import launch_gradio

            creator = SchemaCreator(client)
            agent = SchemaAgent(client)
            ConsoleUI.info("Launching Gradio at http://localhost:7860", dim=True)
            launch_gradio(creator, agent)
            return 0

        if args.interactive:
            interactive_loop(client)
            return 0

        if args.command == "create":
            cmd_create(args, client)
        elif args.command == "run":
            cmd_run(args, client)
        elif args.command == "list":
            cmd_list(args, client)
        elif args.command == "models":
            cmd_models(args, client)
        elif args.command == "show":
            cmd_show(args, client)
        else:
            parser.print_help()
        return 0
    except Exception as e:
        ConsoleUI.error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
