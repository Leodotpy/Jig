"""Gradio web interface for Jig."""

import json
import re
from pathlib import Path
from typing import List, Optional

from jig.agent import SchemaAgent
from jig.constants import PAIRINGS_DIR
from jig.creator import SchemaCreator
from jig.repository import PairingRepository


def get_pairings_list() -> List[str]:
    """List available pairing names. Excludes backup directories (*_backup_*)."""
    if not PAIRINGS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in PAIRINGS_DIR.iterdir()
        if d.is_dir() and "_backup_" not in d.name
    )


def _sanitize_name(text: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", text).strip().lower()
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe or "untitled"


def create_gradio_create_handler(creator: SchemaCreator):
    """Create handler for Gradio create button."""

    def handler(purpose: str, name: str, force: bool) -> tuple[str, str, str]:
        if not purpose:
            return "Error: Purpose required", "", ""
        if not name:
            name = _sanitize_name(purpose)
        try:
            result = creator.create(purpose, name, force=force)
            if not result:
                return "Skipped (already exists)", "", ""
            schema_preview = json.dumps(result.get("response_schema", {}), indent=2)
            prompt_preview = result.get("system_prompt", "")
            return f"Created: {name}", schema_preview, prompt_preview
        except Exception as e:
            return f"Error: {str(e)}", "", ""

    return handler


def create_gradio_run_handler(agent: SchemaAgent):
    """Create handler for Gradio run button. When stream=True, yields (text, status) as output generates."""

    def handler(
        pairing_name: str,
        input_text: str,
        input_file: Optional[object],
        input_images: Optional[List[object]],
        temperature: float,
        output_file: Optional[str],
        stream: bool,
    ):
        if not pairing_name:
            yield "", "Error: Select a pairing"
            return
        if input_file is not None and hasattr(input_file, "name"):
            try:
                input_data = Path(input_file.name).read_text(encoding="utf-8")
            except Exception as e:
                yield "", f"Error reading file: {e}"
                return
        else:
            input_data = input_text or ""
        image_paths = None
        if input_images:
            paths = []
            for f in input_images if isinstance(input_images, list) else [input_images]:
                if f is not None and hasattr(f, "name"):
                    paths.append(f.name)
            if paths:
                image_paths = paths
        if not input_data.strip() and not image_paths:
            yield "", "Error: Provide input text, a .txt file, or image(s)"
            return
        if not stream:
            try:
                result = agent.run(
                    input_data.strip() or "(no text)",
                    pairing_name,
                    pairing_name,
                    output_path=output_file or None,
                    temperature=temperature,
                    image_paths=image_paths,
                )
                yield json.dumps(
                    result, indent=2
                ), f"Success using pairing: {pairing_name}"
            except Exception as e:
                yield "", f"Error: {str(e)}"
            return
        try:
            for content, result in agent.run_stream(
                input_data.strip() or "(no text)",
                pairing_name,
                pairing_name,
                output_path=output_file or None,
                temperature=temperature,
                image_paths=image_paths,
            ):
                if result is not None:
                    yield json.dumps(
                        result, indent=2
                    ), f"Success using pairing: {pairing_name}"
                else:
                    yield content or "", "Streaming..."
        except Exception as e:
            yield "", f"Error: {str(e)}"

    return handler


def get_models_list(client) -> List[str]:
    """Return list of available model names/IDs for the connected backend."""
    try:
        return client.list_models()
    except Exception:
        return []


def launch_gradio(creator: SchemaCreator, agent: SchemaAgent) -> None:
    """Launch Gradio web interface."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio not installed. Install with: pip install gradio")

    client = agent.client
    model_list = get_models_list(client)
    default_model = client.model or (model_list[0] if model_list else None)
    if default_model and not client.model:
        client.model = default_model

    repo = getattr(creator, "repo", None) or PairingRepository()
    initial_pairings = get_pairings_list()
    default_pairing = initial_pairings[0] if initial_pairings else None

    def _pairing_payload(name: Optional[str]) -> dict:
        base = {
            "schema": "",
            "prompt": "",
            "description": "",
            "status": "Select a pairing to load",
            "meta_summary": "Pairings live in the pairings/ directory.",
        }
        if not name:
            return base
        try:
            data = repo.load(name)
        except Exception as exc:
            base["status"] = f"Error loading '{name}': {exc}"
            base["meta_summary"] = base["status"]
            return base

        schema_text = json.dumps(data.get("schema", {}), indent=2, ensure_ascii=False)
        prompt_text = data.get("prompt", "")
        meta = data.get("meta") or {}
        description = meta.get("description", "").strip()
        created = meta.get("created")
        model = meta.get("model")
        meta_bits = []
        if description:
            meta_bits.append(description)
        if created:
            meta_bits.append(f"Created: {created}")
        if model:
            meta_bits.append(f"Model: {model}")
        meta_summary = " | ".join(meta_bits) if meta_bits else f"Loaded {name}"
        return {
            "schema": schema_text,
            "prompt": prompt_text,
            "description": description,
            "status": f"Loaded pairing '{name}'",
            "meta_summary": meta_summary,
        }

    def load_inference_preview(name: Optional[str]):
        payload = _pairing_payload(name)
        return payload["schema"], payload["prompt"], payload["meta_summary"]

    def load_editor_fields(name: Optional[str]):
        payload = _pairing_payload(name)
        return (
            payload["schema"],
            payload["prompt"],
            payload["description"],
            payload["status"],
        )

    def save_editor_fields(
        name: Optional[str], schema_text: str, prompt_text: str, description_text: str
    ):
        if not name:
            return (
                schema_text,
                prompt_text,
                description_text,
                "Select or type a pairing name before saving.",
            )
        if not prompt_text.strip():
            return schema_text, prompt_text, description_text, "Prompt cannot be empty."
        if not schema_text.strip():
            return (
                schema_text,
                prompt_text,
                description_text,
                "Schema JSON cannot be empty.",
            )
        try:
            parsed_schema = json.loads(schema_text)
        except json.JSONDecodeError as exc:
            return (
                schema_text,
                prompt_text,
                description_text,
                f"Schema JSON error: {exc}",
            )
        try:
            existing = repo.load(name)
            meta = existing.get("meta") or {}
        except Exception:
            meta = {"name": name}
        desc_clean = description_text.strip()
        if desc_clean:
            meta["description"] = desc_clean
        elif "description" in meta:
            meta.pop("description")
        meta.setdefault("name", name)
        repo.save(name, parsed_schema, prompt_text, meta)
        formatted = json.dumps(parsed_schema, indent=2, ensure_ascii=False)
        payload = _pairing_payload(name)
        payload["schema"] = formatted
        return (
            payload["schema"],
            prompt_text,
            payload["description"],
            f"Saved pairing '{name}'.",
        )

    def format_schema_json(schema_text: str):
        if not schema_text.strip():
            return schema_text, "Add JSON first to format."
        try:
            parsed_schema = json.loads(schema_text)
        except json.JSONDecodeError as exc:
            return schema_text, f"Schema JSON error: {exc}"
        pretty = json.dumps(parsed_schema, indent=2, ensure_ascii=False)
        return pretty, "Schema formatted."

    def refresh_pairing_views(current_run: Optional[str], current_edit: Optional[str]):
        choices = get_pairings_list()
        new_run = (
            current_run if current_run in choices else (choices[0] if choices else None)
        )
        new_edit = current_edit if current_edit in choices else new_run
        run_payload = _pairing_payload(new_run)
        edit_payload = _pairing_payload(new_edit)
        return (
            gr.update(choices=choices, value=new_run),
            run_payload["schema"],
            run_payload["prompt"],
            run_payload["meta_summary"],
            gr.update(choices=choices, value=new_edit),
            edit_payload["schema"],
            edit_payload["prompt"],
            edit_payload["description"],
            edit_payload["status"],
        )

    pairing_defaults = _pairing_payload(default_pairing)

    def on_model_select(sel: Optional[str]) -> str:
        if sel:
            client.model = sel
            return f"Using model: **{sel}**"
        return "Using default model"

    def refresh_models_dropdown():
        models = get_models_list(client)
        val = client.model or (models[0] if models else None)
        return gr.update(choices=models, value=val)

    gradio_create = create_gradio_create_handler(creator)
    gradio_run = create_gradio_run_handler(agent)

    # Lavender/violet accent theme
    try:
        lavender_theme = gr.themes.Default(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
        )
    except Exception:
        lavender_theme = gr.themes.Soft()

    css = """
    .input-text { min-height: 100px; }
    .json-output { min-height: 200px; }
    """
    with gr.Blocks(title="Jig") as demo:
        gr.Markdown("# Jig")
        gr.Markdown(
            "Create structured output schemas and run inference with local models in LM Studio/Ollama "
            "or multimodal checkpoints like **zai-org/GLM-4.6**."
        )
        gr.Markdown(
            "If this tool helps you, please consider ⭐ [starring Jig on GitHub](https://github.com/Leodotpy/jig). "
            "Your support keeps local LLM tooling thriving!"
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=model_list,
                value=default_model,
                allow_custom_value=False,
            )
            refresh_models_btn = gr.Button("Refresh models", size="sm")
            model_status = gr.Markdown(value=on_model_select(default_model))

        model_dropdown.change(
            fn=on_model_select,
            inputs=[model_dropdown],
            outputs=[model_status],
        )
        refresh_models_btn.click(
            fn=refresh_models_dropdown,
            outputs=[model_dropdown],
        ).then(
            fn=on_model_select,
            inputs=[model_dropdown],
            outputs=[model_status],
        )

        with gr.Tab("Creator"):
            with gr.Row():
                with gr.Column(scale=2):
                    purpose = gr.Textbox(
                        label="Purpose",
                        placeholder="Describe what the AI should do (e.g., Extract invoice details)",
                        lines=3,
                    )
                    name = gr.Textbox(
                        label="Pairing Name",
                        placeholder="Auto-generated if empty (e.g., invoice_extractor)",
                    )
                    force = gr.Checkbox(label="Force Overwrite", value=False)
                    create_btn = gr.Button(
                        "Create Pairing", variant="primary", size="lg"
                    )

                with gr.Column(scale=3):
                    status = gr.Textbox(label="Status", interactive=False)
                    with gr.Row():
                        schema_preview = gr.Code(
                            label="Schema Preview",
                            language="json",
                            elem_classes=["json-output"],
                        )
                        prompt_preview = gr.Textbox(
                            label="Prompt Preview",
                            lines=10,
                            interactive=False,
                        )

            create_btn.click(
                gradio_create,
                inputs=[purpose, name, force],
                outputs=[status, schema_preview, prompt_preview],
            )

        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column():
                    pairing_dropdown = gr.Dropdown(
                        label="Select Pairing",
                        choices=initial_pairings,
                        value=default_pairing,
                        allow_custom_value=True,
                    )
                    refresh_btn = gr.Button("Refresh List", size="sm")

                    with gr.Tab("Text Input"):
                        input_text = gr.Textbox(
                            label="Input Text",
                            lines=5,
                            placeholder="Enter text to process... (or use Image tab for vision)",
                            elem_classes=["input-text"],
                        )
                    with gr.Tab("File Input"):
                        input_file = gr.File(
                            label="Upload Input File (.txt)",
                            file_types=[".txt"],
                        )
                    with gr.Tab("Image Input"):
                        input_images = gr.File(
                            label="Upload Image(s) for vision models",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        image_preview = gr.Gallery(
                            label="Preview",
                            columns=3,
                        )

                        def update_image_preview(files):
                            if not files:
                                return []
                            items = files if isinstance(files, list) else [files]
                            return [
                                f.name
                                for f in items
                                if f is not None and hasattr(f, "name")
                            ]

                        input_images.change(
                            fn=update_image_preview,
                            inputs=[input_images],
                            outputs=[image_preview],
                        )
                        gr.Markdown(
                            "*Run multimodal checkpoints like `zai-org/GLM-4.6` (text + vision) or other "
                            "vision-ready models (llava, deepseek-vl, moondream) inside LM Studio or Ollama.*"
                        )

                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                    )
                    output_file = gr.Textbox(
                        label="Save Output To (optional)",
                        placeholder="path/to/output.json",
                    )
                    stream_output = gr.Checkbox(
                        label="Stream output (show text as it generates)",
                        value=True,
                    )
                    run_btn = gr.Button("Run Inference", variant="primary", size="lg")

                with gr.Column():
                    gr.Markdown("### Pairing Preview")
                    pairing_preview_meta = gr.Markdown(
                        value=pairing_defaults["meta_summary"]
                    )
                    with gr.Row():
                        pairing_schema_preview = gr.Code(
                            label="Schema",
                            language="json",
                            value=pairing_defaults["schema"],
                            interactive=False,
                            elem_classes=["json-output"],
                        )
                        pairing_prompt_preview = gr.Textbox(
                            label="Prompt",
                            value=pairing_defaults["prompt"],
                            lines=15,
                            interactive=False,
                        )
                    gr.Markdown("### Result")
                    result_json = gr.Code(
                        label="Result (JSON)",
                        language="json",
                        elem_classes=["json-output"],
                    )
                    run_status = gr.Textbox(label="Status", interactive=False)

            pairing_dropdown.change(
                fn=load_inference_preview,
                inputs=[pairing_dropdown],
                outputs=[
                    pairing_schema_preview,
                    pairing_prompt_preview,
                    pairing_preview_meta,
                ],
            )

            run_btn.click(
                gradio_run,
                inputs=[
                    pairing_dropdown,
                    input_text,
                    input_file,
                    input_images,
                    temperature,
                    output_file,
                    stream_output,
                ],
                outputs=[result_json, run_status],
            )

        with gr.Tab("Editor"):
            gr.Markdown(
                "Load any pairing, tweak the JSON Schema or system prompt, then save back to `pairings/`."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    editor_pairing_dropdown = gr.Dropdown(
                        label="Pairing",
                        choices=initial_pairings,
                        value=default_pairing,
                        allow_custom_value=True,
                        info="Type a new name to start from a blank template.",
                    )
                    editor_description = gr.Textbox(
                        label="Description",
                        value=pairing_defaults["description"],
                        placeholder="Short summary shown in meta.json",
                    )
                    editor_status = gr.Textbox(
                        label="Status",
                        value=pairing_defaults["status"],
                        interactive=False,
                    )
                    format_schema_btn = gr.Button("Format Schema JSON", size="sm")
                    save_pairing_btn = gr.Button("Save Pairing", variant="primary")
                with gr.Column(scale=2):
                    editor_schema = gr.Code(
                        label="Schema JSON",
                        language="json",
                        value=pairing_defaults["schema"],
                        interactive=True,
                        elem_classes=["json-output"],
                    )
                with gr.Column(scale=2):
                    editor_prompt = gr.Textbox(
                        label="System Prompt",
                        value=pairing_defaults["prompt"],
                        lines=20,
                    )

            editor_pairing_dropdown.change(
                fn=load_editor_fields,
                inputs=[editor_pairing_dropdown],
                outputs=[
                    editor_schema,
                    editor_prompt,
                    editor_description,
                    editor_status,
                ],
            )
            format_schema_btn.click(
                fn=format_schema_json,
                inputs=[editor_schema],
                outputs=[editor_schema, editor_status],
            )
            save_pairing_btn.click(
                fn=save_editor_fields,
                inputs=[
                    editor_pairing_dropdown,
                    editor_schema,
                    editor_prompt,
                    editor_description,
                ],
                outputs=[
                    editor_schema,
                    editor_prompt,
                    editor_description,
                    editor_status,
                ],
            )

        refresh_btn.click(
            fn=refresh_pairing_views,
            inputs=[pairing_dropdown, editor_pairing_dropdown],
            outputs=[
                pairing_dropdown,
                pairing_schema_preview,
                pairing_prompt_preview,
                pairing_preview_meta,
                editor_pairing_dropdown,
                editor_schema,
                editor_prompt,
                editor_description,
                editor_status,
            ],
        )

        with gr.Tab("Help"):
            gr.Markdown(
                """
                ## Usage Guide

                ### Model selection
                At the top, choose which **Model** to use for Create and Run. Click **Refresh models** to reload the list from LM Studio or Ollama.
                Tip: pull `zai-org/GLM-4.6` for a single checkpoint that handles both text and images. Llama,
                DeepSeek, and other local models also work via the same dropdown.

                ### Creator Tab
                1. Describe what you want the AI to extract or generate
                2. Give it a name (or leave blank for auto-generation)
                3. Click **Create Pairing**
                4. View the generated JSON Schema and System Prompt

                ### Inference Tab
                1. Select a **Model** (top) and a **Pairing** from the dropdowns
                2. Preview the JSON Schema + prompt on the right before running
                3. Enter text, upload a .txt file, and/or upload image(s) for vision models (GLM-4.6, llava, etc.)
                4. Adjust temperature if needed (lower = more deterministic)
                5. Click **Run Inference** to see structured JSON output

                ### Editor Tab
                - Load any pairing and make manual edits to the schema or system prompt
                - Use **Format Schema JSON** to pretty-print and validate on the fly
                - Enter a new pairing name to bootstrap a blank template, then **Save**

                ### File Locations
                All pairings are saved in the `pairings/` directory.
                """
            )

    # Strip CLI args so Gradio/browser doesn't receive --gradio etc.
    import sys

    _argv = sys.argv
    sys.argv = [_argv[0]]
    try:
        # Gradio 6: theme passed to launch(); older versions use Blocks(theme=...)
        try:
            demo.launch(share=False, css=css, theme=lavender_theme)
        except TypeError:
            demo.launch(share=False, css=css)
    finally:
        sys.argv = _argv
