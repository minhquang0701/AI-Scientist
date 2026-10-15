import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jsonschema


CODEX_HEADLESS_PREFIX = "codex-headless"


def is_codex_headless_model(model: str | None) -> bool:
    return bool(model) and (
        model == CODEX_HEADLESS_PREFIX
        or model.startswith(f"{CODEX_HEADLESS_PREFIX}/")
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _resolve_codex_model(model: str) -> str | None:
    if model.startswith(f"{CODEX_HEADLESS_PREFIX}/"):
        return model.split("/", 1)[1]
    return os.environ.get("CODEX_HEADLESS_MODEL") or "gpt-5.5"


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and item.get("type") == "image_url":
                parts.append("[image attached]")
            else:
                parts.append(_stringify_content(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "type" in content:
            return _stringify_content([content])
        return json.dumps(content, indent=2, ensure_ascii=False)
    return str(content)


def build_chat_prompt(
    *,
    system_message: str | None,
    msg_history: list[dict[str, Any]] | None,
    user_message: str | None,
    extra_instructions: str | None = None,
) -> str:
    sections = [
        "You are running as a non-interactive model backend for AI-Scientist-v2.",
        "Do not modify files or run commands unless the prompt explicitly asks for shell output.",
        "Return only the response requested by the prompt. Do not add assistant framing.",
    ]
    if extra_instructions:
        sections.append(extra_instructions.strip())
    if os.environ.get("CODEX_HEADLESS_INSTRUCTIONS"):
        sections.append(os.environ["CODEX_HEADLESS_INSTRUCTIONS"].strip())
    if system_message:
        sections.extend(["\n# System Message", system_message.strip()])
    if msg_history:
        sections.append("\n# Conversation History")
        for message in msg_history:
            role = message.get("role", "unknown")
            content = _stringify_content(message.get("content"))
            sections.append(f"\n## {role}\n{content}")
    if user_message:
        sections.extend(["\n# Current User Message", user_message.strip()])
    return "\n".join(sections).strip() + "\n"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
        if match:
            stripped = match.group(1).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            parsed, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Codex headless response did not contain a JSON object: {text[:500]}")


class CodexHeadlessClient:
    def __init__(self, model: str = CODEX_HEADLESS_PREFIX):
        self.model_alias = model
        self.codex_model = _resolve_codex_model(model)
        self.codex_bin = os.environ.get("CODEX_BIN") or shutil.which("codex")
        if not self.codex_bin:
            raise ValueError(
                "Codex headless requested, but no `codex` executable was found. "
                "Set CODEX_BIN or add codex to PATH."
            )
        self.timeout = int(os.environ.get("CODEX_HEADLESS_TIMEOUT", "1800"))
        self.sandbox = os.environ.get("CODEX_HEADLESS_SANDBOX", "read-only")
        self.cwd = os.environ.get("CODEX_HEADLESS_CWD") or os.getcwd()

    def complete(
        self,
        *,
        system_message: str | None = None,
        user_message: str | None = None,
        msg_history: list[dict[str, Any]] | None = None,
        images: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        prompt = build_chat_prompt(
            system_message=system_message,
            msg_history=msg_history,
            user_message=user_message,
            extra_instructions=extra_instructions,
        )
        return self.complete_raw(
            prompt,
            images=images,
            output_schema=output_schema,
        )

    def complete_raw(
        self,
        prompt: str,
        *,
        images: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        output_file = tempfile.NamedTemporaryFile(
            prefix="codex-headless-response-", suffix=".txt", delete=False
        )
        output_file.close()
        schema_path = None
        try:
            cmd = [
                self.codex_bin,
                "exec",
                "--sandbox",
                self.sandbox,
                "--cd",
                self.cwd,
                "--color",
                "never",
                "--output-last-message",
                output_file.name,
            ]
            if _env_flag("CODEX_HEADLESS_EPHEMERAL", True):
                cmd.append("--ephemeral")
            if _env_flag("CODEX_HEADLESS_SKIP_GIT_CHECK", True):
                cmd.append("--skip-git-repo-check")
            if self.codex_model:
                cmd.extend(["--model", self.codex_model])
            if output_schema is not None:
                schema_file = tempfile.NamedTemporaryFile(
                    prefix="codex-headless-schema-",
                    suffix=".json",
                    mode="w",
                    encoding="utf-8",
                    delete=False,
                )
                json.dump(output_schema, schema_file)
                schema_file.close()
                schema_path = schema_file.name
                cmd.extend(["--output-schema", schema_path])
            for image in images or []:
                if image and Path(image).exists():
                    cmd.extend(["--image", image])
            extra_args = os.environ.get("CODEX_HEADLESS_EXTRA_ARGS", "")
            if extra_args:
                cmd.extend(shlex.split(extra_args))
            cmd.append("-")

            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                check=False,
            )
            with open(output_file.name, "r", encoding="utf-8") as f:
                response = f.read().strip()
            if proc.returncode != 0:
                stderr_tail = proc.stderr.strip()[-4000:]
                raise RuntimeError(
                    f"Codex headless failed with exit code {proc.returncode}: {stderr_tail}"
                )
            if response:
                return response
            stdout = proc.stdout.strip()
            if stdout:
                return stdout.splitlines()[-1].strip()
            raise RuntimeError("Codex headless returned an empty response.")
        finally:
            for path in (output_file.name, schema_path):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def complete_json(
        self,
        *,
        system_message: str | None,
        user_message: str | None,
        output_schema: dict[str, Any],
        function_name: str | None = None,
        function_description: str | None = None,
    ) -> dict[str, Any]:
        instruction = (
            "Return a single JSON object that validates against the provided schema."
        )
        if function_name:
            instruction += (
                f" Treat this JSON object as the arguments for function `{function_name}`."
            )
        if function_description:
            instruction += f"\nFunction description: {function_description}"
        instruction += (
            "\nThe JSON object must validate against this schema:\n"
            f"{json.dumps(output_schema, indent=2)}"
        )

        response = self.complete(
            system_message=system_message,
            user_message=user_message,
            output_schema=output_schema,
            extra_instructions=instruction,
        )
        parsed = extract_json_object(response)
        jsonschema.Draft7Validator(output_schema).validate(parsed)
        return parsed
