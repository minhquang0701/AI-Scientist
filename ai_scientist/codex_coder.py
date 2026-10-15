import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from ai_scientist.codex_headless import _env_flag, _resolve_codex_model


class CodexCoder:
    """Small adapter that gives `codex exec` the `.run(prompt)` API v1 expects."""

    def __init__(self, model: str, fnames: list[str], cwd: str):
        self.model_alias = model
        self.codex_model = _resolve_codex_model(model)
        self.fnames = [str(Path(fname).resolve()) for fname in fnames]
        self.cwd = str(Path(cwd).resolve())
        self.codex_bin = os.environ.get("CODEX_BIN") or shutil.which("codex")
        if not self.codex_bin:
            raise ValueError(
                "Codex coder requested, but no `codex` executable was found. "
                "Set CODEX_BIN or add codex to PATH."
            )
        self.timeout = int(os.environ.get("CODEX_CODER_TIMEOUT", "1800"))
        self.sandbox = os.environ.get("CODEX_CODER_SANDBOX", "workspace-write")

    def run(self, prompt: str) -> str:
        output_file = tempfile.NamedTemporaryFile(
            prefix="codex-coder-response-", suffix=".txt", delete=False
        )
        output_file.close()
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

            extra_args = os.environ.get("CODEX_CODER_EXTRA_ARGS", "")
            if extra_args:
                cmd.extend(shlex.split(extra_args))
            cmd.append("-")

            editable = "\n".join(f"- {path}" for path in self.fnames)
            full_prompt = f"""You are the code-editing backend for AI-Scientist v1.
Work inside this experiment directory:
{self.cwd}

Editable files:
{editable}

Follow the user's request by editing the relevant files directly. Keep edits focused.
Do not run long training jobs. Quick syntax checks or tiny inspections are fine.
If the prompt asks you to finish by saying ALL_COMPLETED, only say it when the
experiment plan is genuinely complete.

User request:
{prompt}
"""
            proc = subprocess.run(
                cmd,
                input=full_prompt,
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
                    f"Codex coder failed with exit code {proc.returncode}: {stderr_tail}"
                )
            if response:
                return response
            stdout = proc.stdout.strip()
            return stdout.splitlines()[-1].strip() if stdout else ""
        finally:
            try:
                os.unlink(output_file.name)
            except OSError:
                pass
