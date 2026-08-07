"""Install, configure, and run OpenCode inside a sandbox."""

import dataclasses
import json
import shlex

from ares.containers import containers

LOG_PATH = "/logs/agent/opencode.jsonl"
PID_PATH = "/tmp/ares-opencode.pid"
DEFAULT_VERSION = "1.14.48"


@dataclasses.dataclass(kw_only=True)
class OpenCodeCLI:
    container: containers.Container
    version: str = DEFAULT_VERSION

    async def install(self) -> None:
        command = " && ".join(
            [
                "if ! command -v curl >/dev/null; then "
                "if command -v apt-get >/dev/null; then apt-get update && apt-get install -y curl ca-certificates; "
                "elif command -v apk >/dev/null; then apk add --no-cache curl ca-certificates; "
                "else echo 'curl is required' >&2; exit 1; fi; fi",
                'if [ ! -s "$HOME/.nvm/nvm.sh" ]; then '
                "curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash; fi",
                '. "$HOME/.nvm/nvm.sh"',
                "nvm install 22",
                f"npm install -g opencode-ai@{shlex.quote(self.version)}",
            ]
        )
        result = await self.container.exec_run(command, timeout_s=600)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to install OpenCode: {result.output}")

    async def configure(self, proxy_url: str) -> None:
        config = {
            "$schema": "https://opencode.ai/config.json",
            "agent": {"title": {"disable": True}},
            "provider": {
                "ares": {
                    "name": "ARES",
                    "npm": "@ai-sdk/openai",
                    "env": [],
                    "options": {"apiKey": "unused", "baseURL": f"{proxy_url}/v1"},
                    "models": {
                        "ares": {
                            "name": "ARES-mediated model",
                            "limit": {"context": 1_000_000, "output": 128_000},
                        }
                    },
                }
            },
        }
        config_json = shlex.quote(json.dumps(config, separators=(",", ":")))
        result = await self.container.exec_run(
            f'mkdir -p "$HOME/.config/opencode" && printf %s {config_json} > "$HOME/.config/opencode/opencode.json"'
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to configure OpenCode: {result.output}")

    async def run(self, task: str) -> None:
        command = (
            '. "$HOME/.nvm/nvm.sh" && '
            f"echo $$ > {PID_PATH} && "
            "exec opencode --model=ares/ares run --format=json --thinking "
            f"--dangerously-skip-permissions -- {shlex.quote(task)}"
        )
        result = await self.container.exec_run(
            f"bash -lc {shlex.quote(command)} > {LOG_PATH} 2>&1",
            env={"OPENCODE_FAKE_VCS": "git"},
        )
        if result.exit_code != 0:
            log_result = await self.container.exec_run(f"tail -c 10000 {LOG_PATH}")
            raise RuntimeError(f"OpenCode exited with code {result.exit_code}: {log_result.output}")

    async def stop(self) -> None:
        await self.container.exec_run(f"if [ -f {PID_PATH} ]; then kill $(cat {PID_PATH}) 2>/dev/null || true; fi")
