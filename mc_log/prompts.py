from __future__ import annotations

from pathlib import Path

from astrbot.api import logger

from .config import PROMPT_FILES


class PromptManager:
    def __init__(self, prompt_dir: Path):
        self.prompt_dir = Path(prompt_dir)
        self.prompts: dict[str, str] = {}
        self.prompts_ready = False

    def load_prompts(self):
        self.prompts = {}
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for key, filename in PROMPT_FILES.items():
            content = self.read_prompt_file(filename)
            if not content:
                missing.append(filename)
            self.prompts[key] = content
        self.prompts_ready = len(missing) == 0
        sizes = {k: len(v or "") for k, v in self.prompts.items()}
        logger.info(f"[mc_log] 提示词加载结果: ready={self.prompts_ready}, sizes={sizes}")
        if missing:
            logger.error(f"[mc_log] 提示词文件缺失或为空: {missing}; 目录: {self.prompt_dir}")

    def read_prompt_file(self, filename: str) -> str:
        path = self.prompt_dir / filename
        try:
            if not path.exists():
                return ""
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
            logger.warning(f"[mc_log] 提示词文件为空: {path}")
        except Exception as exc:
            logger.warning(f"[mc_log] 读取提示词文件失败 {path}: {exc}")
        return ""

    def get_prompt(self, key: str) -> str:
        return self.prompts.get(key, "")

    def render_prompt(self, template: str, values: dict[str, str]) -> str:
        out = template
        for key, value in values.items():
            out = out.replace(f"{{{{{key}}}}}", str(value))
        return out
