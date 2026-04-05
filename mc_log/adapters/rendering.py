from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import textwrap
import time
from pathlib import Path

import astrbot.api.message_components as Comp
from astrbot.api import logger

from ..config import MAX_CODE_BLOCK_MESSAGE_CHARS

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


class RenderingAdapter:
    def __init__(self, config_manager, runtime, html_template_path: Path, html_render_func):
        self.config_manager = config_manager
        self.runtime = runtime
        self.html_template_path = Path(html_template_path)
        self.html_render_func = html_render_func

    def _cfg(self):
        return self.config_manager.get()

    def _msg(self, key: str) -> str:
        return self.config_manager.msg(key)

    def _preview_text(self, text: str, limit: int = 200) -> str:
        if text is None:
            return ""
        sample = str(text).replace("\r", " ").replace("\n", "\\n")
        if len(sample) <= limit:
            return sample
        return sample[:limit] + "...[truncated]"

    async def render_report(self, markdown_text: str, run_id: str = "", deadline: float | None = None) -> tuple[str, str]:
        mode = self._cfg()["render_mode"]
        if mode == "html_to_image":
            try:
                timeout_sec = min(float(self._cfg()["html_render_timeout_sec"]), max(0.1, self.runtime.time_left(deadline)))
                url = await asyncio.wait_for(self.render_markdown_html(markdown_text), timeout=timeout_sec)
                if url:
                    logger.info(f"[mc_log][{run_id}] HTML渲染成功: url={self._preview_text(url, 200)}")
                    return "image_url", url
            except asyncio.TimeoutError:
                logger.warning(f"[mc_log][{run_id}] HTML渲染超时: {self._cfg()['html_render_timeout_sec']}s")
                if self.runtime.time_left(deadline) <= 0:
                    raise TimeoutError("global timeout reached during render")
            except Exception as exc:
                logger.warning(f"[mc_log][{run_id}] HTML渲染失败: {exc}")
            fallback = markdown_text + "\n\n" + self._msg("html_render_fallback_notice")
            return "text", fallback

        if mode == "text_to_image":
            try:
                b64 = self.render_text_image_base64(markdown_text)
                logger.info(f"[mc_log][{run_id}] 文本转图成功: b64_chars={len(b64)}")
                return "image_b64", b64
            except Exception as exc:
                logger.warning(f"[mc_log][{run_id}] 文本转图片渲染失败: {exc}")
                fallback = markdown_text + "\n\n" + self._msg("text_render_fallback_notice")
                return "text", fallback

        logger.warning(f"[mc_log][{run_id}] 未知渲染模式: {mode}，回退到HTML渲染")
        try:
            url = await asyncio.wait_for(self.render_markdown_html(markdown_text), timeout=self._cfg()["html_render_timeout_sec"])
            if url:
                return "image_url", url
        except Exception as exc:
            logger.warning(f"[mc_log][{run_id}] 未知模式回退后 HTML渲染仍失败: {exc}")
        fallback = markdown_text + "\n\n" + self._msg("html_render_fallback_notice")
        return "text", fallback

    async def render_markdown_html(self, markdown_text: str) -> str:
        if not self.html_template_path.exists():
            raise FileNotFoundError(f"html template not found: {self.html_template_path}")
        template = self.html_template_path.read_text(encoding="utf-8")
        time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime())
        image_width = int(self._cfg().get("image_width", 640))
        url = await self.html_render_func(
            template,
            {
                "message": markdown_text,
                "text": markdown_text,
                "time": time_str,
                "image_width": image_width,
            },
            options={
                "type": "png",
                "quality": None,
                "omit_background": True,
                "full_page": True,
                "viewport_width": image_width,
                "animations": "disabled",
                "caret": "hide",
                "scale": "css",
            },
        )
        return str(url)

    def render_text_image_base64(self, text: str) -> str:
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow not available")
        lines = []
        for raw_line in text.splitlines():
            wrapped = textwrap.wrap(raw_line, width=50) or [""]
            lines.extend(wrapped)
        if not lines:
            lines = [""]

        font = self.load_font()
        line_height = 30
        width = int(self._cfg().get("image_width", 640))
        margin = 36
        height = margin * 2 + line_height * len(lines) + 24
        image = Image.new("RGB", (width, height), color=(248, 251, 248))
        draw = ImageDraw.Draw(image)
        y = margin
        for line in lines:
            draw.text((margin, y), line, fill=(28, 40, 32), font=font)
            y += line_height

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def load_font(self):
        if not PIL_AVAILABLE:
            return None
        for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arial.ttf"):
            if os.path.exists(candidate):
                try:
                    return ImageFont.truetype(candidate, 22)
                except Exception:
                    continue
        return ImageFont.load_default()

    def build_forward_response(self, event, source_name: str, strategy: str, elapsed: float, render_mode: str, render_payload: str):
        sender_name = self._msg("forward_sender_name")
        sender_uin = str(event.get_self_id() or "0")
        summary_template = self._msg("summary_template")
        try:
            summary = summary_template.format(elapsed=elapsed, source_name=source_name, strategy=strategy)
        except Exception:
            summary = self._cfg()["messages"]["summary_template"].format(
                elapsed=elapsed,
                source_name=source_name,
                strategy=strategy,
            )
        node1 = Comp.Node(content=[Comp.Plain(summary)], name=sender_name, uin=sender_uin)
        if render_mode == "image_url":
            core_content = [Comp.Image.fromURL(render_payload)]
        elif render_mode == "image_b64":
            core_content = [Comp.Image.fromBase64(render_payload)]
        else:
            core_content = [Comp.Plain(render_payload)]
        node2 = Comp.Node(content=core_content, name=sender_name, uin=sender_uin)
        return event.chain_result([Comp.Nodes(nodes=[node1, node2])])

    def build_code_blocks_message(self, markdown_text: str) -> str:
        blocks = self.extract_fenced_code_blocks(markdown_text)
        if not blocks:
            return ""
        logger.info(f"[mc_log] 检测到代码块数量: {len(blocks)}")
        parts = ["以下是分析结果中的代码块（便于复制）："]
        remaining = MAX_CODE_BLOCK_MESSAGE_CHARS - len(parts[0]) - 2
        for idx, (lang, code) in enumerate(blocks, start=1):
            fence_lang = lang or "text"
            snippet = f"\n\n[代码块 {idx}]\n```{fence_lang}\n{code}\n```"
            if len(snippet) <= remaining:
                parts.append(snippet)
                remaining -= len(snippet)
                continue
            if remaining <= 80:
                break
            max_code_len = max(40, remaining - len(f"\n\n[代码块 {idx}]\n```{fence_lang}\n\n```") - 20)
            clipped = code[:max_code_len] + "\n...[代码块内容已截断]..."
            parts.append(f"\n\n[代码块 {idx}]\n```{fence_lang}\n{clipped}\n```")
            remaining = 0
            break
        return "".join(parts)

    def extract_fenced_code_blocks(self, markdown_text: str) -> list[tuple[str, str]]:
        if not markdown_text:
            return []
        pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
        out: list[tuple[str, str]] = []
        for match in pattern.finditer(markdown_text):
            lang = (match.group(1) or "").strip()
            code = (match.group(2) or "").strip("\n")
            if not code.strip():
                continue
            out.append((lang, code))
        return out
