from __future__ import annotations

import re


EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
LAN_IP_RE = re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})\b")
WIN_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s:\"'<>|]+")
UNIX_PATH_RE = re.compile(r"/(?:home|root|Users|var|etc|opt|srv|mnt|tmp)/[^\s:\"'<>|]+")
HOST_PORT_RE = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?::\d{2,5})\b")
SECRET_KV_RE = re.compile(
    r'(?<![A-Za-z0-9_])((?:"|\')?(?:access[_-]?token|api[_-]?key|secret|password|token|auth[_-]?token)(?:"|\')?)(\s*[:=]\s*)(["\']?)([A-Za-z0-9._\-+/=]{8,})(\3)',
    re.I,
)
AUTH_HEADER_RE = re.compile(r"(\bAuthorization\b\s*[:=]\s*)(Bearer\s+)?([A-Za-z0-9._\-+/=]{16,})", re.I)
JWT_LIKE_RE = re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])")
GENERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9_+\-/=]{24,})(?![A-Za-z0-9])")
MC_UUID_PLAYER_RE = re.compile(r"\b(UUID of player)\s+([A-Za-z0-9_]{2,32})\b", re.I)
USER_KV_RE = re.compile(r"\b(playername|username|user|ign|nickname|nick)\s*[:=]\s*([A-Za-z0-9_]{2,32})\b", re.I)


class PrivacyService:
    def replace_secret_kv(self, match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}<TOKEN>{match.group(5)}"

    def replace_auth_header(self, match: re.Match[str]) -> str:
        bearer = match.group(2) or ""
        return f"{match.group(1)}{bearer}<TOKEN>"

    def is_structured_key_context(self, text: str, start: int, end: int) -> bool:
        left = text[:start].rstrip()
        right = text[end:]
        if not re.match(r'^\s*["\']?\s*:', right):
            return False
        if not left:
            return True
        return bool(re.search(r'["\'{,\[\s]$', left))

    def looks_like_secret_token(self, token: str) -> bool:
        if len(token) < 24:
            return False

        has_lower = any(char.islower() for char in token)
        has_upper = any(char.isupper() for char in token)
        has_digit = any(char.isdigit() for char in token)
        has_symbol = any(char in "_-+/=" for char in token)

        if re.fullmatch(r"[a-z][a-z0-9_]{23,}", token):
            return False
        if not (has_digit or has_symbol):
            return False
        if has_lower and has_upper and has_digit:
            return True
        if has_symbol and (has_upper or has_digit):
            return True
        if has_lower and has_digit and len(token) >= 32 and token.count("_") <= 1 and token.count("-") <= 2:
            return True
        return False

    def replace_generic_token(self, match: re.Match[str]) -> str:
        token = match.group(1)
        if self.is_structured_key_context(match.string, match.start(1), match.end(1)):
            return token
        if self.looks_like_secret_token(token):
            return "<TOKEN>"
        return token

    def redact_text(self, text: str) -> str:
        if not text:
            return text
        out = str(text)
        out = WIN_PATH_RE.sub("<PATH>", out)
        out = UNIX_PATH_RE.sub("<PATH>", out)
        out = MC_UUID_PLAYER_RE.sub(r"\1 <USER>", out)
        out = USER_KV_RE.sub(r"\1=<USER>", out)
        out = LAN_IP_RE.sub("<LAN_IP>", out)
        out = IP_RE.sub("<IP>", out)
        out = HOST_PORT_RE.sub("<HOST>", out)
        out = EMAIL_RE.sub("<EMAIL>", out)
        out = UUID_RE.sub("<UUID>", out)
        out = SECRET_KV_RE.sub(self.replace_secret_kv, out)
        out = AUTH_HEADER_RE.sub(self.replace_auth_header, out)
        out = JWT_LIKE_RE.sub("<TOKEN>", out)
        out = GENERIC_TOKEN_RE.sub(self.replace_generic_token, out)
        return out

    def sanitize_for_persistence(self, text: str, limit: int = 4000) -> str:
        if not text:
            return ""
        clean = self.redact_text(text)
        if len(clean) > limit:
            return clean[:limit] + "...[truncated]"
        return clean

    def guard_for_llm(self, text: str) -> str:
        return self.redact_text(text)

    def guard_for_output(self, text: str) -> str:
        return self.redact_text(text)
