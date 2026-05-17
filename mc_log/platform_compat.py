from __future__ import annotations

from typing import Any


WEIXIN_OC_HUB_PLATFORM_NAME = "weixin_oc_hub"
HUB_SESSION_SEPARATOR = "%"


def event_platform_name(event: Any) -> str:
    try:
        return str(event.get_platform_name() or "").strip()
    except Exception:
        return ""


def event_session_id(event: Any) -> str:
    try:
        return str(event.get_session_id() or "").strip()
    except Exception:
        return ""


def is_weixin_oc_hub_event(event: Any) -> bool:
    return event_platform_name(event) == WEIXIN_OC_HUB_PLATFORM_NAME


def should_force_plain_text_output(event: Any) -> bool:
    return is_weixin_oc_hub_event(event)


def hub_remote_session_id(session_id: str) -> str:
    if HUB_SESSION_SEPARATOR not in session_id:
        return ""
    _account_key, _separator, remote_user_id = session_id.partition(
        HUB_SESSION_SEPARATOR
    )
    return remote_user_id.strip()


def session_whitelist_candidates(event: Any) -> set[str]:
    session_id = event_session_id(event)
    candidates = {session_id} if session_id else set()

    unified_origin = str(getattr(event, "unified_msg_origin", "") or "").strip()
    if unified_origin:
        candidates.add(unified_origin)

    if not is_weixin_oc_hub_event(event) or not session_id:
        return candidates

    remote_session_id = hub_remote_session_id(session_id)
    if remote_session_id:
        candidates.add(remote_session_id)

    platform_name = event_platform_name(event)
    candidates.add(f"{platform_name}:FriendMessage:{session_id}")
    candidates.add(f"{platform_name}:FRIEND_MESSAGE:{session_id}")
    return candidates
