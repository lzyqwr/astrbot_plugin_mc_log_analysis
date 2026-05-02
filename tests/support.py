from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace


DATA_ROOT = Path(__file__).resolve().parent / "_tmp_data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def install_astrbot_stubs():
    if "astrbot" in sys.modules:
        return

    astrbot_mod = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    message_mod = types.ModuleType("astrbot.api.message_components")
    event_mod = types.ModuleType("astrbot.api.event")
    event_filter_mod = types.ModuleType("astrbot.api.event.filter")
    star_mod = types.ModuleType("astrbot.api.star")
    core_mod = types.ModuleType("astrbot.core")
    agent_mod = types.ModuleType("astrbot.core.agent")
    tool_mod = types.ModuleType("astrbot.core.agent.tool")
    utils_mod = types.ModuleType("astrbot.core.utils")
    path_mod = types.ModuleType("astrbot.core.utils.astrbot_path")

    logger = logging.getLogger("astrbot_test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    class Plain:
        def __init__(self, text):
            self.text = text

    class File:
        def __init__(self, name="", file="", url=""):
            self.name = name
            self.file_ = file
            self.url = url

        async def get_file(self, allow_return_url=True):
            return self.file_ or self.url

    class Image:
        def __init__(self, value, mode):
            self.value = value
            self.mode = mode

        @classmethod
        def fromURL(cls, value):
            return cls(value, "url")

        @classmethod
        def fromBase64(cls, value):
            return cls(value, "b64")

        @classmethod
        def fromFileSystem(cls, value):
            return cls(value, "file")

    class Node:
        def __init__(self, content, name="", uin=""):
            self.content = content
            self.name = name
            self.uin = uin

    class Nodes:
        def __init__(self, nodes):
            self.nodes = nodes

    class FakeResult:
        def __init__(self, payload):
            self.payload = payload
            self.stopped = False

        def stop_event(self):
            self.stopped = True

    class AstrMessageEvent:
        pass

    class Star:
        def __init__(self, context):
            self.context = context

        async def html_render(self, template, values, options=None):
            return "http://example.com/rendered.png"

    class Context:
        pass

    class PermissionType:
        ADMIN = "admin"

    def noop_decorator(*args, **kwargs):
        def _wrap(obj):
            return obj

        return _wrap

    class EventMessageType:
        ALL = "all"

    class FunctionTool:
        def __init__(self, name, description, parameters, handler):
            self.name = name
            self.description = description
            self.parameters = parameters
            self.handler = handler

    class ToolSet:
        def __init__(self):
            self.tools = []

        def add_tool(self, tool):
            self.tools.append(tool)

        def __len__(self):
            return len(self.tools)

    def register(*args, **kwargs):
        def _wrap(cls):
            return cls

        return _wrap

    def event_message_type(*args, **kwargs):
        return noop_decorator(*args, **kwargs)

    def get_astrbot_data_path():
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        return str(DATA_ROOT)

    message_mod.Plain = Plain
    message_mod.File = File
    message_mod.Image = Image
    message_mod.Node = Node
    message_mod.Nodes = Nodes
    api_mod.logger = logger
    filter_namespace = SimpleNamespace(
        PermissionType=PermissionType,
        permission_type=noop_decorator,
        command=noop_decorator,
    )

    event_mod.AstrMessageEvent = AstrMessageEvent
    event_mod.filter = filter_namespace
    event_filter_mod.EventMessageType = EventMessageType
    event_filter_mod.event_message_type = event_message_type
    star_mod.Context = Context
    star_mod.Star = Star
    star_mod.register = register
    tool_mod.FunctionTool = FunctionTool
    tool_mod.ToolSet = ToolSet
    path_mod.get_astrbot_data_path = get_astrbot_data_path

    sys.modules["astrbot"] = astrbot_mod
    sys.modules["astrbot.api"] = api_mod
    sys.modules["astrbot.api.message_components"] = message_mod
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.event.filter"] = event_filter_mod
    sys.modules["astrbot.api.star"] = star_mod
    sys.modules["astrbot.core"] = core_mod
    sys.modules["astrbot.core.agent"] = agent_mod
    sys.modules["astrbot.core.agent.tool"] = tool_mod
    sys.modules["astrbot.core.utils"] = utils_mod
    sys.modules["astrbot.core.utils.astrbot_path"] = path_mod

    astrbot_mod.api = api_mod
    core_mod.agent = agent_mod
    core_mod.utils = utils_mod
    agent_mod.tool = tool_mod
    utils_mod.astrbot_path = path_mod

    return FakeResult


class FakeToolManager:
    def __init__(self, tools):
        self._tools = tools

    def get_func(self, name):
        return self._tools.get(name)


class FakeContext:
    def __init__(self, provider_ids=None):
        self.provider_ids = set(provider_ids or [])
        self.tools = {}
        self.next_llm_text = "分析结果"

    def add_llm_tools(self, *tools):
        for tool in tools:
            self.tools[tool.name] = tool

    def get_llm_tool_manager(self):
        return FakeToolManager(self.tools)

    def get_provider_by_id(self, provider_id):
        return object() if provider_id in self.provider_ids else None

    async def llm_generate(self, **kwargs):
        return SimpleNamespace(completion_text=self.next_llm_text)

    async def tool_loop_agent(self, **kwargs):
        return SimpleNamespace(completion_text=self.next_llm_text)


class FakeMessageObj:
    def __init__(self, message_id="mid-1", raw_message=None):
        self.message_id = message_id
        self.raw_message = raw_message or {}


class FakeEvent:
    def __init__(
        self,
        messages=None,
        message_id="mid-1",
        raw_message=None,
        sender_id="user-1",
        sender_name="tester",
        group_id="group-1",
        session_id="session-1",
        platform_name="test",
        self_id="bot-1",
    ):
        self._messages = list(messages or [])
        self.message_obj = FakeMessageObj(message_id=message_id, raw_message=raw_message)
        self._sender_id = sender_id
        self._sender_name = sender_name
        self._group_id = group_id
        self._session_id = session_id
        self._platform_name = platform_name
        self._self_id = self_id

    def get_messages(self):
        return list(self._messages)

    def plain_result(self, text):
        return SimpleNamespace(payload=text, stopped=False, stop_event=lambda: setattr(self, "_stopped", True))

    def chain_result(self, payload):
        return SimpleNamespace(payload=payload, stopped=False, stop_event=lambda: setattr(self, "_stopped", True))

    def get_sender_id(self):
        return self._sender_id

    def get_sender_name(self):
        return self._sender_name

    def get_group_id(self):
        return self._group_id

    def get_session_id(self):
        return self._session_id

    def get_platform_name(self):
        return self._platform_name

    def get_self_id(self):
        return self._self_id


class FakeFileComponent:
    def __init__(self, name="", source=""):
        self.name = name
        self.file_ = source
        self.url = ""

    async def get_file(self, allow_return_url=True):
        return self.file_
