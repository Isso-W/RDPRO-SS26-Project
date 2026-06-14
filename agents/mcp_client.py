"""Small official MCP client used by both agents."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import asynccontextmanager, contextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _decode_value(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@contextmanager
def _stdio_error_stream():
    """Provide a real file descriptor when running inside IPython/Colab."""
    try:
        sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stream:
            yield stream
    else:
        yield sys.stderr


def result_value(result):
    if getattr(result, "isError", False):
        messages = [
            getattr(item, "text", "")
            for item in getattr(result, "content", [])
            if getattr(item, "text", "")
        ]
        raise RuntimeError("\n".join(messages) or "MCP tool call failed.")
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        decoded = _decode_value(structured)
        if isinstance(decoded, dict) and set(decoded) == {"result"}:
            return _decode_value(decoded["result"])
        return decoded
    content = getattr(result, "content", [])
    for item in content:
        text = getattr(item, "text", None)
        if text:
            return _decode_value(text)
    return None


@asynccontextmanager
async def mcp_session():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=dict(os.environ),
    )
    with _stdio_error_stream() as errlog:
        async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session


async def call_tool(session: ClientSession, name: str, arguments: dict):
    return result_value(await session.call_tool(name, arguments))
