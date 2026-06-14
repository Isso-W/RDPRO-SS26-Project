"""Parse the final JSON object emitted by generated Module 4 scripts."""

from __future__ import annotations

import json


def extract_last_json(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    last = None
    index = 0
    while index < len(text):
        if text[index] == "{":
            try:
                value, consumed = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                index += 1
                continue
            if isinstance(value, dict):
                last = value
            index += consumed
            continue
        index += 1
    return last
