#!/usr/bin/env python3
"""Ground-truth renderer for test_render.jl: render gemma4_template.jinja with
real jinja2 for a list of cases fed as JSON on stdin, print JSON list of
rendered strings. bos_token renders empty because the Julia side never emits
BOS as text (tokenization adds it)."""
import json
import sys

from jinja2 import Environment, BaseLoader

spec = json.load(sys.stdin)
src = open(spec["template"]).read()
env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
tpl = env.from_string(src)


def raise_exception(msg):
    raise Exception(msg)


out = []
for case in spec["cases"]:
    out.append(tpl.render(
        messages=case.get("messages", []),
        tools=case.get("tools") or None,
        add_generation_prompt=case.get("add_generation_prompt", True),
        enable_thinking=case.get("enable_thinking", False),
        preserve_thinking=False,
        bos_token="",
        raise_exception=raise_exception,
    ))
print(json.dumps(out))
