"""Test backends.vllm.parse_tool_calls covers both Qwen3-8B and Qwen3.6+ formats.

Qwen3-8B chat_template emits JSON body: ``<tool_call>{"name":..., "arguments":...}</tool_call>``.
Qwen3.6+ chat_template emits XML body: ``<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>``.
The parser must handle BOTH so the same backend works against either model.

Sample XML strings below were copied verbatim from real failing trajectories under
``analysis/run/hle/qwen36_35b_a3b_fp8_baseline/run_20260430_074257/trajectories/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make backends.vllm importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.vllm import parse_tool_calls  # noqa: E402


def test_qwen3_8b_json_format_no_args():
    text = '<tool_call>\n{"name": "explore", "arguments": {}}\n</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [("explore", {})], calls


def test_qwen3_8b_json_format_structured_output():
    text = (
        '<tool_call>\n{"name": "StructuredOutput", '
        '"arguments": {"answer": "B", "reasoning": "x", "confidence": 0.85}}\n'
        '</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    name, args = calls[0]
    assert name == "StructuredOutput"
    assert args == {"answer": "B", "reasoning": "x", "confidence": 0.85}, args


def test_qwen36_xml_format_no_args():
    # Verbatim from qid 668825f80a642802bdfeadfa
    text = "<tool_call>\n<function=explore>\n</function>\n</tool_call>"
    calls = parse_tool_calls(text)
    assert calls == [("explore", {})], calls


def test_qwen36_xml_format_structured_output_with_params():
    text = (
        "<tool_call>\n<function=StructuredOutput>\n"
        "<parameter=answer>\nB\n</parameter>\n"
        "<parameter=reasoning>\nbecause of X\n</parameter>\n"
        "<parameter=confidence>\n0.85\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    name, args = calls[0]
    assert name == "StructuredOutput"
    # confidence (numeric) JSON-decoded to float; strings pass through trimmed.
    assert args == {"answer": "B", "reasoning": "because of X", "confidence": 0.85}, args


def test_qwen36_xml_format_string_value_with_quotes_inside():
    # Strings with embedded special chars must NOT be JSON-decoded as numbers.
    text = (
        "<tool_call>\n<function=StructuredOutput>\n"
        "<parameter=answer>\nKatie kicked the knotted kite string.\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert calls == [("StructuredOutput", {"answer": "Katie kicked the knotted kite string."})], calls


def test_xml_format_multiline_value():
    text = (
        "<tool_call>\n<function=StructuredOutput>\n"
        "<parameter=reasoning>\nLine 1\nLine 2\nLine 3\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text)
    name, args = calls[0]
    assert name == "StructuredOutput"
    assert args == {"reasoning": "Line 1\nLine 2\nLine 3"}, args


def test_no_tool_call_returns_empty():
    assert parse_tool_calls("") == []
    assert parse_tool_calls("just thinking, no tool call here") == []


def test_multiple_tool_calls_mixed_formats():
    # Should not happen in practice (one model = one format) but parser must be robust.
    text = (
        '<tool_call>\n{"name": "explore", "arguments": {}}\n</tool_call>\n'
        "Some text in between.\n"
        "<tool_call>\n<function=explore>\n</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert calls == [("explore", {}), ("explore", {})], calls


def test_malformed_tool_call_skipped_silently():
    text = "<tool_call>\nthis is neither JSON nor XML\n</tool_call>"
    assert parse_tool_calls(text) == []


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"\nALL {len(tests)} PASSED")
