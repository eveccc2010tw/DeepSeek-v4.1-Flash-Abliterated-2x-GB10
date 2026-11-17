import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).parents[1] / "tools/benchmark-matched.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class MatchedTests(unittest.TestCase):
    def test_http_request_disables_both_supported_thinking_protocols(self):
        import hashlib
        case = dict(messages=[{"role": "user", "content": "test"}],
                    scenario="short", trial=1, max_tokens=5, expected_prompt_tokens=103)
        case["prompt_sha256"] = hashlib.sha256(json.dumps(case["messages"]).encode()).hexdigest()
        class Reply(list):
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
        with patch.object(bench.urllib.request, "urlopen", return_value=Reply(self.fixture())) as send:
            result = bench.run_case("http://example.invalid/v1", "private-test-key", case)
        request = send.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["messages"], case["messages"])
        self.assertNotIn("private-test-key", json.dumps(result))

    def fixture(self):
        objects = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "ab"}}]},
            {"choices": [{"delta": {"content": "c"}, "finish_reason": "length"}]},
            {"usage": {"prompt_tokens": 103, "completion_tokens": 5,
                       "completion_tokens_details": {"reasoning_tokens": 0},
                       "prompt_tokens_details": {"cached_tokens": 0}}},
        ]
        lines = [b"data: " + json.dumps(o).encode() + b"\n" for o in objects] + [b"data: [DONE]\n"]
        return lines

    def test_sse_timing_uses_first_and_last_text_events_and_reported_tokens(self):
        times = iter([11, 13, 14])
        r = bench.collect(self.fixture(), 10, clock=lambda: next(times))
        self.assertEqual((r["ttft_seconds"], r["total_seconds"], r["decode_tokens_per_second"]), (1, 4, 2))
        self.assertEqual(r["last_token_seconds"], 3)
        self.assertTrue(bench.matched(r, {"expected_prompt_tokens": 103, "max_tokens": 5}))

    def test_incomplete_cached_reasoning_or_early_stop_cannot_enter_matched_summary(self):
        times = iter([11, 13, 14])
        good = bench.collect(self.fixture(), 10, clock=lambda: next(times))
        variants = [{"done": False}, {"finish_reason": "stop"}, {"reasoning_chars": 1},
                    {"decode_tokens_per_second": None}]
        for change in variants:
            self.assertFalse(bench.matched({**good, **change}, {"expected_prompt_tokens": 103, "max_tokens": 5}))
        for key, value in (("cached_tokens", 1), ("reasoning_tokens", 1)):
            r = copy.deepcopy(good)
            parent = "prompt_tokens_details" if key == "cached_tokens" else "completion_tokens_details"
            r["usage"][parent][key] = value
            self.assertFalse(bench.matched(r, {"expected_prompt_tokens": 103, "max_tokens": 5}))

    def test_recovered_cases_preserve_all_four_original_hashes(self):
        import hashlib
        cases = json.loads((Path(__file__).parents[1] / "examples/matched-speed-cases.json").read_text())
        self.assertEqual(len(cases), 4)
        for case in cases:
            self.assertEqual(hashlib.sha256(json.dumps(case["messages"]).encode()).hexdigest(), case["prompt_sha256"])
