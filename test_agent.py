"""Offline checks: python -m unittest test_agent -v (no API, no training)."""
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

import agent


def response(proposal, **extra):
    return dict(status="completed", output=[dict(type="message", content=[
                dict(type="output_text", text=json.dumps(proposal))])], **extra)


class AgentTests(unittest.TestCase):
    def test_no_history_hides_search_outcomes_but_agent_receives_them(self):
        rows = [{"trial": 0, "final_dev_loss": 4.9},
                {"trial": 1, "final_dev_loss": 12345.6789, "hypothesis": "unique previous outcome"}]
        hidden = json.dumps(agent.build_context({}, rows, "no-history"))
        visible = json.dumps(agent.build_context({}, rows, "agent"))
        self.assertNotIn("12345.6789", hidden)
        self.assertNotIn("unique previous outcome", hidden)
        self.assertIn("12345.6789", visible)
        self.assertEqual(agent.build_context({}, rows, "no-history")["baseline"], rows[0])

    def test_unknown_and_repeated_candidates_are_rejected(self):
        proposal = dict(candidate_id="c000", hypothesis="test", expected_effect="test")
        self.assertEqual(agent.parse_response(response(proposal), {"c000": {}}), proposal)
        with self.assertRaises(ValueError):
            agent.parse_response(response(proposal), {"c001": {}})

    def test_refusal_and_incomplete_response_do_not_become_proposals(self):
        refusal = dict(status="completed", output=[dict(type="message", content=[dict(type="refusal")])])
        for item in (refusal, dict(status="incomplete", output=[])):
            with self.assertRaises(ValueError):
                agent.parse_response(item, agent.candidates())

    def test_budget_or_code_changes_prevent_method_comparison(self):
        plans = [{k: {} for k in ("base_config", "search_space", "trials", "data_sha256", "source_sha256")},
                 {k: {} for k in ("base_config", "search_space", "trials", "data_sha256", "source_sha256")}]
        plans[1]["base_config"] = {"max_steps": 999}
        summary = dict(status="completed", baseline_dev_loss=4.9)
        with patch.object(agent, "read_json", side_effect=[plans[0], summary, plans[1], summary]):
            with self.assertRaisesRegex(ValueError, "different budgets"):
                agent.compare(["a", "b"])

    def test_http_key_is_only_in_header_and_errors_redact_it(self):
        key = "offline-test-secret"
        body = agent.make_request("test-model", 2048, {"c000": {}}, {})
        seen = []

        def fake_open(request, timeout):
            seen.append(request)
            return io.StringIO(json.dumps(response(dict(candidate_id="c000", hypothesis="a", expected_effect="b"))))

        with patch.object(agent.urllib.request, "urlopen", side_effect=fake_open):
            result = agent.call_openai(body, key)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(seen[0].get_header("Authorization"), "Bearer " + key)
        self.assertNotIn(key, seen[0].data.decode())
        self.assertEqual(json.loads(seen[0].data)["store"], False)
        error = urllib.error.HTTPError("https://api.openai.com/v1/responses", 401, "Unauthorized", {}, io.BytesIO(key.encode()))
        with patch.object(agent.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                agent.call_openai(body, key)
        self.assertNotIn(key, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
