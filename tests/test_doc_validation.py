"""Regression tests for documentation checks; these are not GPU product tests."""

import tempfile
import unittest
from pathlib import Path

from tools.check_docs import SECRET, markdown_errors, workflow_errors


class DocumentationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.doc = self.root / "sample.md"

    def check_markdown(self, text):
        self.doc.write_text(text, encoding="utf-8")
        return markdown_errors(self.doc, self.root)

    def test_existing_local_and_external_links(self):
        (self.root / "target.md").write_text("# Target\n", encoding="utf-8")
        self.assertEqual([], self.check_markdown("[ok](target.md) [web](https://example.com)"))

    def test_missing_target(self):
        self.assertIn("missing local link", self.check_markdown("[bad](missing.md)")[0])

    def test_repository_escape(self):
        self.assertIn("escapes repository", self.check_markdown("[bad](../outside.md)")[0])

    def test_literal_links_in_examples_are_ignored(self):
        self.assertEqual([], self.check_markdown("```text\n[label](not-a-file)\n```\n"))

    def test_json_and_stream_framing_payloads(self):
        self.assertEqual([], self.check_markdown('```json\n{"a":1}\n```\n'))
        self.assertEqual([], self.check_markdown('```text\ndata: {"a":1}\n\ndata: [DONE]\n```'))

    def test_invalid_json_and_stream_payload(self):
        self.assertTrue(self.check_markdown("```json\n{broken}\n```"))
        self.assertTrue(self.check_markdown("```text\ndata: {broken}\n```"))

    def test_unclosed_fence(self):
        self.assertIn("unclosed", self.check_markdown("```python\nx = 1")[0])

    def test_spaces_in_local_path(self):
        (self.root / "two words.md").write_text("# Target", encoding="utf-8")
        self.assertEqual([], self.check_markdown("[ok](<two words.md>)"))

    def test_credential_shape_detection_without_real_secret(self):
        self.assertIsNotNone(SECRET.search("gh" + "p_" + "x" * 36))
        self.assertIsNone(SECRET.search("ordinary documentation"))

    def test_workflow_pin_and_permissions(self):
        path = self.root / "ci.yml"
        workflow = (
            "name: CI\non: [pull_request]\npermissions:\n  contents: read\n"
            "jobs:\n  docs:\n    timeout-minutes: 10\n    steps:\n"
            "      - uses: actions/checkout@" + "a" * 40 + "\n"
            "        with:\n          persist-credentials: false\n"
        )
        path.write_text(workflow, encoding="utf-8")
        self.assertEqual([], workflow_errors(path))
        path.write_text(workflow.replace("a" * 40, "v6"), encoding="utf-8")
        self.assertTrue(workflow_errors(path))

    def test_privileged_trigger_is_rejected(self):
        path = self.root / "ci.yml"
        path.write_text(
            "on: [pull_request_target]\npermissions:\n  contents: read\njobs: {}\n",
            encoding="utf-8",
        )
        self.assertIn("privileged", workflow_errors(path)[0])


if __name__ == "__main__":
    unittest.main()
