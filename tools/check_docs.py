"""Validate repository documentation without executing examples or fetching links."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

LINK = re.compile(r"\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")
SECRET = re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----")


def markdown_errors(path: Path, root: Path) -> list[str]:
    """Check local link targets, fenced JSON/SSE examples and balanced fences."""
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    outside: list[str] = []
    language: str | None = None
    block: list[str] = []
    for line in text.splitlines():
        if line.startswith("```"):
            if language is None:
                language = line[3:].strip()
                block = []
            else:
                payloads = ["\n".join(block)] if language == "json" else []
                payloads += [
                    item[6:] for item in block if item.startswith("data: ") and item[6:] != "[DONE]"
                ]
                for payload in payloads:
                    try:
                        json.loads(payload)
                    except json.JSONDecodeError as exc:
                        errors.append(f"{path.name}: invalid JSON example: {exc.msg}")
                language = None
        elif language is None:
            outside.append(line)
        else:
            block.append(line)
    if language is not None:
        errors.append(f"{path.name}: unclosed code fence")
    for target in LINK.findall("\n".join(outside)):
        target = target.strip("<>")
        url = urlsplit(target)
        if url.scheme or url.netloc:
            continue
        resolved = (path.parent / unquote(url.path)).resolve() if url.path else path
        if not resolved.is_relative_to(root.resolve()):
            errors.append(f"{path.name}: link escapes repository: {target}")
        elif not resolved.exists():
            errors.append(f"{path.name}: missing local link: {target}")
    return errors


def workflow_errors(path: Path) -> list[str]:
    """Check YAML and minimum privilege/pinning conventions used by this repo."""
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except yaml.YAMLError as exc:
        return [f"{path.name}: invalid YAML: {exc}"]
    if not isinstance(data, dict) or "on" not in data or "jobs" not in data:
        return [f"{path.name}: missing workflow trigger/jobs"]
    errors: list[str] = []
    triggers = data["on"]
    if "pull_request_target" in triggers:
        errors.append(f"{path.name}: privileged PR target trigger is not allowed")
    if data.get("permissions") != {"contents": "read"}:
        errors.append(f"{path.name}: default permissions must be contents: read")
    for job in data["jobs"].values():
        if "timeout-minutes" not in job:
            errors.append(f"{path.name}: job lacks timeout")
        for step in job.get("steps", []):
            action = step.get("uses", "")
            if action and not re.fullmatch(r"[\w./-]+@[a-f0-9]{40}", action):
                errors.append(f"{path.name}: action is not pinned to a commit: {action}")
            if action.startswith("actions/checkout@"):
                if step.get("with", {}).get("persist-credentials") != "false":
                    errors.append(f"{path.name}: checkout must not persist credentials")
    return errors


def traceability_errors(root: Path) -> list[str]:
    prds = root / "doc/prd"
    mapping = root / "doc/delivery/02-traceability.md"
    tests = root / "doc/testing/01-test-plan.md"
    features = root / "doc/delivery/01-feature-breakdown.md"
    if not prds.exists() and not mapping.exists():
        return []  # Early research increments do not yet declare product requirements.
    if not all(path.exists() for path in (prds, mapping, tests, features)):
        return ["Required traceability documents are missing"]
    required = set()
    for path in prds.glob("*.md"):
        required.update(re.findall(r"^\| (REQ-\w+) \|", path.read_text(encoding="utf-8"), re.M))
    text = mapping.read_text(encoding="utf-8")
    mapped = set(re.findall(r"^\| (REQ-\w+) \|", text, re.M))
    defined_tests = set(re.findall(r"^\| (T-\w+) \|", tests.read_text(encoding="utf-8"), re.M))
    defined_features = set(re.findall(r"^\| (F\d+) ", features.read_text(encoding="utf-8"), re.M))
    errors = []
    if required != mapped:
        errors.append(f"Requirement mapping mismatch: {sorted(required ^ mapped)}")
    missing_tests = set(re.findall(r"\bT-[A-Z]\d+\b", text)) - defined_tests
    missing_features = set(re.findall(r"\bF\d+\b", text)) - defined_features
    if missing_tests:
        errors.append(f"Undefined acceptance tests: {sorted(missing_tests)}")
    if missing_features:
        errors.append(f"Undefined implementation features: {sorted(missing_features)}")
    return errors


def validate(root: Path) -> list[str]:
    errors = []
    files = [root / "README.md", *sorted((root / "doc").rglob("*.md"))]
    for path in files:
        errors.extend(markdown_errors(path, root))
    for path in (root / "doc").rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}: invalid JSON: {exc.msg}")
    for path in (root / ".github/workflows").glob("*.yml"):
        errors.extend(workflow_errors(path))
    errors.extend(traceability_errors(root))
    for directory in (root / "doc", root / "tools", root / "tests", root / ".github"):
        for path in directory.rglob("*"):
            if path.suffix in {".md", ".py", ".yml", ".json", ".txt"}:
                if SECRET.search(path.read_text(encoding="utf-8")):
                    errors.append(f"{path.relative_to(root)}: credential-shaped content detected")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = validate(args.root.resolve())
    for error in errors:
        print(error)
    print(f"Documentation validation: {len(errors)} error(s)")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
