"""The gate runner propagates test failures and does not pass skipped checks."""

from types import SimpleNamespace

import pytest

from tools import checkgates


@pytest.mark.parametrize("outcome,exit_code", [("failure", 1), ("skipped", 0)])
def test_required_checkgate_fails_without_a_verified_test(monkeypatch, outcome, exit_code):
    from pathlib import Path

    def run(command, **_kwargs):
        Path(command[-1]).write_text(f"<testsuite><testcase><{outcome}/></testcase></testsuite>")
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(checkgates.subprocess, "run", run)
    assert checkgates.main(["CG-DAILY"]) == 1


def test_unimplemented_checkgate_is_not_reported_as_pass(capsys):
    assert checkgates.main(["CG-PERFORMANCE"]) == 0
    assert '"status": "NOT TESTED"' in capsys.readouterr().out
