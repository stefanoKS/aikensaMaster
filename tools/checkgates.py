"""Run Phase 2 software checkgates without using production equipment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
GATES = {
    "CG-BASELINE": (("compileall", "-q", "aikensa", "tests", "tools"), "Source syntax only."),
    "CG-CAMERA": (("pytest", "tests/test_camera_lifecycle.py"), "Fake cameras, no IC4 qualification."),
    "CG-GEOMETRY": (("pytest", "tests/test_geometry.py", "tests/test_merge_coverage.py", "tests/test_calibration_and_revisions.py"),
                    "Synthetic matrices; legacy production fallback coverage incomplete."),
    "CG-CONFIGURATION": (("pytest", "tests/test_daily_inspection.py", "tests/test_machine_configuration.py"),
                          "Daily overrides and legacy configuration; shared production schema pending."),
    "CG-DAILY": (("pytest", "tests/test_daily_inspection.py",
                  "tests/test_camera_lifecycle.py::test_daily_settings_change_drives_real_worker_and_rolls_back"),
                 "Synthetic camera/model input; physical inspection and MySQL not qualified."),
    "CG-PRODUCTION": (("pytest", "tests/test_inspection_strategy.py"), "Widget 17 parity; 18 migration pending."),
    "CG-SETTINGS": (("pytest", "tests/test_settings_ui.py"), "Offscreen Qt; daily draft preview pending."),
    "CG-RESULTS": (("pytest", "tests/test_daily_inspection.py::test_daily_recording_is_idempotent_and_failed_images_do_not_create_a_result"),
                   "Local SQLite and PNG checks; no physical database qualification."),
    "CG-UI": (("pytest", "tests/test_settings_ui.py"), "Offscreen Qt; physical display not qualified."),
    "CG-PERFORMANCE": (None, "Timed inspection workload benchmark not implemented."),
    "CG-REGRESSION": (("pytest",), "Software only; no physical-machine qualification."),
}


def check_gate(gate):
    args, limitation = GATES[gate]
    counts = dict(passed=0, failed=0, skipped=0, errors=0)
    if args is None:
        return dict(gate=gate, status="NOT TESTED", limitation=limitation, command=None,
                    exit_code=None, duration_seconds=0.0, counts=counts)
    cmd = [sys.executable, "-m", *args]
    start = perf_counter()
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "QT_QPA_PLATFORM": "offscreen"}
    with TemporaryDirectory(prefix="aikensa-checkgate-") as directory:
        report = Path(directory) / "pytest.xml"
        if args[0] == "pytest":
            cmd.extend(["-o", "addopts=", "-q", "--junitxml", str(report)])
        completed = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if report.is_file():
            for case in ET.parse(report).iter("testcase"):
                if case.find("error") is not None:
                    counts["errors"] += 1
                elif case.find("failure") is not None:
                    counts["failed"] += 1
                elif case.find("skipped") is not None:
                    counts["skipped"] += 1
                else:
                    counts["passed"] += 1
        success = completed.returncode == 0 and not any(
            counts[k] for k in ("failed", "skipped", "errors")
        ) and (args[0] != "pytest" or counts["passed"] > 0)
    return dict(
        gate=gate, status="PASS" if success else "FAIL", command=cmd,
        exit_code=completed.returncode, duration_seconds=round(perf_counter() - start, 3),
        counts=counts, limitation=limitation,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gates", nargs="*", metavar="GATE", help="Known gates: " + ", ".join(GATES))
    args = parser.parse_args(argv)
    gates = args.gates or ["CG-BASELINE", "CG-REGRESSION"]
    unknown = [gate for gate in gates if gate not in GATES]
    if unknown:
        parser.error("unknown gates: " + ", ".join(unknown))
    failed = False
    for gate in gates:
        result = check_gate(gate)
        print(json.dumps(result), flush=True)
        failed |= result["status"] == "FAIL"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
