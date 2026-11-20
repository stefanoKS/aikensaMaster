"""Machine configuration backup / verify / restore / status utility.

Usage (from the repository root)::

    python tools/machine_config_backup.py backup
    python tools/machine_config_backup.py verify  [--backup DIR]
    python tools/machine_config_backup.py restore [--backup DIR] [--dry-run]
    python tools/machine_config_backup.py status

The backup is written *outside* the repository work tree (default:
``<repo>/../aikensa_backups``) so a ``git pull`` can never touch it.
Every copied file is SHA-256 hashed and re-verified before the tool reports
success.  ``restore`` refuses to apply a backup that does not verify.

Exit codes: 0 = success, 1 = verification problem, 2 = usage error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aikensa.core.settings.migration import (  # noqa: E402
    backup_machine_files,
    collect_runtime_files,
    list_backups,
    migration_status,
    restore_machine_files,
    verify_backup,
)
from aikensa.core.settings.paths import AppPaths  # noqa: E402


def _git_tracked(paths: AppPaths) -> set[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(paths.root), "ls-files"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def cmd_backup(paths: AppPaths, args) -> int:
    destination = Path(args.backup) if args.backup else None
    result = backup_machine_files(paths, destination, reason=args.reason)

    print(f"backup directory : {result.backup_dir}")
    print(f"files copied     : {len(result.copied)}")
    for name in result.copied:
        print(f"  - {name}")
    print(f"manifest         : {result.manifest_path}")

    if result.ok:
        print("RESULT: BACKUP VERIFIED (SHA-256 checksums match)")
        return 0
    print("RESULT: BACKUP NOT VERIFIED")
    for problem in result.problems:
        print(f"  ! {problem}")
    return 1


def cmd_verify(paths: AppPaths, args) -> int:
    if args.backup:
        targets = [Path(args.backup)]
    else:
        targets = list_backups(None, paths)
    if not targets:
        print("No backup found. Run 'backup' first.")
        return 1

    failed = False
    for target in targets:
        problems = verify_backup(target)
        if problems:
            failed = True
            print(f"{target}: NOT VERIFIED")
            for problem in problems:
                print(f"  ! {problem}")
        else:
            print(f"{target}: OK")
    return 1 if failed else 0


def cmd_restore(paths: AppPaths, args) -> int:
    if args.backup:
        target = Path(args.backup)
    else:
        backups = list_backups(None, paths)
        if not backups:
            print("No backup found.")
            return 1
        target = backups[0]

    problems = verify_backup(target)
    if problems:
        print(f"{target}: NOT VERIFIED - refusing to restore")
        for problem in problems:
            print(f"  ! {problem}")
        return 1

    restored = restore_machine_files(paths, target, dry_run=args.dry_run)
    mode = "WOULD RESTORE" if args.dry_run else "RESTORED"
    print(f"{mode} {len(restored)} file(s) from {target}")
    for name in restored:
        print(f"  - {name}")
    return 0


def cmd_status(paths: AppPaths, args) -> int:
    status = migration_status(paths, _git_tracked(paths))
    runtime = collect_runtime_files(paths)

    print(f"application root        : {paths.root}")
    print(f"machine files present   : {len(runtime)}")
    print(f"still tracked by Git    : {len(status['still_tracked'])}")
    for name in status["still_tracked"]:
        print(f"  ! {name}")
    print(f"backups found           : {status['backup_count']}")
    print(f"backups verified        : {status['verified_backup_count']}")
    print(
        "untracking safe         : "
        + ("YES" if status["untracking_safe"] else "NO (verified backup + untracked files required)")
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="application root (default: auto-detect)")
    sub = parser.add_subparsers(dest="command", required=True)

    backup = sub.add_parser("backup", help="copy and verify machine-specific files")
    backup.add_argument("--backup", default=None, help="destination directory")
    backup.add_argument("--reason", default="manual", help="label recorded in the manifest")
    backup.set_defaults(func=cmd_backup)

    verify = sub.add_parser("verify", help="re-check a backup's checksums")
    verify.add_argument("--backup", default=None)
    verify.set_defaults(func=cmd_verify)

    restore = sub.add_parser("restore", help="restore a verified backup")
    restore.add_argument("--backup", default=None)
    restore.add_argument("--dry-run", action="store_true")
    restore.set_defaults(func=cmd_restore)

    status = sub.add_parser("status", help="show migration status")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    paths = AppPaths(root=Path(args.root).resolve()) if args.root else AppPaths.discover(REPO_ROOT)
    return args.func(paths, args)


if __name__ == "__main__":
    raise SystemExit(main())
