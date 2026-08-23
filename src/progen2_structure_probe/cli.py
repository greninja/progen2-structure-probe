"""Command-line entry point for local validation and remote runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import write_json_atomic
from .config import load_config, resolved_config_record
from .provenance import git_revision, runtime_record, sha256_tree


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="progen2-probe")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate-config")
    validate.add_argument("config", type=Path)

    experiment1 = subcommands.add_parser("experiment1")
    experiment1.add_argument("config", type=Path)
    experiment1.add_argument("--progen-repo", type=Path, required=True)
    experiment1.add_argument("--progen-checkpoint", type=Path, required=True)
    experiment1.add_argument("--esm-repo", type=Path, required=True)
    experiment1.add_argument("--output-dir", type=Path)

    experiment2 = subcommands.add_parser("experiment2")
    experiment2.add_argument("config", type=Path)
    experiment2.add_argument("--progen-repo", type=Path, required=True)
    experiment2.add_argument("--progen-checkpoint", type=Path, required=True)
    experiment2.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate-config":
        print(json.dumps(resolved_config_record(args.config), indent=2, sort_keys=True))
        return

    output_dir = args.output_dir or Path(config["run"]["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "resolved_config.json", resolved_config_record(args.config))

    expected_progen_commit = config["model"]["progen2"]["upstream_commit"]
    actual_progen_commit = git_revision(args.progen_repo)
    if actual_progen_commit != expected_progen_commit:
        raise ValueError(
            f"ProGen source commit {actual_progen_commit} does not match {expected_progen_commit}"
        )
    provenance = {
        "runtime": runtime_record(),
        "progen_upstream_path": str(args.progen_repo.resolve()),
        "progen_upstream_commit": actual_progen_commit,
        "progen_checkpoint_path": str(args.progen_checkpoint.resolve()),
        "progen_checkpoint_tree_sha256": sha256_tree(args.progen_checkpoint),
    }
    if args.command == "experiment1":
        expected_esm_commit = config["model"]["esm2"]["upstream_commit"]
        actual_esm_commit = git_revision(args.esm_repo)
        if actual_esm_commit != expected_esm_commit:
            raise ValueError(
                f"ESM source commit {actual_esm_commit} does not match {expected_esm_commit}"
            )
        provenance.update(
            esm_upstream_path=str(args.esm_repo.resolve()),
            esm_upstream_commit=actual_esm_commit,
        )
    write_json_atomic(output_dir / "provenance.json", provenance)

    from .models.progen2 import OfficialProGen2

    progen = OfficialProGen2(
        args.progen_repo,
        args.progen_checkpoint,
        device=config["run"]["device"],
        fp16=config["run"]["precision"] == "float16",
    )
    if args.command == "experiment1":
        from .experiment1 import run_experiment1
        from .models.esm2 import OfficialESM2

        esm = OfficialESM2(args.esm_repo, device=config["run"]["device"])
        run_experiment1(config, progen, esm, output_dir)
    elif args.command == "experiment2":
        from .experiment2 import run_experiment2

        run_experiment2(config, progen, output_dir)


if __name__ == "__main__":
    main()
