"""Command-line entry point for local validation and remote runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import canonical_json_sha256, sha256_file, write_json_atomic
from .config import load_config, resolved_config_record
from .provenance import git_revision, runtime_record, sha256_tree


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="progen2-probe")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate-config")
    validate.add_argument("config", type=Path)

    smoke = subcommands.add_parser("smoke-models")
    smoke.add_argument("config", type=Path)
    smoke.add_argument("--progen-repo", type=Path, required=True)
    smoke.add_argument("--progen-checkpoint", type=Path, required=True)
    smoke.add_argument("--esm-repo", type=Path, required=True)
    smoke.add_argument("--sequence-length", type=int, default=128)
    smoke.add_argument("--output-dir", type=Path, default=Path("results/smoke"))

    cohort = subcommands.add_parser("build-cohort")
    cohort.add_argument("config", type=Path)
    cohort.add_argument("--work-dir", type=Path, required=True)
    cohort.add_argument("--threads", type=int, default=8)

    experiment1 = subcommands.add_parser("experiment1")
    experiment1.add_argument("config", type=Path)
    experiment1.add_argument("--progen-repo", type=Path, required=True)
    experiment1.add_argument("--progen-checkpoint", type=Path, required=True)
    experiment1.add_argument("--esm-repo", type=Path, required=True)
    experiment1.add_argument("--output-dir", type=Path)
    experiment1.add_argument("--manifest", type=Path)

    hidden_extract = subcommands.add_parser("hidden-extract")
    hidden_extract.add_argument("config", type=Path)
    hidden_extract.add_argument("--progen-repo", type=Path, required=True)
    hidden_extract.add_argument("--progen-checkpoint", type=Path, required=True)
    hidden_extract.add_argument("--output-dir", type=Path)
    hidden_extract.add_argument("--manifest", type=Path)

    hidden_probe = subcommands.add_parser("hidden-probe")
    hidden_probe.add_argument("config", type=Path)
    hidden_probe.add_argument("--representations", type=Path)
    hidden_probe.add_argument("--output-dir", type=Path)

    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate-config":
        print(json.dumps(resolved_config_record(args.config), indent=2, sort_keys=True))
        return

    if args.command == "build-cohort":
        from .cohort import build_cohort

        args.work_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            args.work_dir / "resolved_config.json", resolved_config_record(args.config)
        )
        write_json_atomic(
            args.work_dir / "provenance.json",
            {
                "runtime": runtime_record(),
                "project_commit": git_revision(Path(__file__).resolve().parent),
            },
        )
        print(json.dumps(build_cohort(config, args.work_dir, args.threads), indent=2))
        return

    config_record = resolved_config_record(args.config)
    if args.command in {"experiment1", "hidden-extract"} and args.manifest is not None:
        config["run"]["manifest"] = str(args.manifest.resolve())
        config_record["config"] = config
        config_record["resolved_sha256"] = canonical_json_sha256(config)
        config_record["overrides"] = {"run.manifest": str(args.manifest.resolve())}

    output_dir = args.output_dir or Path(config["run"]["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "resolved_config.json", config_record)

    if args.command == "hidden-probe":
        from .hidden_probe import run_hidden_probe

        representation_index = args.representations or (
            output_dir / "representations" / "index.json"
        )
        write_json_atomic(
            output_dir / "probe_provenance.json",
            {
                "runtime": runtime_record(),
                "project_commit": git_revision(Path(__file__).resolve().parent),
                "representation_index_path": str(representation_index.resolve()),
                "representation_index_sha256": sha256_file(representation_index),
            },
        )
        run_hidden_probe(config, representation_index, output_dir)
        return

    expected_progen_commit = config["model"]["progen2"]["upstream_commit"]
    actual_progen_commit = git_revision(args.progen_repo)
    if actual_progen_commit != expected_progen_commit:
        raise ValueError(
            f"ProGen source commit {actual_progen_commit} does not match {expected_progen_commit}"
        )
    provenance = {
        "runtime": runtime_record(),
        "project_commit": git_revision(Path(__file__).resolve().parent),
        "progen_upstream_path": str(args.progen_repo.resolve()),
        "progen_upstream_commit": actual_progen_commit,
        "progen_checkpoint_path": str(args.progen_checkpoint.resolve()),
        "progen_checkpoint_tree_sha256": sha256_tree(args.progen_checkpoint),
    }
    write_json_atomic(output_dir / "provenance.json", provenance)

    from .models.progen2 import OfficialProGen2

    progen = OfficialProGen2(
        args.progen_repo,
        args.progen_checkpoint,
        device=config["run"]["device"],
        fp16=config["run"]["precision"] == "float16",
    )
    if args.command == "hidden-extract":
        from .hidden_probe import extract_hidden_representations

        extract_hidden_representations(config, progen, output_dir)
        return

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

    from .models.esm2 import OfficialESM2

    esm = OfficialESM2(args.esm_repo, device=config["run"]["device"])
    if args.command == "experiment1":
        from .experiment1 import run_experiment1

        run_experiment1(config, progen, esm, output_dir)
    else:
        from .smoke import run_model_smoke

        run_model_smoke(progen, esm, output_dir, args.sequence_length)


if __name__ == "__main__":
    main()
