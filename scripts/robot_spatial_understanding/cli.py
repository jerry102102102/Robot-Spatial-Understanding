"""Public CLI for legacy robot-model tools and simulation evidence workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .action_bridge import write_action_evidence_source
from .adapters import adapter_for, available_adapters
from .benchmark import BenchmarkSuite
from .corruption import CORRUPTION_KINDS, corrupt_run
from .counterfactual import CounterfactualAssurance
from .errors import RobotSpatialUnderstandingError
from .report import AssuranceReport
from .simulation import SimulationRun
from .task import TaskSpec


NEW_COMMANDS = frozenset(
    {
        "import",
        "capture",
        "evaluate",
        "explain",
        "benchmark",
        "corrupt",
        "inspect-run",
        "list-adapters",
        "action-evidence",
        "counterfactual",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-spatial",
        description=(
            "Evidence-grounded robot model and simulation-result understanding. "
            "Legacy model commands such as validate/export/transform are forwarded unchanged."
        ),
    )
    parser.add_argument("--version", action="version", version=f"robot-spatial {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    list_adapters = subparsers.add_parser("list-adapters", help="list built-in and installed simulator adapters")
    list_adapters.set_defaults(handler=_list_adapters)

    importer = subparsers.add_parser("import", help="normalize one offline simulator export into simulation-run.v1")
    importer.add_argument("--adapter", default="generic-json")
    importer.add_argument("source", type=Path)
    importer.add_argument("--out", type=Path, required=True)
    importer.set_defaults(handler=_import)

    capture = subparsers.add_parser(
        "capture",
        help="capture a supported live simulator or normalize an immutable adapter export",
    )
    capture.add_argument("--adapter", required=True)
    capture.add_argument("--source", type=Path, help="immutable raw state export without reward/success labels")
    capture.add_argument("--env-id", help="live environment ID (ManiSkill or Gymnasium Robotics)")
    capture.add_argument("--seed", type=int, default=2)
    capture.add_argument("--max-steps", type=int, default=50)
    capture.add_argument("--fixed-horizon", type=int, help="fixed live-simulator control horizon")
    capture.add_argument("--controller-gain", type=float, default=10.0)
    capture.add_argument("--trajectory", type=Path, help="action-only HDF5 trajectory for live simulator replay")
    capture.add_argument("--trajectory-index", type=int, default=0)
    capture.add_argument("--entity-map", type=Path, help="simulator-to-evidence entity mapping")
    capture.add_argument("--sim-backend", choices=["physx_cpu", "physx_cuda"], default="physx_cpu")
    capture.add_argument("--render-backend", default="gpu")
    capture.add_argument("--num-envs", type=int, default=1)
    capture.add_argument("--initialization", choices=["goal_at_cube"])
    capture.add_argument("--out", type=Path, required=True)
    capture.set_defaults(handler=_capture)

    evaluate = subparsers.add_parser("evaluate", help="evaluate task predicates and write a layered assurance report")
    evaluate.add_argument("run", type=Path)
    evaluate.add_argument("--task", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate)

    explain = subparsers.add_parser("explain", help="render a verified assurance report as evidence-linked Markdown")
    explain.add_argument("report", type=Path)
    explain.add_argument("--out", type=Path, required=True)
    explain.set_defaults(handler=_explain)

    benchmark = subparsers.add_parser("benchmark", help="predict all cases before revealing isolated reference results")
    benchmark.add_argument("--suite", type=Path, required=True)
    benchmark.add_argument("--out", type=Path, required=True)
    benchmark.set_defaults(handler=_benchmark)

    corrupt = subparsers.add_parser("corrupt", help="create a deterministic negative-control run")
    corrupt.add_argument("run", type=Path)
    corrupt.add_argument("--kind", choices=sorted(CORRUPTION_KINDS), required=True)
    corrupt.add_argument("--channel", default="pose")
    corrupt.add_argument("--out", type=Path, required=True)
    corrupt.set_defaults(handler=_corrupt)

    inspect = subparsers.add_parser("inspect-run", help="verify a run and summarize channels, completeness, and boundaries")
    inspect.add_argument("run", type=Path)
    inspect.set_defaults(handler=_inspect_run)

    action_evidence = subparsers.add_parser(
        "action-evidence",
        help="map verified simulation predicates into the existing action-assurance evidence-source contract",
    )
    action_evidence.add_argument("report", type=Path)
    action_evidence.add_argument("--mapping", type=Path, required=True)
    action_evidence.add_argument("--out", type=Path, required=True)
    action_evidence.set_defaults(handler=_action_evidence)

    counterfactual = subparsers.add_parser(
        "counterfactual",
        help="compare matched action and no-op/perturbation replays for simulation-bounded contribution evidence",
    )
    counterfactual.add_argument("--action-run", type=Path, required=True)
    counterfactual.add_argument("--control-run", type=Path, required=True)
    counterfactual.add_argument("--task", type=Path, required=True)
    counterfactual.add_argument("--out", type=Path, required=True)
    counterfactual.set_defaults(handler=_counterfactual)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def _list_adapters(_: argparse.Namespace) -> int:
    _print({"adapters": available_adapters()})
    return 0


def _import(args: argparse.Namespace) -> int:
    run = adapter_for(args.adapter).import_source(args.source, args.out)
    _print(
        {
            "status": "imported",
            "run": str(run.root.resolve()),
            "run_id": run.manifest["run_id"],
            "manifest_sha256": run.digest,
            "completeness": run.completeness["status"],
        }
    )
    return 0


def _capture(args: argparse.Namespace) -> int:
    adapter = adapter_for(args.adapter)
    if args.env_id:
        capture_episode = getattr(adapter, "capture_episode", None)
        capture_goal_episode = getattr(adapter, "capture_goal_episode", None)
        if capture_episode is not None:
            if args.trajectory is None or args.entity_map is None:
                raise RobotSpatialUnderstandingError(
                    f"adapter {args.adapter!r} live capture requires --trajectory and --entity-map"
                )
            captured = capture_episode(
                args.out,
                env_id=args.env_id,
                seed=args.seed,
                trajectory=args.trajectory,
                entity_map=args.entity_map,
                sim_backend=args.sim_backend,
                render_backend=args.render_backend,
                num_envs=args.num_envs,
                fixed_horizon=args.fixed_horizon or 100,
                trajectory_index=args.trajectory_index,
                initialization=args.initialization,
            )
        elif capture_goal_episode is not None:
            captured = capture_goal_episode(
                args.out,
                env_id=args.env_id,
                seed=args.seed,
                max_steps=args.fixed_horizon or args.max_steps,
                controller_gain=args.controller_gain,
            )
        else:
            raise RobotSpatialUnderstandingError(
                f"adapter {args.adapter!r} does not implement live capture; provide --source instead"
            )
    elif args.source:
        captured = adapter.import_source(args.source, args.out)
    else:
        raise RobotSpatialUnderstandingError("capture requires either --source or --env-id")
    runs = captured if isinstance(captured, list) else [captured]
    _print(
        {
            "status": "captured_live" if args.env_id else "captured_from_immutable_export",
            "run_count": len(runs),
            "runs": [
                {
                    "run": str(run.root.resolve()),
                    "run_id": run.manifest["run_id"],
                    "adapter": run.manifest["adapter"],
                    "manifest_sha256": run.digest,
                    "completeness": run.completeness["status"],
                }
                for run in runs
            ],
        }
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    run = SimulationRun.load(args.run)
    task = TaskSpec.load(args.task)
    report = AssuranceReport.evaluate(run, task)
    path = report.write(args.out)
    _print(
        {
            "status": "evaluated",
            "report": str(path.resolve()),
            "report_sha256": report.digest,
            "verdict": report.data["verdict"],
        }
    )
    return 0


def _explain(args: argparse.Namespace) -> int:
    report = AssuranceReport.load(args.report)
    path = report.explain_to(args.out)
    _print({"status": "explained", "out": str(path.resolve()), "report_sha256": report.digest})
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    result = BenchmarkSuite.load(args.suite).run(args.out)
    _print(
        {
            "status": "scored",
            "report": str((args.out / "benchmark-report.json").resolve()),
            "benchmark_report_sha256": result["benchmark_report_sha256"],
            "episode_metrics": result["episode_metrics"],
        }
    )
    return 0


def _corrupt(args: argparse.Namespace) -> int:
    output = corrupt_run(args.run, args.out, kind=args.kind, channel=args.channel)
    _print({"status": "corrupted_control_created", "kind": args.kind, "out": str(output.resolve())})
    return 0


def _inspect_run(args: argparse.Namespace) -> int:
    run = SimulationRun.load(args.run)
    _print(
        {
            "schema_version": run.manifest["schema_version"],
            "run_id": run.manifest["run_id"],
            "manifest_sha256": run.digest,
            "simulator": run.manifest["simulator"],
            "adapter": run.manifest["adapter"],
            "interval": run.manifest["interval"],
            "channels": {
                name: {
                    "availability": channel["status"],
                    "completeness": run.completeness["channels"][name]["status"],
                }
                for name, channel in run.manifest["channels"].items()
            },
            "boundaries": run.manifest["boundaries"],
        }
    )
    return 0


def _action_evidence(args: argparse.Namespace) -> int:
    source = write_action_evidence_source(args.report, args.mapping, args.out)
    _print(
        {
            "status": "action_evidence_written",
            "out": str(args.out.resolve()),
            "source_id": source["source_id"],
            "record_count": len(source["records"]),
        }
    )
    return 0


def _counterfactual(args: argparse.Namespace) -> int:
    comparison = CounterfactualAssurance.compare(
        SimulationRun.load(args.action_run),
        SimulationRun.load(args.control_run),
        TaskSpec.load(args.task),
    )
    path = comparison.write(args.out)
    _print(
        {
            "status": "counterfactual_compared",
            "out": str(path.resolve()),
            "causal_contribution": comparison.data["causal_contribution"],
            "counterfactual_sha256": comparison.data["counterfactual_sha256"],
        }
    )
    return 0


def _legacy(argv: Sequence[str]) -> int:
    import robot_spatial

    original = sys.argv
    try:
        sys.argv = ["robot-spatial", *argv]
        return int(robot_spatial.main())
    finally:
        sys.argv = original


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in NEW_COMMANDS and arguments[0] not in {"-h", "--help", "--version"}:
        return _legacy(arguments)
    parser = build_parser()
    if not arguments:
        parser.print_help()
        return 0
    try:
        parsed = parser.parse_args(arguments)
        if not hasattr(parsed, "handler"):
            parser.print_help()
            return 0
        return int(parsed.handler(parsed))
    except (OSError, RobotSpatialUnderstandingError) as error:
        _print({"status": "error", "error": str(error)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
