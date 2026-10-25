"""LLM tool/function-calling surface. Numbers come only from these tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coconet_assistant.ensemble import run_ensemble
from coconet_assistant.options import (
    SIMULATION_TIMING_FIELDS,
    list_options,
    normalise_summary_region,
)
from coconet_assistant.plot import plot_results
from coconet_assistant.scenario import (
    ScenarioSpec,
    ScenarioValidationError,
    baseline_config,
    build_scenario,
    spec_from_config,
    validate_scenario,
)
from coconet_assistant.summarise import compare_summaries, summarise_output

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_options",
            "description": (
                "List valid CoCoNet climate scenarios (SSP numbers), regions, and "
                "intervention types with their parameters. Call this before "
                "inventing any parameter name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "detail": {
                        "type": "string",
                        "enum": ["summary", "fields"],
                        "description": "summary (default) or fields for every raw config field.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_scenario",
            "description": (
                "Translate a structured management request into a CoconetConfig "
                "field dict. Intervention types must be from list_options. "
                "Does not run the model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "climate": {
                        "type": "string",
                        "description": "e.g. SSP2-4.5 or 4.5. Mapped onto config.SSP.",
                    },
                    "ssp": {"type": "number"},
                    "intervention_start_year": {
                        "type": "integer",
                        "description": (
                            "Year interventions begin, used when an intervention has no "
                            "start_year of its own. The simulation start year is fixed."
                        ),
                    },
                    "end_year": {"type": "integer"},
                    "ensemble_runs": {
                        "type": "integer",
                        "description": "Stochastic replicates. Minimum 2; default 20.",
                    },
                    "region": {
                        "type": "string",
                        "description": "GBR, FN, N, C, S, or sector.",
                    },
                    "interventions": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "Objects with a 'type' from list_options plus that type's params, "
                            'e.g. {"type": "coral_seeding", "start_year": 2030, "reefs": 50}.'
                        ),
                    },
                    "config_overrides": {
                        "type": "object",
                        "description": "Raw CoconetConfig field names only.",
                    },
                    "apply_region_bounds": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_scenario",
            "description": (
                "Validate a scenario_id or a raw CoconetConfig dict. "
                "Surface every error. Must succeed before run_ensemble."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string"},
                    "config": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ensemble",
            "description": (
                "Run a scenario as an ensemble via the CoCoNet Python API. Only works "
                "after the manager has clicked Confirm in the app; otherwise it is "
                "refused. Never request ensemble_runs=1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string"},
                    "include_baseline": {
                        "type": "boolean",
                        "description": "Also run a no-intervention pair with the same climate/years.",
                    },
                },
                "required": ["scenario_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise_results",
            "description": (
                "Summarise a completed run (mean and 10-90% ensemble bands). "
                "Returns facts you may paraphrase. Do not invent extra numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "region": {"type": "string"},
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_scenarios",
            "description": (
                "Compare two or more completed runs side by side and suggest "
                "next scenarios. All numbers come from the stored summaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_ids": {"type": "array", "items": {"type": "string"}},
                    "baseline_run_id": {"type": "string"},
                    "region": {
                        "type": "string",
                        "description": "FN, N, C, or S; omit for all reefs (GBR-wide).",
                    },
                },
                "required": ["run_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot",
            "description": (
                "Build mean + uncertainty-band series (and PNG files when matplotlib "
                "is installed) for a completed run, optionally versus a baseline run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "baseline_run_id": {"type": "string"},
                    "region": {
                        "type": "string",
                        "description": "FN, N, C, or S; omit for all reefs (GBR-wide).",
                    },
                    "groups": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "coral_total, coral_groups, cots, fish, other",
                    },
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class AssistantStore:
    """In-memory registry of scenarios and completed runs for one chat session."""

    cache_dir: Path
    reefs_file: str | None = None
    coastline_file: str | None = None
    scenarios: dict[str, ScenarioSpec] = field(default_factory=dict)
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Keyed by (run_id, region code or None for GBR-wide) so a regional summary is
    # never reused as a GBR-wide one.
    summaries: dict[tuple[str, str | None], dict[str, Any]] = field(default_factory=dict)
    last_validated_id: str | None = None
    last_progress: dict[str, Any] | None = None
    confirmed_ids: set[str] = field(default_factory=set)
    progress_hook: Any | None = None

    def remember_scenario(self, spec: ScenarioSpec) -> ScenarioSpec:
        if self.reefs_file:
            spec.config["reefs_file"] = self.reefs_file
        if self.coastline_file:
            spec.config["coastline_file"] = self.coastline_file
        self.scenarios[spec.scenario_id] = spec
        return spec

    def require_scenario(self, scenario_id: str) -> ScenarioSpec:
        if scenario_id not in self.scenarios:
            raise ScenarioValidationError([f"Unknown scenario_id {scenario_id!r}."])
        return self.scenarios[scenario_id]

    def mark_confirmed(self, scenario_id: str) -> None:
        self.confirmed_ids.add(scenario_id)

    def record_progress(self, payload: dict[str, Any]) -> None:
        self.last_progress = payload
        if self.progress_hook is not None:
            self.progress_hook(payload)


class ToolDispatcher:
    """Execute named tools against an ``AssistantStore``."""

    def __init__(
        self,
        store: AssistantStore,
        *,
        runner: Any | None = None,
    ) -> None:
        self.store = store
        self.runner = runner

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        handlers = {
            "list_options": self._list_options,
            "build_scenario": self._build_scenario,
            "validate_scenario": self._validate_scenario,
            "run_ensemble": self._run_ensemble,
            "summarise_results": self._summarise_results,
            "compare_scenarios": self._compare_scenarios,
            "plot": self._plot,
        }
        if name not in handlers:
            return {"ok": False, "errors": [f"Unknown tool {name!r}."]}
        try:
            return {"ok": True, "tool": name, "result": handlers[name](args)}
        except ScenarioValidationError as exc:
            return {"ok": False, "tool": name, "errors": exc.errors}
        except Exception as exc:
            return {"ok": False, "tool": name, "errors": [str(exc)]}

    def _list_options(self, args: dict[str, Any]) -> dict[str, Any]:
        return list_options(detail=str(args.get("detail") or "summary"))

    def _build_scenario(self, args: dict[str, Any]) -> dict[str, Any]:
        ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
        ## ~ Keep simulation timing out of the LLM's hands ~ ##
        ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
        # Language models read "from 2027" as the simulation start year, which breaks
        # spin-up and save_year. A top-level start year is treated as the default
        # intervention start; the simulation timing stays at the model defaults.
        notes: list[str] = []
        default_start = args.get("intervention_start_year", args.get("start_year"))
        interventions = []
        for item in args.get("interventions") or []:
            entry = dict(item) if isinstance(item, dict) else item
            if isinstance(entry, dict) and default_start is not None and "start_year" not in entry:
                entry["start_year"] = default_start
            interventions.append(entry)
        if args.get("projection_year") is not None:
            notes.append("projection_year is fixed by the app and was ignored.")
        overrides = dict(args.get("config_overrides") or {})
        for name in SIMULATION_TIMING_FIELDS:
            if name in overrides:
                overrides.pop(name)
                notes.append(f"{name} is fixed by the app and was ignored.")

        spec = build_scenario(
            name=args.get("name"),
            description=args.get("description") or "",
            climate=args.get("climate"),
            ssp=args.get("ssp"),
            end_year=args.get("end_year"),
            ensemble_runs=args.get("ensemble_runs"),
            region=args.get("region"),
            interventions=interventions,
            config_overrides=overrides or None,
            apply_region_bounds=args.get("apply_region_bounds", True),
        )
        spec.notes.extend(notes)
        spec = self.store.remember_scenario(spec)
        # build_scenario already validated the config, so the app can show the
        # confirmation card straight away.
        self.store.last_validated_id = spec.scenario_id
        return spec.to_dict()

    def _validate_scenario(self, args: dict[str, Any]) -> dict[str, Any]:
        spec: ScenarioSpec
        if args.get("scenario_id"):
            spec = self.store.require_scenario(str(args["scenario_id"]))
            validate_scenario(spec.config)
        elif args.get("config"):
            spec = spec_from_config(dict(args["config"]))
            spec = self.store.remember_scenario(spec)
        else:
            raise ScenarioValidationError(["Provide scenario_id or config."])
        self.store.last_validated_id = spec.scenario_id
        return {
            "scenario_id": spec.scenario_id,
            "valid": True,
            "name": spec.name,
            "display": spec.to_dict()["display"],
            "notes": spec.notes,
            "message": (
                "Settings are valid. Show them to the manager and wait for "
                "confirmation before calling run_ensemble."
            ),
        }

    def _run_ensemble(self, args: dict[str, Any]) -> dict[str, Any]:
        spec = self.store.require_scenario(str(args.get("scenario_id", "")))
        # Only a human confirmation (``mark_confirmed`` from the app's button or a typed
        # "yes") counts. A ``confirmed`` flag sent by the language model is ignored.
        confirmed = spec.scenario_id in self.store.confirmed_ids
        include_baseline = bool(args.get("include_baseline", True))
        primary = run_ensemble(
            spec,
            cache_dir=self.store.cache_dir,
            confirmed=confirmed,
            run_name=spec.name,
            progress_callback=self.store.record_progress,
            runner=self.runner,
        )
        self.store.runs[primary["run_id"]] = primary
        payload: dict[str, Any] = {"run": primary}
        if include_baseline:
            base_fields = baseline_config(spec.config)
            base_spec = spec_from_config(
                base_fields,
                name=f"{spec.name} baseline",
                description="No-intervention pair with the same climate, years, and ensemble size.",
            )
            base_spec = self.store.remember_scenario(base_spec)
            self.store.mark_confirmed(base_spec.scenario_id)
            baseline = run_ensemble(
                base_spec,
                cache_dir=self.store.cache_dir,
                confirmed=True,
                run_name=base_spec.name,
                progress_callback=self.store.record_progress,
                runner=self.runner,
            )
            self.store.runs[baseline["run_id"]] = baseline
            payload["baseline_run"] = baseline
        return payload

    def _require_run(self, run_id: str) -> dict[str, Any]:
        if run_id not in self.store.runs:
            raise ScenarioValidationError([f"Unknown run_id {run_id!r}."])
        return self.store.runs[run_id]

    def _summary(self, run_id: str, region: str | None) -> dict[str, Any]:
        """Summary for one run and region, computed once per (run, region) pair."""
        run = self._require_run(run_id)
        code = normalise_summary_region(region)
        key = (run["run_id"], code)
        if key not in self.store.summaries:
            self.store.summaries[key] = summarise_output(
                run["output_file"],
                region=code,
                scenario_name=run.get("name"),
                scenario_config=run.get("config"),
            )
        return self.store.summaries[key]

    def _summarise_results(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._summary(str(args["run_id"]), args.get("region"))

    def _compare_scenarios(self, args: dict[str, Any]) -> dict[str, Any]:
        run_ids = [str(x) for x in args.get("run_ids") or []]
        if len(run_ids) < 2:
            raise ScenarioValidationError(["compare_scenarios needs at least two run_ids."])
        region = args.get("region")
        summaries: dict[str, dict[str, Any]] = {}
        for run_id in run_ids:
            summaries[self._require_run(run_id)["name"]] = self._summary(run_id, region)
        baseline_name = None
        if args.get("baseline_run_id"):
            baseline_name = self._require_run(str(args["baseline_run_id"])).get("name")
        return compare_summaries(summaries, baseline_name=baseline_name)

    def _plot(self, args: dict[str, Any]) -> dict[str, Any]:
        run = self._require_run(str(args["run_id"]))
        region = normalise_summary_region(args.get("region"))
        summary = self._summary(run["run_id"], region)
        baseline = None
        if args.get("baseline_run_id"):
            baseline = self._summary(str(args["baseline_run_id"]), region)
        plot_dir = self.store.cache_dir / "plots" / run["run_id"]
        return plot_results(
            summary,
            baseline=baseline,
            output_dir=plot_dir,
            groups=args.get("groups"),
            stem=f"{run['run_id']}_{region or 'GBR'}",
        )
