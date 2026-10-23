from __future__ import annotations

import logging
import math
import multiprocessing as mp
import shutil
import tempfile
import time
from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import threadpoolctl

from coconet.config import (
    CoconetConfig,
    effective_ensemble_workers,
    use_parallel_ensemble_run,
)
from coconet.logging_utils import configure_logging
from coconet.netlogo import NetLogoRng, heading_from_dx_dy, nl_ceiling, nl_median, nl_round

CORAL_GROUPS = ("sa", "ta", "mo", "po", "fa", "tt")
# Coral larval kernel inner loop order in spawn (legacy NetLogo). RNG must draw
# uniforms in this order; do not reorder or merge with CORAL_GROUPS.
CORAL_SPAWN_KERNEL_ORDER = ("sa", "tt", "ta", "mo", "po", "fa")
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReefKernel:
    con1: float
    dir1: float
    ang1: float
    dis1: float
    con2: float
    dir2: float
    ang2: float
    dis2: float


@dataclass(slots=True)
class SpinupCheckpoint:
    """Immutable snapshot of site/reef initial conditions after ensemble-0 spinup."""

    E_i: dict[str, np.ndarray]
    G_i: dict[str, np.ndarray]
    S_i: dict[str, np.ndarray]
    B_i: np.ndarray
    T_i: np.ndarray
    C_i: dict[str, np.ndarray]
    C_site_i: np.ndarray
    R_site_i: np.ndarray


class CoconetModel:
    """Python port of legacy CoCoNet NetLogo model."""

    def __init__(self, config: CoconetConfig) -> None:
        self.cfg = config
        self.rng = NetLogoRng(1)

        # Globals
        self.small = 0.000001
        self.year = config.start_year
        self.ensemble = 0
        self.per_km = (2 * 500) / ((25.5 - 10.5) * 111)
        self.ha_per_site = 8
        self.draw_year = 15
        self.draw_month = 10
        self.draw_fortnight = 1
        self.search_mode = int(config.search_mode)
        self.control_region = "GBR"
        self.S_vessels = 0
        self.catchment_condition = 0.0
        self.flood_load = 0.2
        self.cyclone_centre = -1
        self.cyclone_radius = 0.0
        self.cyclone_category = 0.0
        self.number_of_reefs = 0
        self.number_of_sites = 0
        self.output_file = Path(config.output_file)

        # Fixed parameter defaults from setup
        self.adaptability = 1.0
        self.adapt_decay_time = 10.0
        self.adapt_penalty = 0.05
        self.adapt_plasticity = 8.0
        self.rubble_decay_time = 6.0
        self.flood_scale = 50.0
        self.k_sa = 0.0025
        self.k_ta = 0.0020
        self.k_mo = 0.0005
        self.k_po = 0.0010
        self.k_fa = 0.0020
        self.k_tt = 0.0010
        self.pH_scale = 100.0
        self.dhw_scale = 50.0

        self.B_max = 10000.0
        self.B_recruit = 0.5
        self.B_pred_S1 = 500.0
        self.T_max = 200.0
        self.T_recruit = 0.8
        self.T_pred_B = 120.0
        self.E_recruit = 0.005
        self.E_mort = 0.025
        self.E_natal = 0.1
        self.E_pred_S1 = 500.0
        self.E_pred_S = 50.0
        self.G_recruit = 7.0
        self.G_mort = 0.01
        self.G_pred_T = 70.0
        self.reporting_ratio = 0.5

        self.rate_i = {
            "sa": 0.50,
            "ta": 0.40,
            "mo": 0.30,
            "po": 0.15,
            "fa": 0.10,
            "tt": 0.40,
        }
        self.thermal_i = {
            "sa": 1.5,
            "ta": 2.0,
            "mo": 3.0,
            "po": 3.5,
            "fa": 3.5,
            "tt": 7.5,
        }

        self.C_recruit = 0.08
        self.C_allee = 0.02
        self.S_spawning_threshold = 3.0
        self.S_allee = 3.0
        self.S_spawning_failure = 0.5
        self.S_phase = 1.0
        self.S_recruit = 200000.0
        self.S_pred_C = 0.0003
        self.S_prefer = 0.4
        self.S_mort = 0.8
        self.S1_mort = self.S_mort
        self.S2_mort = 0.0
        self.S3_mort = 0.0
        self.S4_mort = 0.0
        self.S5_mort = 0.0
        self.S6_mort = self.S_mort

        self.monitor_total_outbreaks = 0
        self.monitor_active_outbreaks = 0

        # Loaded data placeholders
        self.reef_df: pd.DataFrame | None = None
        self.reef_numeric: np.ndarray | None = None
        self.coast_x: np.ndarray | None = None
        self.coast_y: np.ndarray | None = None

        self.reef_who: np.ndarray | None = None
        self.who_to_reef: dict[int, int] = {}
        self.site_who: np.ndarray | None = None
        self.site_reef: np.ndarray | None = None
        self.site_offsets: np.ndarray | None = None

        self.dist_matrix: np.ndarray | None = None
        self.heading_matrix: np.ndarray | None = None

        # Reef properties / states
        self.reef_id: np.ndarray | None = None
        self.region_name: np.ndarray | None = None
        self.shelf_position: np.ndarray | None = None
        self.sector_number: np.ndarray | None = None
        self.rezone_year: np.ndarray | None = None
        self.future_rezone_year: np.ndarray | None = None
        self.priority_category: np.ndarray | None = None
        self.priority: np.ndarray | None = None
        self.reef_sites: np.ndarray | None = None
        self.benefit: np.ndarray | None = None
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.xcor: np.ndarray | None = None
        self.ycor: np.ndarray | None = None
        self.km_offshore: np.ndarray | None = None

        self.E = {k: None for k in ("0", "1", "2", "3", "4", "5")}
        self.E_i = {k: None for k in ("0", "1", "2", "3", "4", "5")}
        self.G = {k: None for k in ("0", "1", "2", "3", "4", "5")}
        self.G_i = {k: None for k in ("0", "1", "2", "3", "4", "5")}
        self.E_catch_kg: np.ndarray | None = None
        self.G_catch_kg: np.ndarray | None = None

        self.B_r: np.ndarray | None = None
        self.T_r: np.ndarray | None = None
        self.S_r = {k: None for k in ("1", "2", "3", "4", "5", "6")}
        self.S_manta_r: np.ndarray | None = None
        self.C_r = {g: None for g in CORAL_GROUPS}
        self.C_reef: np.ndarray | None = None
        self.R_reef: np.ndarray | None = None
        self.C_out_degree: np.ndarray | None = None
        self.dhw: np.ndarray | None = None
        self.reef_shading: np.ndarray | None = None
        self.regional_shading: np.ndarray | None = None
        self.pH_protect: np.ndarray | None = None
        self.dives_reef: np.ndarray | None = None

        self.bleach_mort_r = {g: None for g in CORAL_GROUPS}
        self.cyclone_mort_r = {g: None for g in CORAL_GROUPS}
        self.predate_mort_r = {g: None for g in CORAL_GROUPS}

        # Site properties / states
        self.B: np.ndarray | None = None
        self.B_i: np.ndarray | None = None
        self.T: np.ndarray | None = None
        self.T_i: np.ndarray | None = None

        self.S = {k: None for k in ("0", "1", "2", "3", "4", "5", "6")}
        self.S_i = {k: None for k in ("0", "1", "2", "3", "4", "5", "6")}
        self.S_manta: np.ndarray | None = None

        self.C = {g: None for g in CORAL_GROUPS}
        self.C_i = {g: None for g in CORAL_GROUPS}
        self.C_site: np.ndarray | None = None
        self.C_site_i: np.ndarray | None = None

        self.rate = {g: None for g in CORAL_GROUPS}
        self.thermal = {g: None for g in CORAL_GROUPS}
        self.bleach_mort = {g: None for g in CORAL_GROUPS}
        self.cyclone_mort = {g: None for g in CORAL_GROUPS}
        self.predate_mort = {g: None for g in CORAL_GROUPS}
        self.R_site: np.ndarray | None = None
        self.R_site_i: np.ndarray | None = None

    # ---- lifecycle ----
    def setup(self, *, init_output: bool = True) -> None:
        setup_start = time.perf_counter()
        logger.info(
            "Model setup started (reefs_file=%s coastline_file=%s output_file=%s init_output=%s)",
            self.cfg.reefs_file,
            self.cfg.coastline_file,
            self.cfg.output_file,
            init_output,
        )
        self.rng.seed(1)
        self._load_coastline()
        self._load_reefs()
        self._allocate_states()
        if init_output:
            self._set_up_output_files()
        self.ensemble = 0
        logger.info(
            "Model setup completed in %.2fs (reefs=%s sites=%s output_file=%s)",
            time.perf_counter() - setup_start,
            self.number_of_reefs,
            self.number_of_sites,
            self.output_file,
        )

    def run(self) -> None:
        run_start = time.perf_counter()
        workers = effective_ensemble_workers(self.cfg.ensemble_threads, self.cfg.ensemble_runs)
        logger.info(
            "Run started (ensemble_runs=%s ensemble_threads=%s effective_worker_cap=%s "
            "start_year=%s spinup_backtrack_years=%s end_year=%s save_year=%s "
            "projection_year=%s search_year=%s)",
            self.cfg.ensemble_runs,
            self.cfg.ensemble_threads,
            workers,
            self.cfg.start_year,
            self.cfg.spinup_backtrack_years,
            self.cfg.end_year,
            self.cfg.save_year,
            self.cfg.projection_year,
            self.cfg.search_year,
        )
        self.setup(init_output=True)
        if use_parallel_ensemble_run(self.cfg.ensemble_threads, self.cfg.ensemble_runs):
            logger.info(
                "Parallel simulation phase: process_pool_size=%s "
                "(spawn workers after ensemble-0 spinup on main process; "
                "BLAS limited to 1 thread per worker via threadpoolctl).",
                workers,
            )
            self._run_with_parallel_simulation_ensembles(run_start, workers)
        else:
            if self.cfg.ensemble_runs > 1 and workers == 1:
                logger.info(
                    "Simulation ensembles run serially on the main thread "
                    "(ensemble_threads=%s; use --ensemble-threads > 1 or 0/auto to run multiple worker processes).",
                    self.cfg.ensemble_threads,
                )
            self._run_sequential_ensemble_loop(run_start)

    def _run_sequential_ensemble_loop(self, run_start: float) -> None:
        while self.ensemble <= self.cfg.ensemble_runs:
            ensemble_start = time.perf_counter()
            ensemble_kind = "spinup" if self.ensemble == 0 else "simulation"
            logger.info(
                "Ensemble %s/%s started (%s).",
                self.ensemble,
                self.cfg.ensemble_runs,
                ensemble_kind,
            )
            self.initialise_run()
            self._run_ensemble_year_steps()
            self._maybe_write_priority_benefit()
            logger.info(
                "Ensemble %s/%s completed in %.2fs.",
                self.ensemble,
                self.cfg.ensemble_runs,
                time.perf_counter() - ensemble_start,
            )
            self.ensemble += 1
        logger.info(
            "Run completed in %.2fs (output_file=%s final_ensemble=%s).",
            time.perf_counter() - run_start,
            self.output_file,
            self.ensemble - 1,
        )

    def _run_with_parallel_simulation_ensembles(self, run_start: float, max_workers: int) -> None:
        self.ensemble = 0
        ensemble_start = time.perf_counter()
        logger.info(
            "Ensemble %s/%s started (spinup).",
            self.ensemble,
            self.cfg.ensemble_runs,
        )
        self.initialise_run()
        self._run_ensemble_year_steps()
        self._maybe_write_priority_benefit()
        logger.info(
            "Ensemble %s/%s completed in %.2fs.",
            self.ensemble,
            self.cfg.ensemble_runs,
            time.perf_counter() - ensemble_start,
        )

        checkpoint = self.export_spinup_checkpoint()
        tmpdir = Path(tempfile.mkdtemp(prefix="coconet-ensemble-"))
        try:
            futures: dict[int, Future[None]] = {}
            # spawn: fresh interpreters so we do not fork a huge post-spinup parent;
            # threads were wrong here — Python bytecode in the year loop holds the GIL.
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                for e in range(1, self.cfg.ensemble_runs + 1):
                    out_part = tmpdir / f"output_{e}.csv"
                    pri_part = tmpdir / f"priority_{e}.csv" if self.search_mode == 1 else None
                    futures[e] = executor.submit(
                        _run_simulation_ensemble_worker,
                        replace(self.cfg),
                        checkpoint,
                        e,
                        out_part,
                        pri_part,
                    )
                for e in range(1, self.cfg.ensemble_runs + 1):
                    futures[e].result()

            with self.output_file.open("ab") as out_f:
                for e in range(1, self.cfg.ensemble_runs + 1):
                    part = tmpdir / f"output_{e}.csv"
                    if part.is_file():
                        with part.open("rb") as in_f:
                            shutil.copyfileobj(in_f, out_f)

            if self.search_mode == 1:
                pri_main = Path("priority_reef_benefit.csv")
                with pri_main.open("ab") as out_f:
                    for e in range(1, self.cfg.ensemble_runs + 1):
                        part = tmpdir / f"priority_{e}.csv"
                        if part.is_file():
                            with part.open("rb") as in_f:
                                shutil.copyfileobj(in_f, out_f)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.ensemble = self.cfg.ensemble_runs + 1
        logger.info(
            "Run completed in %.2fs (output_file=%s final_ensemble=%s).",
            time.perf_counter() - run_start,
            self.output_file,
            self.ensemble - 1,
        )

    def export_spinup_checkpoint(self) -> SpinupCheckpoint:
        return SpinupCheckpoint(
            E_i={k: np.copy(v) for k, v in self.E_i.items()},
            G_i={k: np.copy(v) for k, v in self.G_i.items()},
            S_i={k: np.copy(v) for k, v in self.S_i.items()},
            B_i=np.copy(self.B_i),
            T_i=np.copy(self.T_i),
            C_i={k: np.copy(v) for k, v in self.C_i.items()},
            C_site_i=np.copy(self.C_site_i),
            R_site_i=np.copy(self.R_site_i),
        )

    def import_spinup_checkpoint(self, checkpoint: SpinupCheckpoint) -> None:
        for k, v in checkpoint.E_i.items():
            self.E_i[k][:] = v
        for k, v in checkpoint.G_i.items():
            self.G_i[k][:] = v
        for k, v in checkpoint.S_i.items():
            self.S_i[k][:] = v
        self.B_i[:] = checkpoint.B_i
        self.T_i[:] = checkpoint.T_i
        for k, v in checkpoint.C_i.items():
            self.C_i[k][:] = v
        self.C_site_i[:] = checkpoint.C_site_i
        self.R_site_i[:] = checkpoint.R_site_i

    def _maybe_write_priority_benefit(self, path: Path | None = None) -> None:
        if self.search_mode == 1 and self.year >= self.cfg.search_year:
            logger.debug(
                "Writing priority benefit output (ensemble=%s year=%s).",
                self.ensemble,
                self.year,
            )
            self._write_priority_benefit(path)

    def _run_ensemble_year_steps(self) -> None:
        while self.year <= self.cfg.end_year:
            logger.info(
                "Progress: ensemble=%s year=%s",
                self.ensemble,
                self.year,
            )
            self._seed_year()
            self.draw_year = 15 + self.rng.random_int(7)
            if self.year in (2015, 2016, 2017, 2018, 2019, 2020, 2021):
                self.draw_year = self.year - 2000
            self.draw_month = 10 + self.rng.random_int(3)
            self.draw_fortnight = 1 + self.rng.random_int(4)

            self.cyclone()
            self._seed_year()
            if self.year < self.cfg.projection_year:
                self.bleaching()
            else:
                if self.year == self.cfg.projection_year:
                    logger.debug(
                        "Entering projection bleaching mode at year=%s.",
                        self.year,
                    )
                prob = self._projection_bleach_probability()
                reduction = 0.2 * (self.cyclone_category - 0.5) / 4.5
                if (prob - reduction) > self.rng.random_float(1.0):
                    self.bleaching()

            for reef_idx in self.rng._rs.permutation(self.number_of_reefs):
                reef_idx = int(reef_idx)
                self.grow_corals(reef_idx)
                self.grow_cots(reef_idx)
                self.grow_fish(reef_idx)
                self.consume_corals(reef_idx)
                self.consume_cots(reef_idx)
                self.dives_reef[reef_idx] = 0

            self._apply_interventions()

            for reef_idx in self.rng._rs.permutation(self.number_of_reefs):
                self.spawn_corals(int(reef_idx))

            if self._cots_spawn_gate() > self.rng.random_float(2 * self.S_spawning_failure):
                for reef_idx in self.rng._rs.permutation(self.number_of_reefs):
                    self.spawn_cots(int(reef_idx))

            for reef_idx in self.rng._rs.permutation(self.number_of_reefs):
                self.spawn_fish(int(reef_idx))

            self.reef_populations()
            if self.ensemble == 0 and self.year == self.cfg.start_year:
                logger.info(
                    "Captured start conditions at year=%s during spinup.",
                    self.year,
                )
                self.save_start_conditions()
                self.year = self.cfg.end_year - 1

            if self.ensemble > 0 and self.year >= self.cfg.save_year and self.search_mode == 0:
                logger.debug(
                    "Writing annual output (ensemble=%s year=%s).",
                    self.ensemble,
                    self.year,
                )
                self.write_output()

            self.year += 1

    def _cots_spawn_gate(self) -> float:
        """CoTS spawning success probability gate (legacy line 1067).
        
        Returns probability from 0-1 based on sinusoidal cycle. NetLogo sin takes degrees.
        """
        angle = 2 * 3.1416 * (2010 + self.S_phase + self.rng.random_float(4) - self.year) / 16
        return 0.5 * (1 + math.sin(math.radians(angle)))

    # ---- setup helpers ----
    def _load_coastline(self) -> None:
        path = Path(self.cfg.coastline_file)
        df = pd.read_csv(path)
        self.coast_x = 67.0 * (pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy() - 147.6)
        self.coast_y = 67.0 * (pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy() + 18.0)

    def _load_reefs(self) -> None:
        path = Path(self.cfg.reefs_file)
        self.reef_df = pd.read_csv(path, low_memory=False)
        self.reef_numeric = self.reef_df.apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64
        )

        self.reef_id = self.reef_df.iloc[:, 1].astype(str).to_numpy()
        self.y = self.reef_numeric[:, 2]
        self.x = self.reef_numeric[:, 3]
        self.region_name = self.reef_df.iloc[:, 6].astype(str).to_numpy()
        self.shelf_position = self.reef_df.iloc[:, 8].astype(str).to_numpy()
        self.sector_number = self.reef_numeric[:, 9].astype(np.int32)
        self.rezone_year = self.reef_numeric[:, 10].astype(np.int32)
        self.priority_category = self.reef_df.iloc[:, 11].astype(str).to_numpy()
        self.reef_sites = self.reef_numeric[:, 18].astype(np.int32)
        self.number_of_reefs = self.reef_df.shape[0]
        self.number_of_sites = int(self.reef_sites.sum())
        self.xcor = 67.0 * (self.x - 147.6)
        self.ycor = 67.0 * (self.y + 18.0)

        # Reproduce NetLogo who numbering with coastline created first.
        coast_count = len(self.coast_x) + 1
        self.reef_who = np.zeros(self.number_of_reefs, dtype=np.int32)
        self.site_reef = np.zeros(self.number_of_sites, dtype=np.int32)
        self.site_who = np.zeros(self.number_of_sites, dtype=np.int32)
        self.site_offsets = np.zeros(self.number_of_reefs + 1, dtype=np.int32)

        who = coast_count
        s = 0
        for i in range(self.number_of_reefs):
            self.reef_who[i] = who
            self.who_to_reef[int(who)] = i
            who += 1
            n = int(self.reef_sites[i])
            self.site_offsets[i] = s
            self.site_reef[s : s + n] = i
            self.site_who[s : s + n] = np.arange(who, who + n, dtype=np.int32)
            s += n
            who += n
        self.site_offsets[self.number_of_reefs] = s

        # km offshore from nearest coast.
        dx = self.xcor[:, None] - self.coast_x[None, :]
        dy = self.ycor[:, None] - self.coast_y[None, :]
        self.km_offshore = np.sqrt(dx * dx + dy * dy).min(axis=1) / self.per_km

        # Pairwise distance/heading among reefs for dispersal.
        rdx = self.xcor[None, :] - self.xcor[:, None]
        rdy = self.ycor[None, :] - self.ycor[:, None]
        self.dist_matrix = np.sqrt(rdx * rdx + rdy * rdy)
        self.heading_matrix = heading_from_dx_dy(rdx, rdy)

    def _allocate_states(self) -> None:
        n_reefs = self.number_of_reefs
        n_sites = self.number_of_sites

        self.future_rezone_year = np.full(n_reefs, 9999, dtype=np.int32)
        self.priority = np.zeros(n_reefs, dtype=np.int32)
        self.benefit = np.zeros(n_reefs, dtype=np.float64)
        self.E_catch_kg = np.zeros(n_reefs, dtype=np.float64)
        self.G_catch_kg = np.zeros(n_reefs, dtype=np.float64)
        self.reef_shading = np.zeros(n_reefs, dtype=np.float64)
        self.regional_shading = np.zeros(n_reefs, dtype=np.float64)
        self.pH_protect = np.zeros(n_reefs, dtype=np.float64)
        self.dives_reef = np.zeros(n_reefs, dtype=np.float64)
        self.dhw = np.zeros(n_reefs, dtype=np.float64)
        self.C_out_degree = np.zeros(n_reefs, dtype=np.float64)

        self.B_r = np.zeros(n_reefs, dtype=np.float64)
        self.T_r = np.zeros(n_reefs, dtype=np.float64)
        self.S_manta_r = np.zeros(n_reefs, dtype=np.float64)
        self.C_reef = np.zeros(n_reefs, dtype=np.float64)
        self.R_reef = np.zeros(n_reefs, dtype=np.float64)

        for age in ("0", "1", "2", "3", "4", "5"):
            self.E[age] = np.zeros(n_reefs, dtype=np.float64)
            self.E_i[age] = np.zeros(n_reefs, dtype=np.float64)
            self.G[age] = np.zeros(n_reefs, dtype=np.float64)
            self.G_i[age] = np.zeros(n_reefs, dtype=np.float64)

        for age in ("1", "2", "3", "4", "5", "6"):
            self.S_r[age] = np.zeros(n_reefs, dtype=np.float64)

        for g in CORAL_GROUPS:
            self.C_r[g] = np.zeros(n_reefs, dtype=np.float64)
            self.bleach_mort_r[g] = np.zeros(n_reefs, dtype=np.float64)
            self.cyclone_mort_r[g] = np.zeros(n_reefs, dtype=np.float64)
            self.predate_mort_r[g] = np.zeros(n_reefs, dtype=np.float64)

        self.B = np.zeros(n_sites, dtype=np.float64)
        self.B_i = np.zeros(n_sites, dtype=np.float64)
        self.T = np.zeros(n_sites, dtype=np.float64)
        self.T_i = np.zeros(n_sites, dtype=np.float64)
        self.S_manta = np.zeros(n_sites, dtype=np.float64)

        for age in ("0", "1", "2", "3", "4", "5", "6"):
            self.S[age] = np.zeros(n_sites, dtype=np.float64)
            self.S_i[age] = np.zeros(n_sites, dtype=np.float64)

        for g in CORAL_GROUPS:
            self.C[g] = np.zeros(n_sites, dtype=np.float64)
            self.C_i[g] = np.zeros(n_sites, dtype=np.float64)
            self.rate[g] = np.zeros(n_sites, dtype=np.float64)
            self.thermal[g] = np.zeros(n_sites, dtype=np.float64)
            self.bleach_mort[g] = np.zeros(n_sites, dtype=np.float64)
            self.cyclone_mort[g] = np.zeros(n_sites, dtype=np.float64)
            self.predate_mort[g] = np.zeros(n_sites, dtype=np.float64)

        self.C_site = np.zeros(n_sites, dtype=np.float64)
        self.C_site_i = np.zeros(n_sites, dtype=np.float64)
        self.R_site = np.zeros(n_sites, dtype=np.float64)
        self.R_site_i = np.zeros(n_sites, dtype=np.float64)

    # ---- utility helpers ----
    def _seed_year(self) -> None:
        self.rng.seed((self.ensemble + 1) * self.year)

    def _sites_for_reef(self, reef_idx: int) -> slice:
        s0 = int(self.site_offsets[reef_idx])
        s1 = int(self.site_offsets[reef_idx + 1])
        return slice(s0, s1)

    def _one_site_for_reef(self, reef_idx: int) -> int | None:
        sl = self._sites_for_reef(reef_idx)
        if sl.start == sl.stop:
            return None
        return self.rng.random_int(sl.stop - sl.start) + sl.start

    def _one_global_site(self) -> int | None:
        if self.number_of_sites == 0:
            return None
        return self.rng.random_int(self.number_of_sites)

    def _kernel_base_coral(self, draw_year: int, draw_month: int) -> int:
        return 20 + (((draw_year - 15) * 3 + (draw_month - 10)) * 9)

    def _kernel_base_cots(self, draw_year: int, draw_fortnight: int) -> int:
        return 209 + (((draw_year - 15) * 4 + (draw_fortnight - 1)) * 9)

    def _kernel_base_grouper(self, draw_year: int, draw_fortnight: int) -> int:
        return 461 + (((draw_year - 15) * 4 + (draw_fortnight - 1)) * 9)

    def _reef_kernel(self, reef_idx: int, base_col: int) -> ReefKernel:
        row = self.reef_numeric[reef_idx]
        return ReefKernel(
            con1=float(row[base_col]),
            dir1=float(row[base_col + 1]),
            ang1=float(row[base_col + 2]),
            dis1=float(row[base_col + 3]),
            con2=float(row[base_col + 4]),
            dir2=float(row[base_col + 5]),
            ang2=float(row[base_col + 6]),
            dis2=float(row[base_col + 7]),
        )

    def _targets_in_cone(
        self, source: int, distance: float, direction: float, angle: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if distance <= 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        d = self.dist_matrix[source]
        h = self.heading_matrix[source]
        delta = np.abs((h - direction + 180.0) % 360.0 - 180.0)
        mask = (d <= distance) & (delta <= angle / 2.0)
        mask[source] = False
        targets = np.flatnonzero(mask)
        return targets.astype(np.int32), d[targets]

    def _targets_in_radius(self, source: int, distance: float) -> np.ndarray:
        d = self.dist_matrix[source]
        mask = d <= distance
        mask[source] = False
        return np.flatnonzero(mask).astype(np.int32)

    # ---- core procedures ----
    def initialise_run(self) -> None:
        self._seed_year()
        self.catchment_condition = 0.0

        for reef_idx in range(self.number_of_reefs):
            self.future_rezone_year[reef_idx] = 9999
            temp_y = max(0.0, 1.1 - (((self.y[reef_idx] + 15.0) / 15.0) ** 2))
            pH_x = 1.0 - 0.20 * math.exp(-1.0 * self.km_offshore[reef_idx] / self.pH_scale)
            self.pH_protect[reef_idx] = 0.0
            self.reef_shading[reef_idx] = 0.0
            self.regional_shading[reef_idx] = 0.0

            if self.search_mode == 1:
                self.cfg.consolidation_reefs = min(self.cfg.consolidation_reefs, 1)
                self.cfg.shading_reefs = min(self.cfg.shading_reefs, 1)
                self.cfg.seed_reefs = min(self.cfg.seed_reefs, 1)
                self.cfg.slick_reefs = min(self.cfg.slick_reefs, 1)
                self.cfg.pH_reefs = min(self.cfg.pH_reefs, 1)

            sl = self._sites_for_reef(reef_idx)
            self.rate["sa"][sl] = self.rate_i["sa"] * temp_y * pH_x * 1.0
            self.rate["ta"][sl] = (
                self.rate_i["ta"]
                * temp_y
                * pH_x
                * (1.0 * ((self.rate_i["sa"] / self.rate_i["ta"]) ** 0.073))
            )
            self.rate["mo"][sl] = (
                self.rate_i["mo"]
                * temp_y
                * pH_x
                * (1.0 * ((self.rate_i["sa"] / self.rate_i["mo"]) ** 0.073))
            )
            self.rate["po"][sl] = (
                self.rate_i["po"]
                * temp_y
                * pH_x
                * (1.0 * ((self.rate_i["sa"] / self.rate_i["po"]) ** 0.073))
            )
            self.rate["fa"][sl] = (
                self.rate_i["fa"]
                * temp_y
                * pH_x
                * (1.0 * ((self.rate_i["sa"] / self.rate_i["fa"]) ** 0.073))
            )
            self.rate["tt"][sl] = (
                self.rate_i["tt"]
                * temp_y
                * pH_x
                * (1.0 * ((self.rate_i["sa"] / self.rate_i["tt"]) ** 0.073))
            )

            for g in CORAL_GROUPS:
                self.thermal[g][sl] = self.thermal_i[g]

        if self.ensemble == 0:
            back = max(0, self.cfg.spinup_backtrack_years)
            self.year = max(1, self.cfg.start_year - back)
            for reef_idx in range(self.number_of_reefs):
                self.E["5"][reef_idx] = 0
                self.E["4"][reef_idx] = 1
                self.E["3"][reef_idx] = 2
                self.E["2"][reef_idx] = 4
                self.E["1"][reef_idx] = 8
                self.E["0"][reef_idx] = 16

                self.G["5"][reef_idx] = 1
                self.G["4"][reef_idx] = 2
                self.G["3"][reef_idx] = 4
                self.G["2"][reef_idx] = 8
                self.G["1"][reef_idx] = 16
                self.G["0"][reef_idx] = 32

                sl = self._sites_for_reef(reef_idx)
                n = sl.stop - sl.start
                self.B[sl] = self.rng._rs.randint(0, int(0.2 * self.B_max), size=n)
                self.T[sl] = self.rng._rs.randint(0, int(0.2 * self.T_max), size=n)
                self.S["6"][sl] = 0
                self.S["5"][sl] = 0
                self.S["4"][sl] = 1
                self.S["3"][sl] = 5
                self.S["2"][sl] = 20
                self.S["1"][sl] = 100
                self.S["0"][sl] = 500

                self.C["sa"][sl] = 0.1
                self.C["ta"][sl] = 0.1
                self.C["mo"][sl] = 0.1
                self.C["po"][sl] = 0.1
                self.C["fa"][sl] = 0.1
                self.C["tt"][sl] = 0.0
                self.C_site[sl] = (
                    self.C["sa"][sl]
                    + self.C["ta"][sl]
                    + self.C["mo"][sl]
                    + self.C["po"][sl]
                    + self.C["fa"][sl]
                    + self.C["tt"][sl]
                )
                self.R_site[sl] = 0.2
        else:
            self.year = self.cfg.start_year
            for age in ("0", "1", "2", "3", "4", "5"):
                self.E[age][:] = self.E_i[age]
                self.G[age][:] = self.G_i[age]
            for age in ("0", "1", "2", "3", "4", "5", "6"):
                self.S[age][:] = self.S_i[age]
            self.B[:] = self.B_i
            self.T[:] = self.T_i
            for g in CORAL_GROUPS:
                self.C[g][:] = self.C_i[g]
            self.C_site[:] = (
                self.C["sa"]
                + self.C["ta"]
                + self.C["mo"]
                + self.C["po"]
                + self.C["fa"]
                + self.C["tt"]
            )
            self.R_site[:] = self.R_site_i

        self.reef_populations()
        self._assign_priority()
        if self.cfg.unregulated_fishing:
            self.rezone_year[:] = 9999
            self.future_rezone_year[:] = 9999

    def _assign_priority(self) -> None:
        p = 1
        for category in ("T", "P", "N"):
            idxs = np.flatnonzero(self.priority_category == category)
            for reef_idx in idxs:
                self.priority[reef_idx] = p
                if p < self.cfg.rezoned_reefs and self.rezone_year[reef_idx] == 9999:
                    self.future_rezone_year[reef_idx] = self.cfg.start_modified_zoning
                p += 1

    def save_start_conditions(self) -> None:
        for age in ("0", "1", "2", "3", "4", "5"):
            self.E_i[age][:] = self.E[age]
            self.G_i[age][:] = self.G[age]
        self.B_i[:] = self.B
        self.T_i[:] = self.T
        for age in ("0", "1", "2", "3", "4", "5", "6"):
            self.S_i[age][:] = self.S[age]
        for g in CORAL_GROUPS:
            self.C_i[g][:] = self.C[g]
        self.C_site_i[:] = self.C_site
        self.R_site_i[:] = self.R_site

    def grow_fish(self, reef_idx: int) -> None:
        self._seed_year()
        temp_depend = 1.0 + 0.01 * (self.y[reef_idx] + 25.0) ** 2
        g_weighted = (
            self.G["1"][reef_idx]
            + 2 * self.G["2"][reef_idx]
            + 3 * self.G["3"][reef_idx]
            + 4 * self.G["4"][reef_idx]
            + 5 * self.G["5"][reef_idx]
        ) / 15.0

        sl = self._sites_for_reef(reef_idx)
        c_site = self.C_site[sl]
        r_site = self.R_site[sl]
        b = self.B[sl]
        t = self.T[sl]

        predation = (
            self.T_pred_B
            * t
            * b
            / (b + self.T_pred_B * t + self.small)
            * np.exp(-1.0 * (c_site + r_site))
        )
        b = b * (1 + self.B_recruit) - predation - self.B_recruit * b * b / self.B_max
        b = np.array([nl_median(100.0, bb, hh) for bb, hh in zip(b, self.B_max * (c_site + r_site))])
        self.B[sl] = b

        predation = (
            self.G_pred_T
            * g_weighted
            * t
            / (t + self.G_pred_T * g_weighted + self.small)
            * np.exp(-1.0 * c_site)
        )
        t = t * (1 + self.T_recruit) - predation - self.T_recruit * t * t / self.T_max
        t = np.array([nl_median(1.0, tt, hh) for tt, hh in zip(t, self.T_max * c_site)])
        self.T[sl] = t

        mortality = self.E_mort * temp_depend / (1.0 + self.C_reef[reef_idx])
        self.E["5"][reef_idx] = nl_round(
            (self.E["4"][reef_idx] + self.E["5"][reef_idx])
            * max(0.0, (1 - 0.25 * mortality * (self.E["4"][reef_idx] + self.E["5"][reef_idx])))
        )
        self.E["4"][reef_idx] = nl_round(
            self.E["3"][reef_idx] * max(0.0, (1 - 0.33 * mortality * self.E["3"][reef_idx]))
        )
        self.E["3"][reef_idx] = nl_round(
            self.E["2"][reef_idx] * max(0.0, (1 - 0.5 * mortality * self.E["2"][reef_idx]))
        )
        self.E["2"][reef_idx] = nl_round(
            self.E["1"][reef_idx] * max(0.0, (1 - 1.0 * mortality * self.E["1"][reef_idx]))
        )
        self.E["1"][reef_idx] = nl_round(self.E["0"][reef_idx] * self.C_reef[reef_idx] ** 0.6)

        mortality = self.G_mort * temp_depend / (1.0 + self.C_reef[reef_idx])
        self.G["5"][reef_idx] = nl_round(
            (self.G["4"][reef_idx] + self.G["5"][reef_idx])
            * max(0.0, (1 - 0.25 * mortality * (self.G["4"][reef_idx] + self.G["5"][reef_idx])))
        )
        self.G["4"][reef_idx] = nl_round(
            self.G["3"][reef_idx] * max(0.0, (1 - 0.33 * mortality * self.G["3"][reef_idx]))
        )
        self.G["3"][reef_idx] = nl_round(
            self.G["2"][reef_idx] * max(0.0, (1 - 0.5 * mortality * self.G["2"][reef_idx]))
        )
        self.G["2"][reef_idx] = nl_round(
            self.G["1"][reef_idx] * max(0.0, (1 - 1.0 * mortality * self.G["1"][reef_idx]))
        )
        self.G["1"][reef_idx] = nl_round(self.G["0"][reef_idx] * self.C_reef[reef_idx] ** 0.6)

    def apply_fishing(self) -> None:
        self._seed_year()
        self.E_catch_kg[:] = 0.0
        self.G_catch_kg[:] = 0.0

        kg_per_E_3, kg_per_E_4, kg_per_E_5 = 1.0, 1.5, 2.5
        kg_per_G_3, kg_per_G_4, kg_per_G_5 = 1.0, 1.5, 3.0

        reporting_rate = 0.8 + self.rng.random_float(0.2)
        if self.year < 2004 or self.cfg.unregulated_fishing:
            e_annual = max(
                600000.0,
                1500000.0 * (1 - math.exp(-0.01 * math.exp(0.08 * (self.year - 1940)))),
            ) / (self.reporting_ratio * reporting_rate)
            g_annual = max(
                1160000.0,
                2900000.0 * (1 - math.exp(-0.01 * math.exp(0.08 * (self.year - 1940)))),
            ) / (self.reporting_ratio * reporting_rate)
        else:
            e_annual = 400000.0 / reporting_rate
            g_annual = 900000.0 / reporting_rate
        if self.year >= self.cfg.start_modified_fishing:
            e_annual *= 1 - self.cfg.catch_reduction
            g_annual *= 1 - self.cfg.catch_reduction

        e_cumulative = 0.0
        visits = 0
        max_visits = 3000
        while e_cumulative < e_annual and visits < max_visits:
            eligible = np.flatnonzero(
                (self.year < self.rezone_year) & (self.year < self.future_rezone_year)
            )
            if eligible.size == 0:
                break
            reef_idx = int(eligible[self.rng.random_int(eligible.size)])
            yy = 0.12 * (27 + self.y[reef_idx])
            prob_fish = 0.104 / yy * math.exp(-2 * (math.log(yy) ** 2))
            if self.rng.random_float(1.0) < prob_fish:
                visits += 1
                effort = math.exp(-1 * self.rng.random_float(1.0))
                e3 = effort * self.E["3"][reef_idx]
                e4 = effort * self.E["4"][reef_idx]
                e5 = effort * self.E["5"][reef_idx]
                if self.year >= self.cfg.start_lower_sizelimit:
                    e3 = 0.0
                if self.year >= self.cfg.start_upper_sizelimit:
                    e5 = 0.0
                if (
                    self.year >= self.cfg.start_CoTSlimit
                    and (
                        0.55 * self.S_r["2"][reef_idx]
                        + 0.70 * self.S_r["3"][reef_idx]
                        + 0.85 * self.S_r["4"][reef_idx]
                        + 0.95 * self.S_r["5"][reef_idx]
                        + 0.99 * self.S_r["6"][reef_idx]
                    )
                    > 68
                ):
                    e3 = e4 = e5 = 0.0
                self.E["3"][reef_idx] = max(0, nl_round(self.E["3"][reef_idx] - e3))
                self.E["4"][reef_idx] = max(0, nl_round(self.E["4"][reef_idx] - e4))
                self.E["5"][reef_idx] = max(0, nl_round(self.E["5"][reef_idx] - e5))
                catch = (
                    (e3 * kg_per_E_3 + e4 * kg_per_E_4 + e5 * kg_per_E_5)
                    * self.reef_sites[reef_idx]
                    * self.ha_per_site
                )
                self.E_catch_kg[reef_idx] += catch
                e_cumulative += catch

        g_cumulative = 0.0
        visits = 0
        while g_cumulative < g_annual and visits < max_visits:
            eligible = np.flatnonzero(
                (self.year < self.rezone_year) & (self.year < self.future_rezone_year)
            )
            if eligible.size == 0:
                break
            reef_idx = int(eligible[self.rng.random_int(eligible.size)])
            yy = 0.17 * (25 + self.y[reef_idx])
            prob_fish = 0.18 / yy * math.exp(-2 * (math.log(yy) ** 2))
            if self.rng.random_float(1.0) < prob_fish:
                visits += 1
                effort = math.exp(-1 * self.rng.random_float(1.0))
                g3 = effort * self.G["3"][reef_idx]
                g4 = effort * self.G["4"][reef_idx]
                g5 = effort * self.G["5"][reef_idx]
                if self.year >= self.cfg.start_lower_sizelimit:
                    g3 = 0.0
                if self.year >= self.cfg.start_upper_sizelimit:
                    g5 = 0.0
                if (
                    self.year >= self.cfg.start_CoTSlimit
                    and (
                        0.55 * self.S_r["2"][reef_idx]
                        + 0.70 * self.S_r["3"][reef_idx]
                        + 0.85 * self.S_r["4"][reef_idx]
                        + 0.95 * self.S_r["5"][reef_idx]
                        + 0.99 * self.S_r["6"][reef_idx]
                    )
                    > 68
                ):
                    g3 = g4 = g5 = 0.0
                self.G["3"][reef_idx] = max(0, nl_round(self.G["3"][reef_idx] - g3))
                self.G["4"][reef_idx] = max(0, nl_round(self.G["4"][reef_idx] - g4))
                self.G["5"][reef_idx] = max(0, nl_round(self.G["5"][reef_idx] - g5))
                catch = (
                    (g3 * kg_per_G_3 + g4 * kg_per_G_4 + g5 * kg_per_G_5)
                    * self.reef_sites[reef_idx]
                    * self.ha_per_site
                )
                self.G_catch_kg[reef_idx] += catch
                g_cumulative += catch

    def grow_corals(self, reef_idx: int) -> None:
        self._seed_year()
        flood_denom = max(self.flood_scale * self.flood_load, self.small)
        flood_effect = 0.1 + 0.9 * math.exp(-1 * self.km_offshore[reef_idx] / flood_denom)
        pH_effect_t = 0.0
        if self.year >= self.cfg.projection_year:
            pH_effect_t = (1.0 - self.pH_protect[reef_idx]) * math.sqrt(self.cfg.SSP)

        sl = self._sites_for_reef(reef_idx)
        rubble_retention = (self.reef_who[reef_idx] % 11) / 20.0

        mask = (self.C_site[sl] + self.R_site[sl]) < 0.8
        if np.any(mask):
            for g, k in (
                ("sa", self.k_sa),
                ("ta", self.k_ta),
                ("mo", self.k_mo),
                ("po", self.k_po),
                ("fa", self.k_fa),
                ("tt", self.k_tt),
            ):
                self.rate[g][sl][mask] = self.rate[g][sl][mask] ** (1 + pH_effect_t * k)

            denom = 1 + np.sqrt(self.C_site[sl][mask] + self.R_site[sl][mask])
            self.C["fa"][sl][mask] *= 1 + self.rate["fa"][sl][mask] * (1 - flood_effect) / denom
            self.C["po"][sl][mask] *= 1 + self.rate["po"][sl][mask] * (1 - flood_effect) / denom
            self.C["mo"][sl][mask] *= 1 + self.rate["mo"][sl][mask] * (1 - flood_effect) / denom
            self.C["ta"][sl][mask] *= 1 + self.rate["ta"][sl][mask] * (1 - flood_effect) / denom
            self.C["tt"][sl][mask] *= 1 + self.rate["tt"][sl][mask] * (1 - flood_effect) / denom
            self.C["sa"][sl][mask] *= 1 + self.rate["sa"][sl][mask] * (1 - flood_effect) / denom

        self.C_site[sl] = (
            self.C["sa"][sl]
            + self.C["ta"][sl]
            + self.C["mo"][sl]
            + self.C["fa"][sl]
            + self.C["po"][sl]
            + self.C["tt"][sl]
        )

        self.R_site[sl] = np.minimum(
            self.R_site[sl] * (1 - 1 / self.rubble_decay_time),
            rubble_retention,
        )

        over = (self.C_site[sl] + self.R_site[sl]) > 0.7
        if np.any(over):
            scale = self.C_site[sl][over] + self.R_site[sl][over] + 0.3
            for g in CORAL_GROUPS:
                self.C[g][sl][over] = self.C[g][sl][over] / scale
            self.R_site[sl][over] = self.R_site[sl][over] / scale

        self.C_site[sl] = (
            self.C["sa"][sl]
            + self.C["ta"][sl]
            + self.C["mo"][sl]
            + self.C["fa"][sl]
            + self.C["po"][sl]
            + self.C["tt"][sl]
        )

        over1 = (self.C_site[sl] + self.R_site[sl]) > 1.0
        if np.any(over1):
            self.C["fa"][sl][over1] = 0.1
            self.C["po"][sl][over1] = 0.1
            self.C["mo"][sl][over1] = 0.1
            self.C["ta"][sl][over1] = 0.1
            self.C["tt"][sl][over1] = 0.0
            self.C["sa"][sl][over1] = 0.1
            self.C_site[sl][over1] = 0.5
            self.R_site[sl][over1] = 0.1

    def grow_cots(self, reef_idx: int) -> None:
        self._seed_year()
        sl = self._sites_for_reef(reef_idx)
        for s in range(sl.start, sl.stop):
            c_f = nl_median(
                0, self.C["sa"][s] + self.C["ta"][s] + self.C["mo"][s] + self.C["tt"][s], 1
            )
            site_cap = math.sqrt(c_f)
            old = {a: self.S[a][s] for a in ("0", "1", "2", "3", "4", "5", "6")}
            self.S["6"][s] = nl_round(
                (old["5"] + old["6"]) * math.exp(-1 * self.S6_mort / (c_f + self.small))
            )
            self.S["5"][s] = nl_round(old["4"] * math.exp(-1 * self.S5_mort / (c_f + self.small)))
            self.S["4"][s] = nl_round(old["3"] * math.exp(-1 * self.S4_mort / (c_f + self.small)))
            self.S["3"][s] = nl_round(old["2"] * math.exp(-1 * self.S3_mort / (c_f + self.small)))
            self.S["2"][s] = nl_round(
                old["1"] * site_cap * math.exp(-1 * self.S2_mort / (c_f + self.small))
            )
            self.S["1"][s] = nl_round(
                (old["0"] + old["1"] * (1 - site_cap))
                * math.exp(-1 * self.S1_mort / (self.R_site[s] + self.small))
            )
            self.S["0"][s] = 0
            self.S_manta[s] = (
                0.50 * self.S["2"][s]
                + 0.70 * self.S["3"][s]
                + 0.85 * self.S["4"][s]
                + 0.95 * self.S["5"][s]
                + 0.99 * self.S["6"][s]
            )

    def consume_corals(self, reef_idx: int) -> None:
        self._seed_year()
        sl = self._sites_for_reef(reef_idx)
        for s in range(sl.start, sl.stop):
            pred = self.S_pred_C * (
                0.1 * self.S["1"][s]
                + 1 * self.S["2"][s]
                + 2 * self.S["3"][s]
                + 3 * self.S["4"][s]
                + 4 * self.S["5"][s]
                + 5 * self.S["6"][s]
            )
            for g in ("sa", "tt", "ta", "mo", "po", "fa"):
                consume = min(self.S_prefer * pred, 0.9 * self.C[g][s])
                self.C[g][s] -= consume
                self.R_site[s] = nl_median(0.0, 1.0, self.R_site[s] + 2 * consume)
                pred -= consume

    def consume_cots(self, reef_idx: int) -> None:
        self._seed_year()
        e_weighted = (
            self.E["1"][reef_idx]
            + 2 * self.E["2"][reef_idx]
            + 3 * self.E["3"][reef_idx]
            + 4 * self.E["4"][reef_idx]
            + 5 * self.E["5"][reef_idx]
        ) / 15.0
        sl = self._sites_for_reef(reef_idx)
        for s in range(sl.start, sl.stop):
            e_site = self.rng.random_int(2 * e_weighted)
            pred = (
                self.B_pred_S1
                * self.B[s]
                * self.S["1"][s]
                / (self.S["1"][s] + self.B_pred_S1 * self.B[s] + self.small)
            )
            self.S["1"][s] = nl_round(max(10.0, self.S["1"][s] - pred))
            pred = (
                self.E_pred_S1
                * e_site
                * self.S["1"][s]
                / (self.S["1"][s] + self.E_pred_S1 * e_site + self.small)
                * math.exp(-1 * self.R_site[s])
            )
            self.S["1"][s] = nl_ceiling(max(10.0, self.S["1"][s] - pred))
            for a in ("2", "3", "4", "5", "6"):
                pred = (
                    self.E_pred_S
                    * e_site
                    * self.S[a][s]
                    / (self.S[a][s] + self.E_pred_S * e_site + self.small)
                )
                self.S[a][s] = nl_ceiling(max(0.0, self.S[a][s] - pred))

    def spawn_fish(self, reef_idx: int) -> None:
        self._seed_year()
        kernel = self._reef_kernel(
            reef_idx, self._kernel_base_grouper(self.draw_year, self.draw_fortnight)
        )
        g_natal = self.rng.random_float(math.exp(-50 / math.sqrt(self.reef_sites[reef_idx])))
        g_source = (
            self.G["2"][reef_idx]
            + 2 * self.G["3"][reef_idx]
            + 4 * self.G["4"][reef_idx]
            + 8 * self.G["5"][reef_idx]
        ) * max(0.0, 1 - 0.004 * (self.y[reef_idx] + 25) ** 2)
        self.G["0"][reef_idx] = self.rng.random_int(
            (kernel.con1 + kernel.con2) * self.G_recruit * g_source * g_natal
        )

        self._spawn_grouper_kernel(
            reef_idx, g_source, g_natal, kernel.con1, kernel.dir1, kernel.ang1, kernel.dis1
        )
        self._spawn_grouper_kernel(
            reef_idx, g_source, g_natal, kernel.con2, kernel.dir2, kernel.ang2, kernel.dis2
        )

        e_source = (self.E["4"][reef_idx] + 2 * self.E["5"][reef_idx]) * max(
            0.0, 1 - 0.004 * (self.y[reef_idx] + 25) ** 2
        )
        self.E["0"][reef_idx] = self.rng.random_int(
            self.E_recruit * e_source * self.E_natal * self.reef_sites[reef_idx]
        )
        targets = self._targets_in_radius(reef_idx, 100 * self.per_km)
        for t in targets:
            self.E["0"][t] += self.rng.random_int(
                self.E_recruit * e_source * (1 - self.E_natal) * self.reef_sites[t]
            )

    def _spawn_grouper_kernel(
        self,
        source: int,
        g_source: float,
        g_natal: float,
        con: float,
        direction: float,
        angle: float,
        distance: float,
    ) -> None:
        if distance <= 0 or con <= 0:
            return
        radius = distance * self.per_km
        targets, dists = self._targets_in_cone(source, radius, direction, angle)
        if targets.size == 0:
            return
        include = self.rng._rs.random_sample(size=targets.size) > (dists / radius)
        for t in targets[include]:
            self.G["0"][t] += self.rng.random_int(
                con * self.G_recruit * g_source * (1 - g_natal) * self.reef_sites[t]
            )

    def spawn_corals(self, reef_idx: int) -> None:
        self._seed_year()
        src_site = self._one_site_for_reef(reef_idx)
        if src_site is None:
            return

        hybrid = self.cfg.hybrid_fraction * self.C["sa"][src_site]
        allee_sa = math.exp(-1.0 * (self.C_allee / (self.small + self.C["sa"][src_site])) ** 0.5)
        c_sa_source = allee_sa * (
            (self.C["sa"][src_site] - hybrid)
            + hybrid
            * (
                (hybrid * hybrid + 2 * (1 - self.cfg.dominance) * hybrid * self.C["tt"][src_site])
                / ((hybrid + self.C["tt"][src_site] + self.small) ** 2)
            )
        )
        c_ta_source = self.C["ta"][src_site] * math.exp(
            -1.0 * (self.C_allee / (self.small + self.C["ta"][src_site])) ** 0.5
        )
        c_mo_source = self.C["mo"][src_site] * math.exp(
            -1.0 * (self.C_allee / (self.small + self.C["mo"][src_site])) ** 0.5
        )
        c_po_source = self.C["po"][src_site] * math.exp(
            -1.0 * (self.C_allee / (self.small + self.C["po"][src_site])) ** 0.5
        )
        c_fa_source = self.C["fa"][src_site] * math.exp(
            -1.0 * (self.C_allee / (self.small + self.C["fa"][src_site])) ** 0.5
        )
        allee_tt = math.exp(-1.0 * (self.C_allee / (self.small + self.C["tt"][src_site])) ** 0.5)
        c_tt_source = (
            allee_tt
            * self.C["tt"][src_site]
            * (
                self.C["tt"][src_site]
                * ((2 * self.cfg.dominance * hybrid) + self.C["tt"][src_site])
                / ((hybrid + self.C["tt"][src_site] + self.small) ** 2)
            )
        )

        thermal_source = {g: self.thermal[g][src_site] for g in CORAL_GROUPS}
        kernel = self._reef_kernel(
            reef_idx, self._kernel_base_coral(self.draw_year, self.draw_month)
        )
        self._recruits_total = 0.0

        self._spawn_coral_kernel(
            reef_idx,
            kernel.con1,
            kernel.dir1,
            kernel.ang1,
            kernel.dis1,
            c_sa_source,
            c_ta_source,
            c_mo_source,
            c_po_source,
            c_fa_source,
            c_tt_source,
            thermal_source,
        )
        self._spawn_coral_kernel(
            reef_idx,
            kernel.con2,
            kernel.dir2,
            kernel.ang2,
            kernel.dis2,
            c_sa_source,
            c_ta_source,
            c_mo_source,
            c_po_source,
            c_fa_source,
            c_tt_source,
            thermal_source,
        )

        sl = self._sites_for_reef(reef_idx)
        self.C_site[sl] = (
            self.C["sa"][sl]
            + self.C["ta"][sl]
            + self.C["mo"][sl]
            + self.C["fa"][sl]
            + self.C["po"][sl]
            + self.C["tt"][sl]
        )
        over = (self.C_site[sl] + self.R_site[sl]) > 0.7
        if np.any(over):
            cr = self.C_site[sl][over] + self.R_site[sl][over] + 0.3
            for g in CORAL_GROUPS:
                self.C[g][sl][over] = self.C[g][sl][over] / cr
        self.C_site[sl] = (
            self.C["sa"][sl]
            + self.C["ta"][sl]
            + self.C["mo"][sl]
            + self.C["fa"][sl]
            + self.C["po"][sl]
            + self.C["tt"][sl]
        )
        over1 = (self.C_site[sl] + self.R_site[sl]) > 1.0
        if np.any(over1):
            self.C["fa"][sl][over1] = 0.1
            self.C["po"][sl][over1] = 0.1
            self.C["mo"][sl][over1] = 0.1
            self.C["ta"][sl][over1] = 0.1
            self.C["tt"][sl][over1] = 0.0
            self.C["sa"][sl][over1] = 0.1
            self.R_site[sl][over1] = 0.1
        self.C_out_degree[reef_idx] = self._recruits_total

    def _spawn_coral_kernel(
        self,
        source: int,
        con: float,
        direction: float,
        angle: float,
        distance: float,
        c_sa_source: float,
        c_ta_source: float,
        c_mo_source: float,
        c_po_source: float,
        c_fa_source: float,
        c_tt_source: float,
        thermal_source: dict[str, float],
    ) -> None:
        """Dispersal cone for one connectivity kernel; updates C and thermal per site.

        Recruitment matches the former six-call ``_calculate_recruitment`` loop per
        site. **Faithful vs legacy Python port:**

        - **RNG:** One ``RandomState.random_sample(6)`` per site replaces six
          ``NetLogoRng.random_float`` calls; MT19937 consumption order is unchanged
          (groups in ``CORAL_SPAWN_KERNEL_ORDER``).
        - **Recruits:** Still ``float(U * (c_source * limits)) * (rate / rate_i)``
          per group (same as ``random_float(c_source * limits) * (rate / rate_i)``).
        - **Thermal:** ``numpy.maximum(1.0, …)`` is element-wise equivalent to
          scalar ``max(1.0, …)``.

        **No intentional behavioural change:** reads use pre-update cover/thermal;
        writes match the old loop order (sa, tt, ta, mo, po, fa).
        
        **Accumulation:** Recruits are accumulated into ``self._recruits_total``
        to maintain NetLogo's single running sum across both kernels.
        """
        if distance <= 0 or con <= 0:
            return
        radius = distance * self.per_km
        targets, dists = self._targets_in_cone(source, radius, direction, angle)
        if targets.size == 0:
            return
        include = self.rng._rs.random_sample(size=targets.size) > (dists / radius)
        # Batched recruitment (see module doc note on _spawn_coral_kernel).
        src_arr = np.array(
            (c_sa_source, c_tt_source, c_ta_source, c_mo_source, c_po_source, c_fa_source),
            dtype=np.float64,
        )
        ri = np.array([self.rate_i[g] for g in CORAL_SPAWN_KERNEL_ORDER], dtype=np.float64)
        ts_arr = np.array([thermal_source[g] for g in CORAL_SPAWN_KERNEL_ORDER], dtype=np.float64)
        recruits_total = 0.0
        u_buf = np.empty(6, dtype=np.float64)
        rates = np.empty(6, dtype=np.float64)
        c_ex = np.empty(6, dtype=np.float64)
        th_ex = np.empty(6, dtype=np.float64)
        for reef_idx in targets[include]:
            sl = self._sites_for_reef(int(reef_idx))
            for s in range(sl.start, sl.stop):
                limits = self.C_recruit * con * nl_median(0, 1 - self.C_site[s] - self.R_site[s], 1)
                for i, g in enumerate(CORAL_SPAWN_KERNEL_ORDER):
                    rates[i] = self.rate[g][s]
                    c_ex[i] = self.C[g][s]
                    th_ex[i] = self.thermal[g][s]
                u_buf[:] = self.rng._rs.random_sample(6)
                # Per-group: float(U * (c_source * limits)) * (rate / rate_i) — matches
                # NetLogoRng.random_float(c_source * limits) * (rate / rate_i).
                for i in range(6):
                    u_buf[i] = float(u_buf[i] * (src_arr[i] * limits)) * (rates[i] / ri[i])
                recruits = u_buf  # reuse buffer as recruit vector
                denom = recruits + c_ex + self.small
                thermal_new = np.maximum(1.0, (recruits * ts_arr + c_ex * th_ex) / denom)
                for i, g in enumerate(CORAL_SPAWN_KERNEL_ORDER):
                    self.C[g][s] += recruits[i]
                    self.thermal[g][s] = thermal_new[i]
                    self._recruits_total += recruits[i]

    def spawn_cots(self, reef_idx: int) -> None:
        self._seed_year()
        kernel = self._reef_kernel(
            reef_idx, self._kernel_base_cots(self.draw_year, self.draw_fortnight)
        )
        s_natal = self.rng.random_float(math.exp(-50 / math.sqrt(self.reef_sites[reef_idx])))

        sl = self._sites_for_reef(reef_idx)
        s_source = 0.0
        for s in range(sl.start, sl.stop):
            adults = (
                self.S["2"][s] + self.S["3"][s] + self.S["4"][s] + self.S["5"][s] + self.S["6"][s]
            )
            allee = math.exp(-1.0 * (self.S_allee / (self.small + adults)) ** 0.5)
            s_source += (
                self.S["2"][s]
                + 2 * self.S["3"][s]
                + 4 * self.S["4"][s]
                + 8 * self.S["5"][s]
                + 8 * self.S["6"][s]
            ) * allee
        for s in range(sl.start, sl.stop):
            self.S["0"][s] += self.rng.random_int(
                (kernel.con1 + kernel.con2) * self.S_recruit * s_source * s_natal * self.R_site[s]
            )

        self._spawn_cots_kernel(
            reef_idx, s_source, s_natal, kernel.con1, kernel.dir1, kernel.ang1, kernel.dis1
        )
        self._spawn_cots_kernel(
            reef_idx, s_source, s_natal, kernel.con2, kernel.dir2, kernel.ang2, kernel.dis2
        )

    def _spawn_cots_kernel(
        self,
        source: int,
        s_source: float,
        s_natal: float,
        con: float,
        direction: float,
        angle: float,
        distance: float,
    ) -> None:
        if distance <= 0 or con <= 0:
            return
        radius = distance * self.per_km
        targets, dists = self._targets_in_cone(source, radius, direction, angle)
        if targets.size == 0:
            return
        include = self.rng._rs.random_sample(size=targets.size) > (dists / radius)
        for reef_idx in targets[include]:
            sl = self._sites_for_reef(int(reef_idx))
            for s in range(sl.start, sl.stop):
                self.S["0"][s] = min(
                    self.S["0"][s]
                    + self.rng.random_int(
                        con * self.S_recruit * s_source * (1 - s_natal) * self.R_site[s]
                    ),
                    1000000,
                )

    def release_fish(self) -> None:
        self._seed_year()
        p = 0
        treatments = 0
        while treatments < self.cfg.release_reefs and p <= self.number_of_reefs:
            p += 1
            reef_idx = self._reef_by_priority(p)
            if reef_idx is None:
                continue
            if not self._reef_in_intervention_bounds(reef_idx):
                continue
            adults = (
                self.E["2"][reef_idx]
                + self.E["3"][reef_idx]
                + self.E["4"][reef_idx]
                + self.E["5"][reef_idx]
            )
            if adults >= self.cfg.release_threshold:
                continue
            self.E["1"][reef_idx] = nl_round(
                self.E["1"][reef_idx]
                + self.cfg.release_number
                / max(self.cfg.release_reefs, 1)
                / (self.reef_sites[reef_idx] * self.ha_per_site)
            )
            treatments += 1

    def control_cots(self) -> None:
        self._seed_year()
        dives_remaining = 0.9 * 20 * 36 * 8 * self.S_vessels
        p = 0
        while dives_remaining > 0 and p <= self.number_of_reefs:
            p += 1
            dives_remaining -= 8
            reef_idx = self._reef_by_priority(p)
            if reef_idx is None:
                continue
            if not (
                self.region_name[reef_idx] == self.control_region or self.control_region == "GBR"
            ):
                continue
            if not (
                self.S_manta_r[reef_idx] < self.cfg.CoTS_threshold
                or self.C_reef[reef_idx] > self.cfg.coral_threshold
            ):
                continue
            dives_total = 0.0
            sl = self._sites_for_reef(reef_idx)
            idxs = np.flatnonzero(self.S_manta[sl] > self.cfg.eco_threshold) + sl.start
            for s in idxs:
                dives_site = nl_round((167 / 40) * self.S_manta[s] ** 0.667)
                dives_remaining -= dives_site
                dives_total += dives_site
                diver_detect = 1.5 + self.rng.random_float(0.5)
                self.S["2"][s] = nl_round(0.50 * self.cfg.eco_threshold / diver_detect)
                self.S["3"][s] = nl_round(0.30 * self.cfg.eco_threshold / diver_detect)
                self.S["4"][s] = nl_round(0.15 * self.cfg.eco_threshold / diver_detect)
                self.S["5"][s] = nl_round(0.05 * self.cfg.eco_threshold / diver_detect)
                self.S["6"][s] = nl_round(0.01 * self.cfg.eco_threshold / diver_detect)
                self.S["1"][s] = nl_round(
                    min(
                        self.S["1"][s],
                        2.77
                        * (
                            self.S["2"][s]
                            + self.S["3"][s]
                            + self.S["4"][s]
                            + self.S["5"][s]
                            + self.S["6"][s]
                        ),
                    )
                )
            self.dives_reef[reef_idx] += dives_total

    def control_cots_by_sector(self) -> None:
        self._seed_year()
        dives_remaining = 0.9 * 20 * 36 * 8 * self.S_vessels
        sector = 1
        s_max = -1.0
        for candidate in range(1, 12):
            idxs = np.flatnonzero(self.sector_number == candidate)
            if idxs.size == 0:
                continue
            value = float(np.mean(self.S_manta_r[idxs]))
            if value > s_max:
                s_max = value
                sector = candidate

        p = 0
        while dives_remaining > 0 and p <= self.number_of_reefs:
            p += 1
            dives_remaining -= 8
            reef_idx = self._reef_by_priority(p)
            if reef_idx is None or self.sector_number[reef_idx] != sector:
                continue
            dives_total = 0.0
            sl = self._sites_for_reef(reef_idx)
            idxs = np.flatnonzero(self.S_manta[sl] > self.cfg.eco_threshold) + sl.start
            for s in idxs:
                dives_site = nl_round(0.7 * (167 / 40) * self.S_manta[s] ** 0.667)
                dives_remaining -= dives_site
                dives_total += dives_site
                diver_detect = 1.5 + self.rng.random_float(0.5)
                self.S["2"][s] = nl_round(0.50 * self.cfg.eco_threshold / diver_detect)
                self.S["3"][s] = nl_round(0.30 * self.cfg.eco_threshold / diver_detect)
                self.S["4"][s] = nl_round(0.15 * self.cfg.eco_threshold / diver_detect)
                self.S["5"][s] = nl_round(0.05 * self.cfg.eco_threshold / diver_detect)
                self.S["6"][s] = nl_round(0.01 * self.cfg.eco_threshold / diver_detect)
                self.S["1"][s] = nl_round(
                    min(
                        self.S["1"][s],
                        2.77
                        * (
                            self.S["2"][s]
                            + self.S["3"][s]
                            + self.S["4"][s]
                            + self.S["5"][s]
                            + self.S["6"][s]
                        ),
                    )
                )
            self.dives_reef[reef_idx] += dives_total

    def consolidate_rubble(self) -> None:
        self._seed_year()
        p = 0
        treatments = 0
        threshold = self.cfg.consolidation_hectares / self.ha_per_site
        while treatments < self.cfg.consolidation_reefs and p <= self.number_of_reefs:
            p += 1
            reef_idx = self._reef_by_priority(p)
            if reef_idx is None or not self._reef_in_intervention_bounds(reef_idx):
                continue
            if self.R_reef[reef_idx] <= self.cfg.consolidation_threshold:
                continue
            sl = self._sites_for_reef(reef_idx)
            candidates = np.flatnonzero(self.R_site[sl] > threshold)
            if candidates.size == 0:
                continue
            chosen = int(candidates[self.rng.random_int(candidates.size)] + sl.start)
            self.R_site[chosen] -= threshold
            treatments += 1

    def seed_tt_coral(self) -> None:
        self._seed_year()
        p = 0
        treatments = 0
        seed_fraction = self.cfg.seed_hectares / max(self.cfg.seed_reefs, 1) / self.ha_per_site
        while treatments < self.cfg.seed_reefs and p <= self.number_of_reefs:
            p += 1
            reef_idx = self._reef_by_priority(p)
            if reef_idx is None or not self._reef_in_intervention_bounds(reef_idx):
                continue
            if self.C_reef[reef_idx] >= self.cfg.seed_threshold:
                continue
            sl = self._sites_for_reef(reef_idx)
            candidates = np.flatnonzero(
                (self.C["tt"][sl] < seed_fraction) & (self.C_site[sl] < (1 - seed_fraction))
            )
            if candidates.size == 0:
                continue
            s = int(candidates[self.rng.random_int(candidates.size)] + sl.start)
            self.thermal["tt"][s] = (
                seed_fraction * self.thermal_i["tt"] + self.C["tt"][s] * self.thermal["tt"][s]
            ) / (seed_fraction + self.C["tt"][s] + self.small)
            self.C["tt"][s] += seed_fraction
            treatments += 1

    def seed_coral_slick(self) -> None:
        self._seed_year()
        slick_fraction = self.cfg.slick_hectares / max(self.cfg.slick_reefs, 1) / self.ha_per_site
        source = self._one_global_site()
        if source is None:
            return
        denom = max(self.C_site[source], self.small)
        frac = {g: self.C[g][source] / denom for g in CORAL_GROUPS}
        thermal_slick = {g: self.thermal[g][source] for g in CORAL_GROUPS}

        p = 0
        treatments = 0
        while treatments < self.cfg.slick_reefs and p <= self.number_of_reefs:
            p += 1
            reef_idx = self._reef_by_priority(p)
            if reef_idx is None or not self._reef_in_intervention_bounds(reef_idx):
                continue
            if self.C_r["tt"][reef_idx] >= self.cfg.slick_threshold:
                continue
            sl = self._sites_for_reef(reef_idx)
            candidates = np.flatnonzero(self.C_site[sl] < (1 - slick_fraction))
            if candidates.size == 0:
                continue
            s = int(candidates[self.rng.random_int(candidates.size)] + sl.start)
            for g in CORAL_GROUPS:
                self.thermal[g][s] = (
                    slick_fraction * thermal_slick[g] + self.C[g][s] * self.thermal[g][s]
                ) / (slick_fraction + self.C[g][s] + self.small)
            for g in CORAL_GROUPS:
                self.C[g][s] += frac[g] * slick_fraction
            treatments += 1

    def shade_local_reef(self) -> None:
        self._seed_year()
        self.reef_shading[:] = 0.0
        p = 0
        treatments = 0
        while treatments < self.cfg.shading_reefs and p <= self.number_of_reefs:
            p += 1
            reef_idx = self._reef_by_priority(p)
            if reef_idx is None or not self._reef_in_intervention_bounds(reef_idx):
                continue
            self.reef_shading[reef_idx] = self.cfg.reef_shading_reduction
            treatments += 1

    def shade_regional_coral(self) -> None:
        self._seed_year()
        self.regional_shading[:] = 0.0
        mask = (
            (self.x > self.cfg.intervene_lon_min)
            & (self.x < self.cfg.intervene_lon_max)
            & (self.y > self.cfg.intervene_lat_min)
            & (self.y < self.cfg.intervene_lat_max)
        )
        self.regional_shading[mask] = self.cfg.regional_shading_reduction

    def increase_pH(self) -> None:
        self._seed_year()
        p = 0
        treatments = 0
        while treatments < self.cfg.pH_reefs and p <= self.number_of_reefs:
            p += 1
            reef_idx = self._reef_by_priority(p)
            if reef_idx is None or not self._reef_in_intervention_bounds(reef_idx):
                continue
            self.pH_protect[reef_idx] = self.cfg.pH_protection
            treatments += 1

    def bleaching(self) -> None:
        self._seed_year()
        self.dhw[:] = 0.0
        for g in CORAL_GROUPS:
            self.bleach_mort[g][:] = 0.0

        dhw_max = 0.0
        bleaching_centre: int | None = None
        if self.year > 1997 and self.year < self.cfg.projection_year:
            historical = {
                1998: (4.5, (-22, -21)),
                2002: (12.0, (-22, -21)),
                2003: (6.0, (-22, -21)),
                2016: (11.5, (-12, -11)),
                2017: (11.5, (-16, -15)),
                2020: (10.5, (-20, -19)),
                2022: (8.0, (-16, -15)),
                2024: (10.5, (-22, -21)),
            }
            if self.year in historical:
                dhw_max, lat_band = historical[self.year]
                candidates = np.flatnonzero((self.y > lat_band[0]) & (self.y < lat_band[1]))
                if candidates.size > 0:
                    bleaching_centre = int(candidates[self.rng.random_int(candidates.size)])
                else:
                    import warnings
                    warnings.warn(
                        f"Historical bleaching event {self.year}: empty latitude band "
                        f"({lat_band[0]}, {lat_band[1]}), no reefs found"
                    )
        elif self.year >= self.cfg.projection_year:
            self._seed_year()
            if self.cfg.SSP == 1.9:
                dhw_max = (8 - self.rng.random_float(16)) - 0.0028 * (self.year - 2060) ** 2 + 8
            if self.cfg.SSP == 2.6:
                dhw_max = (8 - self.rng.random_float(16)) - 0.0026 * (self.year - 2070) ** 2 + 10
            if self.cfg.SSP == 4.5:
                dhw_max = (8 - self.rng.random_float(16)) + 0.221 * self.year - 444
            if self.cfg.SSP == 7.0:
                dhw_max = (8 - self.rng.random_float(16)) + 0.0037 * (self.year - 2010) ** 2 + 4
            if self.cfg.SSP == 8.5:
                dhw_max = (8 - self.rng.random_float(16)) + 0.0047 * (self.year - 2000) ** 2 + 1
            bleaching_centre = self.rng.random_int(self.number_of_reefs)

        if dhw_max <= 0 or bleaching_centre is None:
            return

        bleaching_radius = (self.dhw_scale * self.per_km) * dhw_max * (
            0.5 + self.rng.random_float(1.0)
        )
        centre_dist = self.dist_matrix[bleaching_centre]
        affected = np.flatnonzero(centre_dist <= bleaching_radius)
        
        # NetLogo uses random ask order for reefs
        affected_permuted = self.rng._rs.permutation(affected)
        
        for reef_idx in affected_permuted:
            radial = 1 - nl_median(
                0,
                ((centre_dist[reef_idx] / bleaching_radius) ** 2),
                1,
            )
            # NetLogo draws rand at reef level but overwrites it at site level (line 2351 vs 2357)
            # Must draw to maintain RNG stream alignment
            _ = (0.5 + self.rng.random_float(0.5)) * radial
            
            self.dhw[reef_idx] = (
                dhw_max
                * radial
                * (1 - self.reef_shading[reef_idx])
                * (1 - self.regional_shading[reef_idx])
            )
            dhw_reef = self.dhw[reef_idx]
            sl = self._sites_for_reef(int(reef_idx))
            for s in range(sl.start, sl.stop):
                rand = self.rng.random_float(1.0)
                new_rubble = 0.0
                for g in CORAL_GROUPS:
                    mort = max(0.0, rand * (1 - math.exp(-0.12 * (dhw_reef - self.thermal[g][s]))))
                    self.bleach_mort[g][s] = mort
                    new_rubble += 2 * self.C[g][s] * mort
                    self.C[g][s] *= 1 - mort
                    self.thermal[g][s] *= (1 + self.adaptability) ** mort
                    self.thermal[g][s] = min(
                        self.thermal[g][s], self.thermal_i[g] + self.adapt_plasticity
                    )

                self.R_site[s] = min(1.0, self.R_site[s] + new_rubble)

                self.thermal["sa"][s] -= (
                    self.thermal["sa"][s] - self.thermal_i["sa"]
                ) / self.adapt_decay_time
                self.rate["sa"][s] = self.rate_i["sa"] * (
                    1 - self.adapt_penalty * (self.thermal["sa"][s] - self.thermal_i["sa"])
                )
                for g in ("ta", "mo", "po", "fa", "tt"):
                    decay = self.adapt_decay_time / (
                        (self.rate[g][s] + self.small) / (self.rate["sa"][s] + self.small)
                    )
                    self.thermal[g][s] -= (self.thermal[g][s] - self.thermal_i[g]) / decay
                    self.rate[g][s] = self.rate_i[g] * (
                        1 - self.adapt_penalty * (self.thermal[g][s] - self.thermal_i[g])
                    )

    def cyclone(self) -> None:
        self._seed_year()
        smaller = (200 + self.rng.random_int(300)) * self.per_km
        medium = (400 + self.rng.random_int(300)) * self.per_km
        larger = (600 + self.rng.random_int(300)) * self.per_km
        for g in CORAL_GROUPS:
            self.cyclone_mort[g][:] = 0.0

        if self.rng.random_int(100) < 35:
            self.cyclone_category = 2
            self.cyclone_radius = smaller
            self.cyclone_centre = self.rng.random_int(self.number_of_reefs)
            self.cyclone_mortality()

        if self.year > 1975 and self.year < self.cfg.projection_year:
            historical: dict[int, tuple[float, int, tuple[float, float]]] = {
                1976: (smaller, 3, (-23, -22)),
                1980: (smaller, 4, (-23, -22)),
                1986: (medium, 3, (-18, -17)),
                1989: (larger, 3, (-20, -19)),
                1990: (medium, 3, (-15, -14)),
                1991: (larger, 4, (-18, -17)),
                1997: (larger, 3, (-19, -18)),
                1998: (larger, 3, (-16, -15)),
                2005: (smaller, 5, (-14, -13)),
                2006: (medium, 3, (-18, -17)),
                2007: (smaller, 3, (-14, -13)),
                2009: (larger, 4, (-21, -20)),
                2010: (smaller, 3, (-20, -19)),
                2011: (larger, 5, (-18, -17)),
                2014: (medium, 3, (-15, -14)),
                2015: (smaller, 3, (-22, -20)),
                2017: (smaller, 3, (-20, -19)),
                2019: (smaller, 3, (-16, -15)),
            }
            if self.year in historical:
                radius, cat, lat_band = historical[self.year]
                candidates = np.flatnonzero((self.y > lat_band[0]) & (self.y < lat_band[1]))
                if candidates.size > 0:
                    self.cyclone_radius = radius
                    self.cyclone_category = cat
                    self.cyclone_centre = int(candidates[self.rng.random_int(candidates.size)])
                    self.cyclone_mortality()
                else:
                    import warnings
                    warnings.warn(
                        f"Historical cyclone event {self.year}: empty latitude band "
                        f"({lat_band[0]}, {lat_band[1]}), no reefs found"
                    )
        else:
            if self.rng.random_int(100) < 21:
                self.cyclone_radius = (200 + self.rng.random_int(500)) * self.per_km
                self.cyclone_category = 3
                self.cyclone_centre = self.rng.random_int(self.number_of_reefs)
                self.cyclone_mortality()
            if self.rng.random_int(100) < 4:
                self.cyclone_radius = (200 + self.rng.random_int(600)) * self.per_km
                self.cyclone_category = 4
                self.cyclone_centre = self.rng.random_int(self.number_of_reefs)
                self.cyclone_mortality()
            if self.rng.random_int(100) < 2:
                self.cyclone_radius = (200 + self.rng.random_int(700)) * self.per_km
                self.cyclone_category = 5
                self.cyclone_centre = self.rng.random_int(self.number_of_reefs)
                self.cyclone_mortality()

        self.flood_load = 0.2 + 0.4 * (1 + 0.2 * self.cyclone_category - self.catchment_condition)

    def cyclone_mortality(self) -> None:
        self._seed_year()
        if self.cyclone_centre < 0:
            return
        d = self.dist_matrix[self.cyclone_centre]
        affected = np.flatnonzero(d <= self.cyclone_radius)
        
        # NetLogo uses random ask order for reefs
        affected_permuted = self.rng._rs.permutation(affected)
        
        for reef_idx in affected_permuted:
            radial = 1 - nl_median(
                0, ((d[reef_idx] / self.cyclone_radius) ** 2), 1
            )
            shelter = math.sqrt(
                self.reef_sites[reef_idx] * self.number_of_reefs / self.number_of_sites
            )
            sl = self._sites_for_reef(int(reef_idx))
            for s in range(sl.start, sl.stop):
                rand = (0.7 + self.rng.random_float(0.3)) * radial / shelter
                mort_max = rand * (0.25 * self.cyclone_category - 0.30)
                mort_min = rand * max(0.0, 0.3 * self.cyclone_category - 0.9)
                self.cyclone_mort["sa"][s] = (1.0 * mort_max + 0.0 * mort_min) * (
                    self.rate_i["sa"] / (self.rate["sa"][s] + self.small)
                )
                self.cyclone_mort["ta"][s] = (0.9 * mort_max + 0.1 * mort_min) * (
                    self.rate_i["ta"] / (self.rate["ta"][s] + self.small)
                )
                self.cyclone_mort["mo"][s] = (0.7 * mort_max + 0.3 * mort_min) * (
                    self.rate_i["mo"] / (self.rate["mo"][s] + self.small)
                )
                self.cyclone_mort["po"][s] = (0.1 * mort_max + 0.9 * mort_min) * (
                    self.rate_i["po"] / (self.rate["po"][s] + self.small)
                )
                self.cyclone_mort["fa"][s] = (0.0 * mort_max + 1.0 * mort_min) * (
                    self.rate_i["fa"] / (self.rate["fa"][s] + self.small)
                )
                self.cyclone_mort["tt"][s] = (1.0 * mort_max + 0.0 * mort_min) * (
                    self.rate_i["tt"] / (self.rate["tt"][s] + self.small)
                )
                self.R_site[s] = nl_median(
                    0.0,
                    1.0,
                    self.R_site[s]
                    + 2
                    * (
                        self.cyclone_mort["sa"][s] * self.C["sa"][s]
                        + self.cyclone_mort["ta"][s] * self.C["ta"][s]
                        + self.cyclone_mort["mo"][s] * self.C["mo"][s]
                        + self.cyclone_mort["po"][s] * self.C["po"][s]
                        + self.cyclone_mort["fa"][s] * self.C["fa"][s]
                        + self.cyclone_mort["tt"][s] * self.C["tt"][s]
                    ),
                )
                for g in CORAL_GROUPS:
                    self.C[g][s] *= max(self.small, 1 - self.cyclone_mort[g][s])

    def reef_populations(self) -> None:
        for reef_idx in range(self.number_of_reefs):
            sl = self._sites_for_reef(reef_idx)
            self.B_r[reef_idx] = nl_round(float(np.mean(self.B[sl])))
            self.T_r[reef_idx] = nl_round(float(np.mean(self.T[sl])))

            for age in ("2", "3", "4", "5", "6"):
                self.S_r[age][reef_idx] = nl_round(float(np.mean(self.S[age][sl])))
            self.S_r["1"][reef_idx] = nl_round(
                min(
                    self.S_r["1"][reef_idx],
                    2.77
                    * (
                        self.S_r["2"][reef_idx]
                        + self.S_r["3"][reef_idx]
                        + self.S_r["4"][reef_idx]
                        + self.S_r["5"][reef_idx]
                        + self.S_r["6"][reef_idx]
                    ),
                )
            )
            self.S_manta_r[reef_idx] = nl_round(
                0.50 * self.S_r["2"][reef_idx]
                + 0.70 * self.S_r["3"][reef_idx]
                + 0.85 * self.S_r["4"][reef_idx]
                + 0.95 * self.S_r["5"][reef_idx]
                + 0.99 * self.S_r["6"][reef_idx]
            )

            for g in CORAL_GROUPS:
                self.C_r[g][reef_idx] = float(np.mean(self.C[g][sl]))
                self.bleach_mort_r[g][reef_idx] = float(np.mean(self.bleach_mort[g][sl]))
                self.cyclone_mort_r[g][reef_idx] = float(np.mean(self.cyclone_mort[g][sl]))
                self.predate_mort_r[g][reef_idx] = float(np.mean(self.predate_mort[g][sl]))
            self.C_reef[reef_idx] = float(np.mean(self.C_site[sl]))
            self.R_reef[reef_idx] = float(np.mean(self.R_site[sl]))

        if self.search_mode == 1 and self.year >= self.cfg.search_year:
            idx = self._reef_by_priority(1)
            if idx is not None:
                self.benefit[idx] += float(np.mean(self.C_site))

    # ---- intervention and loop helpers ----
    def _apply_interventions(self) -> None:
        if self.year >= self.cfg.start_CoTS_control:
            if self.cfg.CoTS_vessels_GBR > 0:
                self.control_region = "GBR"
                self.S_vessels = self.cfg.CoTS_vessels_GBR
                self.control_cots()
            if self.cfg.CoTS_vessels_FN > 0:
                self.control_region = "FN"
                self.S_vessels = self.cfg.CoTS_vessels_FN
                self.control_cots()
            if self.cfg.CoTS_vessels_N > 0:
                self.control_region = "N"
                self.S_vessels = self.cfg.CoTS_vessels_N
                self.control_cots()
            if self.cfg.CoTS_vessels_C > 0:
                self.control_region = "C"
                self.S_vessels = self.cfg.CoTS_vessels_C
                self.control_cots()
            if self.cfg.CoTS_vessels_S > 0:
                self.control_region = "S"
                self.S_vessels = self.cfg.CoTS_vessels_S
                self.control_cots()
            if self.cfg.CoTS_vessels_sector > 0:
                self.S_vessels = self.cfg.CoTS_vessels_sector
                self.control_cots_by_sector()

        if (
            np.count_nonzero((self.year < self.rezone_year) & (self.year < self.future_rezone_year))
            > 0
        ):
            self.apply_fishing()

        if self.year >= self.cfg.start_catchment_restore and self.cfg.restore_timeframe > 0:
            self.catchment_condition += (1 - self.catchment_condition) / self.cfg.restore_timeframe
        if self.year >= self.cfg.start_rubble_consolidation:
            self.consolidate_rubble()
        if self.year >= self.cfg.start_coral_seeding:
            self.seed_tt_coral()
        if self.year >= self.cfg.start_coral_slick:
            self.seed_coral_slick()
        if self.year >= self.cfg.start_emperor_release:
            self.release_fish()
        if self.year >= self.cfg.start_reef_shading:
            self.shade_local_reef()
        if self.year >= self.cfg.start_regional_shading:
            self.shade_regional_coral()
        if self.year >= self.cfg.start_pH_protection:
            self.increase_pH()

        if self.year >= 2026:
            self._apply_perfect_intervention()

    def _apply_perfect_intervention(self) -> None:
        mode = self.cfg.perfect_intervention
        if mode == "Starfish-control":
            for reef_idx in np.flatnonzero(self.priority <= 100 * self.ensemble):
                sl = self._sites_for_reef(int(reef_idx))
                for age in ("2", "3", "4", "5", "6"):
                    self.S[age][sl] = 0
        if mode == "Coral-replenishment":
            for reef_idx in np.flatnonzero(
                (self.priority <= 100 * self.ensemble) & (self.C_reef < 0.2)
            ):
                sl = self._sites_for_reef(int(reef_idx))
                idxs = np.flatnonzero(self.C_site[sl] < 0.2) + sl.start
                for s in idxs:
                    self.C["sa"][s] += 0.01
                    self.C["ta"][s] += 0.01
                    self.C["mo"][s] += 0.01
                    self.C["po"][s] += 0.01
                    self.C["fa"][s] += 0.01
        if mode == "Coral-enhancement":
            for reef_idx in np.flatnonzero(
                (self.priority <= 100 * self.ensemble) & (self.C_reef < 0.2)
            ):
                sl = self._sites_for_reef(int(reef_idx))
                idxs = np.flatnonzero(self.C_site[sl] < 0.2) + sl.start
                self.C["tt"][idxs] += 0.01
        if mode == "Rubble-stabilisation":
            for reef_idx in np.flatnonzero(self.priority <= 100 * self.ensemble):
                sl = self._sites_for_reef(int(reef_idx))
                self.R_site[sl] = 0
        if mode == "Coral-shading":
            self.reef_shading[self.priority <= 100 * self.ensemble] = 1.0
        if mode == "Fish-protection":
            self.future_rezone_year[self.priority <= 100 * self.ensemble] = 2026
        if mode == "Control-plus-replenishment":
            for reef_idx in np.flatnonzero(self.priority <= 100 * self.ensemble):
                sl = self._sites_for_reef(int(reef_idx))
                for age in ("2", "3", "4", "5", "6"):
                    self.S[age][sl] = 0
                if self.C_reef[reef_idx] < 0.2:
                    idxs = np.flatnonzero(self.C_site[sl] < 0.2) + sl.start
                    for s in idxs:
                        self.C["sa"][s] += 0.01
                        self.C["ta"][s] += 0.01
                        self.C["mo"][s] += 0.01
                        self.C["po"][s] += 0.01
                        self.C["fa"][s] += 0.01
        if mode == "Control-plus-shading":
            for reef_idx in np.flatnonzero(self.priority <= 100 * self.ensemble):
                self.reef_shading[reef_idx] = 1.0
                sl = self._sites_for_reef(int(reef_idx))
                for age in ("2", "3", "4", "5", "6"):
                    self.S[age][sl] = 0
        if mode == "Replenishment-plus-shading":
            for reef_idx in np.flatnonzero(
                (self.priority <= 100 * self.ensemble) & (self.C_reef < 0.2)
            ):
                self.reef_shading[reef_idx] = 1.0
                sl = self._sites_for_reef(int(reef_idx))
                idxs = np.flatnonzero(self.C_site[sl] < 0.2) + sl.start
                for s in idxs:
                    self.C["sa"][s] += 0.01
                    self.C["ta"][s] += 0.01
                    self.C["mo"][s] += 0.01
                    self.C["po"][s] += 0.01
                    self.C["fa"][s] += 0.01
        if mode == "Control-plus-replenishment-plus-shading":
            for reef_idx in np.flatnonzero(self.priority <= 100 * self.ensemble):
                sl = self._sites_for_reef(int(reef_idx))
                for age in ("2", "3", "4", "5", "6"):
                    self.S[age][sl] = 0
                self.reef_shading[reef_idx] = 1.0
                if self.C_reef[reef_idx] < 0.2:
                    idxs = np.flatnonzero(self.C_site[sl] < 0.2) + sl.start
                    for s in idxs:
                        self.C["sa"][s] += 0.01
                        self.C["ta"][s] += 0.01
                        self.C["mo"][s] += 0.01
                        self.C["po"][s] += 0.01
                        self.C["fa"][s] += 0.01

    def _reef_by_priority(self, p: int) -> int | None:
        idxs = np.flatnonzero(self.priority == p)
        if idxs.size == 0:
            return None
        return int(idxs[0])

    def _reef_in_intervention_bounds(self, reef_idx: int) -> bool:
        return (
            self.x[reef_idx] > self.cfg.intervene_lon_min
            and self.x[reef_idx] < self.cfg.intervene_lon_max
            and self.y[reef_idx] > self.cfg.intervene_lat_min
            and self.y[reef_idx] < self.cfg.intervene_lat_max
        )

    def _projection_bleach_probability(self) -> float:
        if self.cfg.SSP == 1.9:
            return 0.39 - 0.00007 * (self.year - 2070) ** 2
        if self.cfg.SSP == 2.6:
            return 0.47 - 0.00011 * (self.year - 2070) ** 2
        if self.cfg.SSP == 4.5:
            return 0.70 - 0.00011 * (self.year - 2090) ** 2
        if self.cfg.SSP == 7.0:
            return 0.98 - 0.00013 * (self.year - 2100) ** 2
        if self.cfg.SSP == 8.5:
            return 0.98 - 0.00017 * (self.year - 2090) ** 2
        return 0.0

    # ---- output ----
    def _set_up_output_files(self) -> None:
        self.rng.seed(1)
        if self.output_file.exists():
            self.output_file.unlink()
        lines: list[str] = []
        lines.append(f"Climate scenario, {self.cfg.SSP}")
        lines.append("")
        lines.append(f"Ensemble runs, {self.cfg.ensemble_runs}")
        lines.append(f"Start year, {self.cfg.start_year}")
        lines.append(f"Spinup backtrack (years), {self.cfg.spinup_backtrack_years}")
        lines.append(f"Save year, {self.cfg.save_year}")
        lines.append(f"Projection year, {self.cfg.projection_year}")
        lines.append(f"End year, {self.cfg.end_year}")
        lines.append(f"Search year, {self.cfg.search_year}")
        lines.append("")
        lines.append(f"CoTS control start year, {self.cfg.start_CoTS_control}")
        lines.append(f"CoTS control ecological threshold (CoTS per ha), {self.cfg.eco_threshold}")
        lines.append(f"CoTS control CoTS threshold (CoTS per ha), {self.cfg.CoTS_threshold}")
        lines.append(f"CoTS control coral threshold (CoTS per ha), {self.cfg.coral_threshold}")
        lines.append(f"CoTS vessels across GBR, {self.cfg.CoTS_vessels_GBR}")
        lines.append(f"CoTS vessels in Far-northern Region, {self.cfg.CoTS_vessels_FN}")
        lines.append(f"CoTS vessels in Northern Region, {self.cfg.CoTS_vessels_N}")
        lines.append(f"CoTS vessels in Central Region, {self.cfg.CoTS_vessels_C}")
        lines.append(f"CoTS vessels in Southern Region, {self.cfg.CoTS_vessels_S}")
        lines.append(f"Catchment restoration start year, {self.cfg.start_catchment_restore}")
        lines.append(f"Catchment restoration timescale (years), {self.cfg.restore_timeframe}")
        lines.append(f"Future zoning start year, {self.cfg.start_modified_zoning}")
        lines.append(f"Number of reefs included in future rezoning, {self.cfg.rezoned_reefs}")
        lines.append("")
        lines.append(f"Reduction in fisheries catch start year, {self.cfg.start_modified_fishing}")
        lines.append(f" Fractional reduction in fisheries catches, {self.cfg.catch_reduction}")
        lines.append("")
        lines.append(f"Upper fish size limit start year, {self.cfg.start_upper_sizelimit}")
        lines.append(f"Lower fish size limit start year, {self.cfg.start_lower_sizelimit}")
        lines.append(
            f"Exclude fishing from active outbreak reefs start year, {self.cfg.start_CoTSlimit}"
        )
        lines.append("")
        lines.append(f"Emperor release start year, {self.cfg.start_emperor_release}")
        lines.append(f"Number of release reefs, {self.cfg.release_reefs}")
        lines.append(f"Maximum adult emperors (per ha) for release, {self.cfg.release_threshold}")
        lines.append(f"Number of juvenile emperors released per reef, {self.cfg.release_number}")
        lines.append("")
        lines.append(f"Regional shading start year, {self.cfg.start_regional_shading}")
        lines.append(
            f"Absolute DHW reduction due to regional shading (DHW), {self.cfg.regional_shading_reduction}"
        )
        lines.append("")
        lines.append(f"Minimum longitude of interventions, {self.cfg.intervene_lon_min}")
        lines.append(f"Maximum longitude of interventions, {self.cfg.intervene_lon_max}")
        lines.append(f"Minimum latitude of interventions, {self.cfg.intervene_lat_min}")
        lines.append(f"Maximum latitude of interventions, {self.cfg.intervene_lat_max}")
        lines.append("")
        lines.append(f"Rubble consolidation start year, {self.cfg.start_rubble_consolidation}")
        lines.append(f"Annual number of consolidated reefs, {self.cfg.consolidation_reefs}")
        lines.append(
            f"Minimum rubble cover threshold for consolidation [0 1], {self.cfg.consolidation_threshold}"
        )
        lines.append(f"Total annual consolidated area (ha) , {self.cfg.consolidation_hectares}")
        lines.append("")
        lines.append(f"Thermally tolerant coral seeding start year, {self.cfg.start_coral_seeding}")
        lines.append(f"Annual number of reefs seeded with coral, {self.cfg.seed_reefs}")
        lines.append(
            f"Maximum coral cover threshold for coral seeding [0 1], {self.cfg.seed_threshold}"
        )
        lines.append(f"Total annual area of seeded corals (ha), {self.cfg.seed_hectares}")
        lines.append(
            "Fraction of staghorn acropora corals able to hybridise with thermally tolerant corals [0 1], "
            f"{self.cfg.hybrid_fraction}"
        )
        lines.append(
            "Dominance of thermally tolerant corals in setting thermal tolerance of hybrids [0 1], "
            f"{self.cfg.dominance}"
        )
        lines.append("")
        lines.append(f"Coral slicks start year, {self.cfg.start_coral_slick}")
        lines.append(f"Annual number of reefs with coral slicks released, {self.cfg.slick_reefs}")
        lines.append(
            f"Maximum coral cover threshold for coral slicks [0 1], {self.cfg.slick_threshold}"
        )
        lines.append(f"Total annual area of slick corals (ha), {self.cfg.slick_hectares}")
        lines.append("")
        lines.append(f"Reef shading start year, {self.cfg.start_reef_shading}")
        lines.append(f"Annual number of reefs locally shaded, {self.cfg.shading_reefs}")
        lines.append(
            f"Fractional DHW reduction due to local shading [0 1], {self.cfg.reef_shading_reduction}"
        )
        lines.append("")
        lines.append(f"Ocean acidification treatment start year, {self.cfg.start_pH_protection}")
        lines.append(f"Annual number of reefs treated for ocean acidification, {self.cfg.pH_reefs}")
        lines.append(
            f"Fractional protection from ocean acidification [0 1], {self.cfg.pH_protection}"
        )
        lines.append("")
        lines.append(f"CoTS vessels in active sector, {self.cfg.CoTS_vessels_sector}")
        lines.append("")
        lines.append(
            "Ensemble, Year, Reef_ID, Region, Shelf_position, Rezone_year, Priority, Longitude, Latitude, km_offshore, Reef_sites,"
            "C_sa, C_ta, C_mo, C_po, C_fa, C_tt, C_out_degree, DHW, bleach_sa, bleach_ta, bleach_mo, bleach_po, bleach_fa, bleach_tt, "
            "Maximum_cyclone_category, cyclone_sa, cyclone_ta, cyclone_mo, cyclone_po, cyclone_fa, cyclone_tt, "
            "predate_sa, predate_ta, predate_mo, predate_po, predate_fa, predate_tt, "
            "S_1, S_2, S_3, S_4, S_5, S_6, S_manta, Control_dives, Benthic_invert, Triggerfish, "
            "E_1, E_2, E_3, E_4, E_5, E_catch_kg,G_1, G_2, G_3, G_4, G_5, G_catch_kg"
        )
        self.output_file.write_text("\n".join(lines) + "\n")
        logger.debug("Initialized output file header at %s.", self.output_file)

    def write_output(self) -> None:
        self.rng.seed(1)
        lines = []
        for i in range(self.number_of_reefs):
            row: Iterable[object] = (
                self.ensemble,
                self.year,
                self.reef_id[i],
                self.region_name[i],
                self.shelf_position[i],
                self.rezone_year[i],
                self.priority_category[i],
                self.x[i],
                self.y[i],
                self.km_offshore[i],
                self.reef_sites[i],
                self.C_r["sa"][i],
                self.C_r["ta"][i],
                self.C_r["mo"][i],
                self.C_r["po"][i],
                self.C_r["fa"][i],
                self.C_r["tt"][i],
                self.C_out_degree[i],
                self.dhw[i],
                self.bleach_mort_r["sa"][i],
                self.bleach_mort_r["ta"][i],
                self.bleach_mort_r["mo"][i],
                self.bleach_mort_r["po"][i],
                self.bleach_mort_r["fa"][i],
                self.bleach_mort_r["tt"][i],
                self.cyclone_category,
                self.cyclone_mort_r["sa"][i],
                self.cyclone_mort_r["ta"][i],
                self.cyclone_mort_r["mo"][i],
                self.cyclone_mort_r["po"][i],
                self.cyclone_mort_r["fa"][i],
                self.cyclone_mort_r["tt"][i],
                self.predate_mort_r["sa"][i],
                self.predate_mort_r["ta"][i],
                self.predate_mort_r["mo"][i],
                self.predate_mort_r["po"][i],
                self.predate_mort_r["fa"][i],
                self.predate_mort_r["tt"][i],
                self.S_r["1"][i],
                self.S_r["2"][i],
                self.S_r["3"][i],
                self.S_r["4"][i],
                self.S_r["5"][i],
                self.S_r["6"][i],
                self.S_manta_r[i],
                self.dives_reef[i],
                self.B_r[i],
                self.T_r[i],
                self.E["1"][i],
                self.E["2"][i],
                self.E["3"][i],
                self.E["4"][i],
                self.E["5"][i],
                self.E_catch_kg[i],
                self.G["1"][i],
                self.G["2"][i],
                self.G["3"][i],
                self.G["4"][i],
                self.G["5"][i],
                self.G_catch_kg[i],
            )
            lines.append(",".join(str(x) for x in row))
        with self.output_file.open("a") as f:
            f.write("\n".join(lines) + "\n")
        logger.debug(
            "Appended %s reef rows to output (ensemble=%s year=%s).",
            len(lines),
            self.ensemble,
            self.year,
        )

    def _write_priority_benefit(self, path: Path | None = None) -> None:
        out = path if path is not None else Path("priority_reef_benefit.csv")
        with out.open("a") as f:
            idx = self._reef_by_priority(1)
            if idx is not None:
                f.write(f"{self.reef_id[idx]}, {self.x[idx]}, {self.y[idx]}, {self.benefit[idx]}\n")
                logger.debug(
                    "Appended priority benefit record for reef_id=%s to %s.",
                    self.reef_id[idx],
                    out,
                )


def _run_simulation_ensemble_worker(
    cfg: CoconetConfig,
    checkpoint: SpinupCheckpoint,
    ensemble_id: int,
    output_part: Path,
    priority_part: Path | None,
) -> None:
    """Child process: build model, restore spinup checkpoint, run one simulation ensemble."""
    configure_logging(cfg.log_level)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        model = CoconetModel(cfg)
        model.setup(init_output=False)
        model.import_spinup_checkpoint(checkpoint)
        model.ensemble = ensemble_id
        model.output_file = output_part
        model.initialise_run()
        model._run_ensemble_year_steps()
        model._maybe_write_priority_benefit(priority_part)
