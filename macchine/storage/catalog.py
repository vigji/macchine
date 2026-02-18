"""DuckDB catalog and fleet registry loading."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from macchine.config import FLEET_REGISTRY_FILE, MERGED_TRACE_INDEX_FILE, METADATA_DIR, TRACE_INDEX_FILE
from macchine.models.core import Machine, Site
from macchine.models.fleet import FleetAssignment, FleetRegistry


def get_trace_index(output_dir: Path) -> pd.DataFrame:
    """Load the trace index as a DataFrame."""
    index_path = output_dir / METADATA_DIR / TRACE_INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(f"Trace index not found at {index_path}. Run 'macchine convert' first.")
    return pd.read_parquet(index_path)


def get_merged_trace_index(output_dir: Path) -> pd.DataFrame:
    """Load the merged (deduplicated + session-merged) trace index as a DataFrame."""
    index_path = output_dir / METADATA_DIR / MERGED_TRACE_INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(
            f"Merged trace index not found at {index_path}. "
            "Run 'python scripts/build_merged_index.py' first."
        )
    df = pd.read_parquet(index_path)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    return df


def _classify_pile_continuity(technique: str, span_hours: float, n_sessions: int) -> str:
    """Classify whether a multi-session pile is genuine continuation or reused name.

    Based on empirical analysis of 504 multi-session piles:
    - GRAB/CUT: typically worked across hours-to-days; genuine up to ~2 weeks
    - KELLY/SCM: element names often reused across months at the same site
    - SOB: short pours; spans > 3 days are suspicious

    Returns one of: "single_session", "genuine", "ambiguous", "reused_name".
    """
    if n_sessions <= 1:
        return "single_session"

    span_days = span_hours / 24

    if technique in ("GRAB", "CUT"):
        if span_days <= 14:
            return "genuine"
        elif span_days <= 30:
            return "ambiguous"
        else:
            return "reused_name"
    elif technique in ("KELLY", "SCM"):
        if span_days <= 3:
            return "genuine"
        elif span_days <= 14:
            return "ambiguous"
        else:
            return "reused_name"
    elif technique == "SOB":
        if span_days <= 3:
            return "genuine"
        elif span_days <= 7:
            return "ambiguous"
        else:
            return "reused_name"
    else:
        # Unknown technique: conservative
        if span_days <= 7:
            return "genuine"
        else:
            return "ambiguous"


def get_pile_index(output_dir: Path) -> pd.DataFrame:
    """Build a pile-level index by grouping all sessions for each (site_id, element_name).

    Unlike the merged trace index (which only joins traces within 5 minutes),
    this groups ALL sessions that belong to the same physical pile, regardless
    of time gaps. A single GRAB panel or CUT panel may be worked on across
    multiple days — this index captures that.

    Each pile is classified for continuity safety:
    - ``single_session``: only one merged session, always safe
    - ``genuine``: multi-session, span consistent with real pile work
    - ``ambiguous``: span is borderline; inspect before concatenating
    - ``reused_name``: element name likely reused for different piles

    Returns a DataFrame with one row per pile and columns:
        site_id, element_name, machine_slug, technique, operator(s),
        n_sessions, total_duration_s, total_samples, first_start, last_start,
        span_hours, all_trace_ids (pipe-separated list of every trace_id),
        pile_continuity (classification string).
    """
    df = get_merged_trace_index(output_dir)
    named = df[
        df["element_name"].notna()
        & (df["element_name"] != "")
        & (df["element_name"] != "xxxxx")
    ].copy()

    if named.empty:
        return pd.DataFrame()

    piles = (
        named.sort_values("start_time")
        .groupby(["site_id", "element_name"])
        .agg(
            machine_slug=("machine_slug", "first"),
            technique=("technique", "first"),
            operators=("operator", lambda x: ", ".join(sorted(set(o for o in x if o)))),
            n_sessions=("trace_id", "count"),
            total_duration_s=("duration_s", "sum"),
            total_samples=("sample_count", "sum"),
            first_start=("start_time", "min"),
            last_start=("start_time", "max"),
            sensor_count=("sensor_count", "max"),
            all_trace_ids=("trace_ids", lambda x: "|".join(x)),
        )
        .reset_index()
    )
    piles["span_hours"] = (
        (piles["last_start"] - piles["first_start"]).dt.total_seconds() / 3600
    )
    piles["pile_continuity"] = piles.apply(
        lambda r: _classify_pile_continuity(r["technique"], r["span_hours"], r["n_sessions"]),
        axis=1,
    )
    return piles


def load_pile_traces(
    output_dir: Path,
    site_id: str,
    element_name: str,
    clean: bool = True,
    max_gap_days: float | None = None,
) -> pd.DataFrame | None:
    """Load and concatenate all trace parquet files for a single pile.

    Resolves every individual trace_id across all sessions for this
    (site_id, element_name), loads their parquets, concatenates them in
    chronological order, and optionally cleans sentinel values.

    Parameters
    ----------
    output_dir : Path to output directory.
    site_id : Site identifier.
    element_name : Pile / element name.
    clean : If True, apply sentinel cleaning.
    max_gap_days : If set, only include sessions that form a contiguous
        chain where each consecutive pair is separated by at most this
        many days.  The longest such contiguous chain starting from the
        first session is used.  This prevents accidental concatenation
        of piles whose element name was reused months apart.

    Returns a single DataFrame with all sensor data for the pile, or None
    if no data could be loaded.
    """
    from macchine.harmonize.calibration import clean_sentinels_df

    df = get_merged_trace_index(output_dir)
    pile_sessions = df[
        (df["site_id"] == site_id) & (df["element_name"] == element_name)
    ].sort_values("start_time")

    if pile_sessions.empty:
        return None

    # Apply max_gap_days filter: keep the longest contiguous chain from the start
    if max_gap_days is not None and len(pile_sessions) > 1:
        times = pile_sessions["start_time"].tolist()
        keep = [0]  # always keep first session
        for i in range(1, len(times)):
            gap = (times[i] - times[i - 1]).total_seconds() / 86400
            if gap <= max_gap_days:
                keep.append(i)
            else:
                break  # stop at first gap that's too large
        pile_sessions = pile_sessions.iloc[keep]

    # Collect all individual trace_ids across all sessions
    all_trace_ids = []
    for _, row in pile_sessions.iterrows():
        ids = row["trace_ids"].split("|") if isinstance(row["trace_ids"], str) else [row["trace_id"]]
        slug = row["machine_slug"] if row["machine_slug"] != "unidentified" else "unknown"
        for tid in ids:
            all_trace_ids.append((tid, site_id, slug))

    # Load each trace parquet
    frames = []
    traces_dir = output_dir / "traces"
    for trace_id, sid, slug in all_trace_ids:
        path = traces_dir / str(sid) / slug / f"{trace_id}.parquet"
        if not path.exists():
            continue
        try:
            tdf = pd.read_parquet(path)
            tdf["_trace_id"] = trace_id
            frames.append(tdf)
        except Exception:
            continue

    if not frames:
        return None

    result = pd.concat(frames, ignore_index=True)

    # Sort by timestamp if available
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
        result = result.sort_values("timestamp").reset_index(drop=True)

    if clean:
        result = clean_sentinels_df(result)

    return result


def get_duckdb_connection(output_dir: Path) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection with trace index loaded as a view."""
    index_path = output_dir / METADATA_DIR / TRACE_INDEX_FILE
    con = duckdb.connect()
    if index_path.exists():
        con.execute(f"CREATE VIEW traces AS SELECT * FROM read_parquet('{index_path}')")
    return con


def load_fleet_registry(output_dir: Path) -> FleetRegistry:
    """Load fleet registry from JSON file."""
    fleet_path = output_dir / METADATA_DIR / FLEET_REGISTRY_FILE
    if not fleet_path.exists():
        raise FileNotFoundError(f"Fleet registry not found at {fleet_path}. Run 'macchine convert' first.")

    with open(fleet_path) as f:
        data = json.load(f)

    registry = FleetRegistry()

    for serial, minfo in data.get("machines", {}).items():
        m = Machine(
            serial_id=serial,
            model_name=minfo["model"],
            machine_number=minfo["number"],
        )
        registry.add_machine(m)

    for site_id, sinfo in data.get("sites", {}).items():
        s = Site(
            site_id=site_id,
            name=sinfo.get("name", site_id),
            raw_dir_name="",
            json_count=sinfo.get("json_count", 0),
            dat_count=sinfo.get("dat_count", 0),
            techniques=sinfo.get("techniques", []),
        )
        registry.add_site(s)

    for ainfo in data.get("assignments", []):
        machine = registry.machines.get(ainfo["machine_serial"])
        site = registry.sites.get(ainfo["site_id"])
        if machine and site:
            from datetime import datetime

            start = datetime.fromisoformat(ainfo["start_date"]) if ainfo.get("start_date") else None
            end = datetime.fromisoformat(ainfo["end_date"]) if ainfo.get("end_date") else None
            registry.add_assignment(FleetAssignment(
                machine=machine,
                site=site,
                start_date=start,
                end_date=end,
                trace_count=ainfo.get("trace_count", 0),
            ))

    return registry


def dataset_info(output_dir: Path) -> str:
    """Generate a dataset summary string."""
    try:
        df = get_trace_index(output_dir)
    except FileNotFoundError as e:
        return str(e)

    lines = [
        "Dataset Summary",
        "=" * 50,
        f"Total traces: {len(df)}",
        f"Formats: {df['format'].value_counts().to_dict()}",
    ]

    if "technique" in df.columns:
        tech_counts = df["technique"].value_counts().to_dict()
        lines.append(f"Techniques: {tech_counts}")

    if "site_id" in df.columns:
        sites = df["site_id"].nunique()
        lines.append(f"Sites: {sites}")

    if "machine_slug" in df.columns:
        machines = df["machine_slug"].nunique()
        lines.append(f"Machines: {machines}")

    if "operator" in df.columns:
        operators = df[df["operator"] != ""]["operator"].nunique()
        lines.append(f"Operators: {operators}")

    if "start_time" in df.columns:
        min_t = pd.to_datetime(df["start_time"]).min()
        max_t = pd.to_datetime(df["start_time"]).max()
        lines.append(f"Date range: {min_t} to {max_t}")

    if "sensor_count" in df.columns:
        total_sensors = df["sensor_count"].sum()
        lines.append(f"Total sensor series: {total_sensors}")

    if "sample_count" in df.columns:
        total_samples = df["sample_count"].sum()
        lines.append(f"Total samples: {total_samples:,}")

    if "duration_s" in df.columns:
        total_hours = df["duration_s"].sum() / 3600
        lines.append(f"Total recording time: {total_hours:.1f} hours")

    return "\n".join(lines)
