import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

from src.utils import get_root, run_from_id

logger = logging.getLogger(__name__)


def _resolve_path(path: str | Path | None, default_relative_dir: str = "logs") -> Path:
    """Resolves a path relative to `<project_root>/<default_relative_dir>` if relative."""
    root = Path(get_root())
    if path is None:
        return root / default_relative_dir if default_relative_dir else root
    p = Path(path)
    if p.is_absolute():
        return p
    return root / default_relative_dir / p if default_relative_dir else root / p


def _serialize_summary(summary: Any) -> dict[str, Any]:
    """Converts a WandB run summary into a JSON-serializable dictionary."""
    if summary is None:
        return {}

    items = []
    if hasattr(summary, "items"):
        try:
            items = list(summary.items())
        except Exception:
            items = []
    elif hasattr(summary, "_json_dict"):
        items = list(summary._json_dict.items())

    serialized: dict[str, Any] = {}
    for k, v in items:
        if isinstance(v, (int, float, str, bool)) or v is None:
            serialized[str(k)] = v
        elif isinstance(v, (np.integer, np.floating)):
            serialized[str(k)] = v.item()
        else:
            try:
                serialized[str(k)] = float(v)
            except (ValueError, TypeError):
                pass
    return serialized


def _get_run_data(
    run_id: str,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetches run data (name and summary metrics) with local JSON caching."""
    resolved_cache_dir = cache_dir or (Path(get_root()) / "logs" / "wandb_cache")
    run_id_clean = str(run_id).replace("/", "_")
    cache_file = resolved_cache_dir / f"{run_id_clean}.json"

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache file '{cache_file}': {e}")

    # Fetch run from WandB API
    run = None
    run_id_str = str(run_id)
    try:
        if "/" in run_id_str:
            run = wandb.Api().run(run_id_str)
        else:
            try:
                run = run_from_id(run_id_str)
            except Exception:
                run = wandb.Api().run(run_id_str)
    except Exception as e:
        logger.warning(f"Could not retrieve WandB run for ID '{run_id_str}': {e}")
        run = None

    if run is None:
        data = {"exists": False, "name": None, "summary": {}}
    else:
        run_name = getattr(run, "name", None)
        if type(run_name).__name__ == "MagicMock":
            run_name = None
        data = {
            "exists": True,
            "name": run_name,
            "summary": _serialize_summary(getattr(run, "summary", None)),
        }

    if use_cache:
        try:
            resolved_cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write cache file '{cache_file}': {e}")

    return data


def extract_metrics_to_csv(
    run_ids: list[str],
    metric_names: list[str],
    output_csv: str | Path,
    run_aliases: list[str] | None = None,
    metric_aliases: list[str] | None = None,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Extracts metrics from a list of WandB run IDs and saves them to a CSV file.

    Args:
        run_ids: List of WandB run IDs (or path 'entity/project/run_id').
        metric_names: List of metric names to extract.
        output_csv: Path to save the output CSV. If relative, saved under `<project_root>/logs/`.
        run_aliases: Optional list of alias names for runs, matching length of `run_ids`.
        metric_aliases: Optional list of column names for metrics, matching length of `metric_names`.
        use_cache: Whether to cache run metrics to disk (default: True).
        cache_dir: Directory for cached metrics (default: `<project_root>/logs/wandb_cache`).

    Returns:
        pd.DataFrame: DataFrame containing the extracted metrics and saved to CSV.
    """
    run_ids = list(run_ids)
    metric_names = list(metric_names)

    if metric_aliases is not None:
        metric_aliases = list(metric_aliases)
        if len(metric_aliases) != len(metric_names):
            raise ValueError(
                f"Length of metric_aliases ({len(metric_aliases)}) must match length of metric_names ({len(metric_names)})."
            )
        col_names = metric_aliases
    else:
        col_names = metric_names

    if run_aliases is not None:
        run_aliases = list(run_aliases)
        if len(run_aliases) != len(run_ids):
            raise ValueError(
                f"Length of run_aliases ({len(run_aliases)}) must match length of run_ids ({len(run_ids)})."
            )

    resolved_cache_dir = (
        _resolve_path(cache_dir, default_relative_dir="logs/wandb_cache")
        if cache_dir is not None
        else (Path(get_root()) / "logs" / "wandb_cache")
    )

    rows: list[dict[str, Any]] = []
    for i, run_id in enumerate(tqdm(run_ids, desc="Extracting metrics")):
        run_data = _get_run_data(run_id, use_cache=use_cache, cache_dir=resolved_cache_dir)
        alias_val = run_aliases[i] if run_aliases is not None else run_data.get("name")
        row: dict[str, Any] = {"id": run_id, "alias": alias_val if alias_val is not None else np.nan}

        summary = run_data.get("summary", {}) if run_data.get("exists", False) else {}
        for metric, col_name in zip(metric_names, col_names):
            val = summary.get(metric, np.nan)
            row[col_name] = val if val is not None else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)[["id", "alias"] + col_names]

    output_path = _resolve_path(output_csv, default_relative_dir="logs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Extracted metrics for {len(run_ids)} runs to '{output_path}'.")

    return df


def _format_value(val: Any, decimals: int) -> str:
    """Formats a single value into a string with given decimal precision."""
    if pd.isna(val):
        return "NaN"
    if isinstance(val, (int, float, np.number)):
        return f"{float(val):.{decimals}f}"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _format_mean_std(mean: Any, std: Any, decimals: int) -> str:
    """Formats a mean and std pair into 'mean ± std' string."""
    m_valid, s_valid = pd.notna(mean), pd.notna(std)
    if m_valid and s_valid:
        return f"{_format_value(mean, decimals)} ± {_format_value(std, decimals)}"
    if m_valid:
        return _format_value(mean, decimals)
    if s_valid:
        return f"NaN ± {_format_value(std, decimals)}"
    return "NaN"


def csv_to_markdown_table(
    csv_file: str | Path | pd.DataFrame,
    output_md: str | Path | None = None,
    decimals: int = 2,
) -> str:
    """
    Generates a Markdown table from a metrics CSV file (or DataFrame).

    If a metric occurs twice, once with `<metric_name>/mean` and once with `<metric_name>/std`,
    they are merged into a single `<metric_name>` column formatted as `mean ± std`.
    The `id` column is ignored.

    Args:
        csv_file: Path to the CSV file (or a pandas DataFrame).
        output_md: Optional path to save the markdown file. If relative, saved under `<project_root>/logs/`.
        decimals: Number of decimal places for numerical values (default: 2).

    Returns:
        str: The formatted Markdown table string.
    """
    if isinstance(csv_file, pd.DataFrame):
        df = csv_file.copy()
    else:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            csv_path = _resolve_path(csv_file, default_relative_dir="logs")
            if not csv_path.exists():
                csv_path = _resolve_path(csv_file, default_relative_dir="")
        df = pd.read_csv(csv_path)

    columns = list(df.columns)
    mean_suffix, std_suffix = "/mean", "/std"

    # Identify paired metrics: <metric>/mean and <metric>/std
    paired_bases = {
        col[: -len(mean_suffix)]
        for col in columns
        if col.endswith(mean_suffix) and f"{col[:-len(mean_suffix)]}{std_suffix}" in columns
    }

    new_data: dict[str, list[str]] = {}
    processed_cols: set[str] = set()

    for col in columns:
        if col in processed_cols or col == "id":
            continue

        if col.endswith(mean_suffix):
            base = col[: -len(mean_suffix)]
            if base in paired_bases:
                mean_col, std_col = f"{base}{mean_suffix}", f"{base}{std_suffix}"
                processed_cols.update([mean_col, std_col])
                new_data[base] = [_format_mean_std(m, s, decimals) for m, s in zip(df[mean_col], df[std_col])]
                continue

        if col.endswith(std_suffix) and col[: -len(std_suffix)] in paired_bases:
            continue

        if col == "alias":
            new_data[col] = ["" if pd.isna(v) else str(v) for v in df[col]]
        else:
            new_data[col] = [_format_value(v, decimals) for v in df[col]]
        processed_cols.add(col)

    formatted_df = pd.DataFrame(new_data)
    markdown_table = formatted_df.to_markdown(index=False)

    if output_md is not None:
        out_path = _resolve_path(output_md, default_relative_dir="logs")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown_table)
        logger.info(f"Markdown table saved to '{out_path}'.")

    return markdown_table
