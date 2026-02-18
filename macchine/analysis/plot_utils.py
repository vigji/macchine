"""Plotting utilities for time-series analysis.

Provides gap-aware plotting and site-boundary marker helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def insert_gaps(x_dates: pd.Series, y_values: pd.Series, max_gap_days: int = 14):
    """Insert NaN into y_values where consecutive x_dates differ by more than max_gap_days.

    Returns (x_out, y_out) with NaN-separated segments so that matplotlib
    does not draw lines across inactive periods.
    """
    if len(x_dates) < 2:
        return x_dates, y_values

    x = pd.Series(x_dates).reset_index(drop=True)
    y = pd.Series(y_values).reset_index(drop=True)

    gaps = x.diff().dt.total_seconds() / 86400  # days
    gap_mask = gaps > max_gap_days

    if not gap_mask.any():
        return x, y

    # Build new arrays with NaN rows inserted at gap positions
    gap_indices = gap_mask[gap_mask].index.tolist()

    x_parts = []
    y_parts = []
    prev = 0
    for gi in gap_indices:
        x_parts.append(x.iloc[prev:gi])
        y_parts.append(y.iloc[prev:gi])
        # Insert a NaN separator
        mid_date = x.iloc[gi - 1] + (x.iloc[gi] - x.iloc[gi - 1]) / 2
        x_parts.append(pd.Series([mid_date]))
        y_parts.append(pd.Series([np.nan]))
        prev = gi
    x_parts.append(x.iloc[prev:])
    y_parts.append(y.iloc[prev:])

    return pd.concat(x_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True)


def plot_with_gaps(ax, x_dates, y_values, max_gap_days: int = 14, **kwargs):
    """Plot time series, inserting NaN where gaps > max_gap_days.

    This prevents matplotlib from drawing misleading lines across
    periods where the machine was inactive (relocation, maintenance).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x_dates : array-like of datetime
    y_values : array-like of float
    max_gap_days : int, default 14
        Gaps longer than this many days get a NaN break.
    **kwargs : passed to ax.plot()
    """
    x_gap, y_gap = insert_gaps(
        pd.Series(x_dates).reset_index(drop=True),
        pd.Series(y_values, dtype=float).reset_index(drop=True),
        max_gap_days=max_gap_days,
    )
    return ax.plot(x_gap, y_gap, **kwargs)


def add_site_markers(ax, df: pd.DataFrame, site_col: str = "site_id",
                     time_col: str = "start_time", alpha: float = 0.08):
    """Add vertical background shading to indicate site boundaries.

    Alternates between light blue and light orange for consecutive sites.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : DataFrame sorted by time_col with site_col column
    site_col : column name for site identifier
    time_col : column name for datetime
    alpha : transparency of background shading
    """
    if df.empty or site_col not in df.columns:
        return

    df = df.sort_values(time_col)
    sites = df[site_col].values
    times = pd.to_datetime(df[time_col]).values

    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c", "#1abc9c"]
    current_site = sites[0]
    start_time = times[0]
    site_idx = 0

    for i in range(1, len(sites)):
        if sites[i] != current_site:
            # Draw band for previous site
            ax.axvspan(start_time, times[i - 1], alpha=alpha,
                       color=colors[site_idx % len(colors)], zorder=0)
            # Add site label at the midpoint
            mid = start_time + (times[i - 1] - start_time) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.98, str(current_site),
                    ha="center", va="top", fontsize=6, alpha=0.5,
                    rotation=90)
            current_site = sites[i]
            start_time = times[i]
            site_idx += 1

    # Final site
    ax.axvspan(start_time, times[-1], alpha=alpha,
               color=colors[site_idx % len(colors)], zorder=0)
    mid = start_time + (times[-1] - start_time) / 2
    ax.text(mid, ax.get_ylim()[1] * 0.98, str(current_site),
            ha="center", va="top", fontsize=6, alpha=0.5,
            rotation=90)
