# Axis bounds and tick mark helpers for matplotlib plots.
#
# Two functions for different use cases:
#
#   get_axis_bounds_and_ticks(data, padding)
#       For linear-scale axes. Returns nice round tick marks.
#       Usage:
#           bounds, ticks = get_axis_bounds_and_ticks([min_y, max_y], padding=0.05)
#           ax.set_ylim(bounds)
#           ax.set_yticks(ticks)
#
#   get_axis_bounds_and_ticks_ln_pct(data, padding)
#       For axes where the data is in log-ratio space (i.e. ln(value/baseline))
#       but labels should display as percentage change. Returns tick positions
#       in log space and corresponding percentage labels.
#       Usage:
#           bounds, ticks, pcts = get_axis_bounds_and_ticks_ln_pct([min_y, max_y], padding=0.05)
#           ax.set_ylim(bounds)
#           ax.set_yticks(ticks)
#           ax.set_yticklabels([f'{p:g}%' for p in pcts])

import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor


def get_axis_bounds_and_ticks(data, padding=0.0):
    """Calculate axis bounds and evenly-spaced 'nice' tick positions for linear data.

    Snaps the bounds to round numbers, anchors ticks at zero when the range
    crosses zero, and chooses a tick spacing that yields ~5-8 ticks.

    Args:
        data: iterable of numeric values (often just [min_val, max_val])
        padding: fractional padding added to each side of the data range
            (e.g. 0.05 adds 5% on each side). Padding is not applied to a
            bound that has been snapped to zero.

    Returns:
        bounds: [min, max] suitable for ax.set_ylim / ax.set_xlim
        ticks_vals: array of tick positions within (and slightly beyond) bounds
    """
    if not data:
        return (0, 1)  # default bounds if no data

    min_val = min(data)
    max_val = max(data)

    # Snap bound to zero when data is close to or entirely on one side of zero
    if abs(min_val) < 1e-6 * max_val or (min_val > 0 and max_val > 0 and min_val <= 0.5 * max_val):
        min_val = 0
    if abs(max_val) < 1e-6 * abs(min_val) or (min_val < 0 and max_val < 0 and max_val >= 0.5 * min_val):
        max_val = 0

    range_val = max_val - min_val

    padding_amount = range_val * padding
    if min_val != 0:
        min_val -= padding_amount
    if max_val != 0:
        max_val += padding_amount

    # Express range as mantissa × 10^power (mantissa in [1, 10))
    range_val = max_val - min_val
    range_power = floor(log10(range_val))
    range_mantissa = range_val / 10**range_power

    # Select a tick pattern (normalized to [0, range_scale]) based on mantissa.
    # The pattern's step size (ticks[1]) is scaled by 10^power to get the
    # actual tick spacing.
    if range_mantissa == 1.0:
        ticks = np.arange(0,1.2,0.2)
        range_scale = 1
    elif range_mantissa <= 1.5:
        ticks = np.arange(0,1.8,0.3)
        range_scale = 1.5
    elif range_mantissa <= 2:
        ticks = np.arange(0,2.2,0.4)
        range_scale = 2
    elif range_mantissa <= 3:
        ticks = np.arange(0,3.5,0.5)
        range_scale = 3 
    elif range_mantissa <= 4:
        ticks = np.arange(0,4.5,0.5)
        range_scale = 4
    elif range_mantissa <= 7:
        ticks = np.arange(0,8,1)
        range_scale = -floor(-range_mantissa)  # round up to nearest integer
    elif range_mantissa <= 10:
        ticks = np.arange(0,12,2)
        range_scale = -floor(-range_mantissa) 
    else:
        print("Unexpected range mantissa:", range_mantissa)
        raise ValueError("Unexpected range mantissa: {}".format(range_mantissa))
    
    min_tick = round(min_val / 10**range_power) * 10**range_power
    max_tick = round(max_val / 10**range_power) * 10**range_power

    min_val = min(min_val, min_tick)
    max_val = max(max_val, max_tick)

    if min_val <= 0 and max_val >= 0:
        # 0 is in range, so anchor ticks at zero
        tick_step = ticks[1] * 10**range_power  # spacing from the tick pattern
        pos_ticks = np.arange(0, max_val + tick_step, tick_step)
        neg_ticks = np.arange(-tick_step, min_val - tick_step, -tick_step)
        ticks_vals = np.sort(np.concatenate([neg_ticks, pos_ticks]))

    else:
        # 0 not in range, so add ticks at regular intervals of 10**range_power
        tick_step = ticks[1] * 10**range_power
        ticks_vals = min_tick + ticks * 10**range_power

    # Ensure ticks extend one step beyond data range on each end
    if len(ticks_vals) > 0 and ticks_vals[0] > min_val:
        ticks_vals = np.concatenate([[ticks_vals[0] - tick_step], ticks_vals])
    if len(ticks_vals) > 0 and ticks_vals[-1] < max_val:
        ticks_vals = np.concatenate([ticks_vals, [ticks_vals[-1] + tick_step]])

    # Update bounds to match outermost ticks
    if len(ticks_vals) > 0:
        min_val = min(min_val, ticks_vals[0])
        max_val = max(max_val, ticks_vals[-1])

    return [min_val, max_val], ticks_vals


# Candidate multipliers and their "niceness" weights for round_pct_nice().
# Each entry is (multiplier, weight). Lower weight = "nicer" number, meaning the
# algorithm will stretch further from the raw value to land on it.
# E.g. 10, 50, 100 are very nice (low weight); 20, 30, 40 are ordinary (weight 1.0).
_MULT_WEIGHTS = [
    (1.0, 1/9),   # 10, 100, 1000 — powers of 10, very nice
    (1.5, 2/3),   # 15, 150, 1500
    (2.0, 1.0),   # 20, 200
    (2.5, 2/3),   # 25, 250
    (3.0, 1.0),   # 30, 300
    (4.0, 1.0),   # 40, 400
    (5.0, 1/3),   # 50, 500 — half-powers of 10, very nice
    (6.0, 1.0),
    (7.0, 1.0),
    (7.5, 2/3),   # 75, 750
    (8.0, 1.0),
    (9.0, 1.0),
]


def round_pct_nice(pct, pct_range=None):
    """Round a percentage-change value to a 'nice' number for axis labels.

    Finds the nearest "nice" number (powers of 10, multiples of 5/25/50/75)
    using a weighted-distance metric where nicer numbers have lower weights,
    so the algorithm prefers them even if they're slightly further away.

    Special handling for pct <= -90: rounds the remainder from -100 to the
    nearest of {1, 2, 5, 10} × 10^k (e.g. -95 → -95, -93 → -95).

    Args:
        pct: the percentage value to round (e.g. 23.7 or -48.2)
        pct_range: optional (min_pct, max_pct) tuple defining the visible axis
            range. Candidates outside this range lose their niceness bonus
            (weight clamped to 1.0) so the algorithm won't stretch to reach
            a "nice" number that would fall outside the plot.

    Returns:
        The nearest nice percentage value (float).
    """
    if pct == 0:
        return 0
    if pct <= -90:
        remainder = 100 + pct  # e.g. pct=-95 -> remainder=5
        if remainder <= 0:
            return -100.0
        mag = 10 ** floor(log10(remainder))
        best = None
        best_dist = float('inf')
        for m in [1, 2, 5, 10]:
            candidate = m * mag
            dist = abs(remainder - candidate)
            if dist < best_dist:
                best_dist = dist
                best = candidate
        return -(100 - best)
    abs_pct = abs(pct)
    magnitude = 10 ** floor(log10(abs_pct))
    best_candidate = None
    best_score = float('inf')
    for mag in [magnitude / 10, magnitude, magnitude * 10]:
        for mult, weight in _MULT_WEIGHTS:
            candidate = mult * mag
            signed = -candidate if pct < 0 else candidate
            # Only apply niceness bonus if candidate is within axis range
            if pct_range is not None and not (pct_range[0] <= signed <= pct_range[1]):
                w = max(weight, 1.0)
            else:
                w = weight
            score = abs(abs_pct - candidate) * w
            if score < best_score:
                best_score = score
                best_candidate = candidate
    if pct < 0:
        best_candidate = -best_candidate
    return best_candidate


def get_axis_bounds_and_ticks_ln_pct(data, padding=0.0):
    """Calculate axis bounds and ticks for log-ratio data with percentage-change labels.

    For data stored as ln(value / baseline), this function computes ~6 evenly
    spaced tick positions anchored at zero, rounds each to a "nice" percentage
    via round_pct_nice(), then converts back to log space. The result is tick
    marks at human-friendly percentages (e.g. -25%, 0%, 25%, 50%) positioned
    correctly on the log-ratio axis.

    Args:
        data: iterable of values in log-ratio space, i.e. ln(value/baseline).
            Often just [min_val, max_val].
        padding: fractional padding added to each side of the data range
            (e.g. 0.05 adds 5% on each side)

    Returns:
        bounds: [min, max] in log space, suitable for ax.set_ylim
        ticks_vals: array of tick positions in log space, suitable for ax.set_yticks
        pct_labels: list of percentage-change values at each tick (e.g. [-25, 0, 25]).
            Use with ax.set_yticklabels([f'{p:g}%' for p in pct_labels]).
    """
    min_log = min(data)
    max_log = max(data)
    span = max_log - min_log
    min_log -= span * padding
    max_log += span * padding

    # Approximate spacing for ~6 ticks
    interval = (max_log - min_log) / 7

    # Build raw tick positions anchored at zero, extending 2 extra intervals
    # beyond bounds so that after rounding we still have ticks near the edges
    raw_ticks = [0.0]
    t = interval
    while t <= max_log + 2 * interval:
        raw_ticks.append(t)
        t += interval
    t = -interval
    while t >= min_log - 2 * interval:
        raw_ticks.append(t)
        t -= interval

    # Convert to nice percentages, deduplicate
    # Compute pct range corresponding to the padded bounds
    pct_min = (np.exp(min_log) - 1) * 100
    pct_max = (np.exp(max_log) - 1) * 100
    seen = set()
    pct_labels = []
    for t in raw_ticks:
        pct = round_pct_nice((np.exp(t) - 1) * 100, pct_range=(pct_min, pct_max))
        if pct not in seen:
            seen.add(pct)
            pct_labels.append(pct)
    pct_labels.sort()

    # Convert to log space and keep ticks within the padded data range
    ticks_vals = np.log(1 + np.array(pct_labels) / 100.)
    mask = (ticks_vals >= min_log - 1e-9) & (ticks_vals <= max_log + 1e-9)
    pct_labels = [p for p, m in zip(pct_labels, mask) if m]
    ticks_vals = ticks_vals[mask]

    # Bounds: at least as wide as data, expand if ticks go beyond
    bounds = [min(min_log, ticks_vals[0]), max(max_log, ticks_vals[-1])]

    return bounds, ticks_vals, pct_labels


def plot_ratios(x_data, ratio_data, x_axis_label='x_axis_label',
                y_axis_label='y_axis_label', title='title', padding=0.05,
                colors=None):
    """Plot ratio data with y-axis formatted as percentage change.

    Takes raw ratio data (value/baseline, where 1.0 = no change) and plots it
    on a log-ratio y-axis with human-friendly percentage-change labels.

    Args:
        x_data: 1-D array-like of length N (x-axis values)
        ratio_data: array-like of ratios (1.0 = no change, 2.0 = +100%, etc.)
            If 1-D (length N): single line plotted.
            If 2-D with shape (N, M): M lines plotted, one per column.
        x_axis_label: label for x-axis
        y_axis_label: label for y-axis
        title: plot title
        padding: fractional padding for y-axis range (default 0.05)
        colors: optional list of colors, one per column of ratio_data.
            If None, uses matplotlib's default color cycle.

    Returns:
        fig, ax: the matplotlib Figure and Axes objects
    """
    ratio_data = np.asarray(ratio_data, dtype=float)
    if ratio_data.ndim == 1:
        ratio_data = ratio_data.reshape(-1, 1)

    if ratio_data.shape[0] != len(x_data):
        raise ValueError(
            f"First dimension of ratio_data ({ratio_data.shape[0]}) "
            f"must match length of x_data ({len(x_data)})")

    if colors is not None and len(colors) != ratio_data.shape[1]:
        raise ValueError(
            f"Length of colors ({len(colors)}) must match number of columns "
            f"in ratio_data ({ratio_data.shape[1]})")

    log_data = np.log(ratio_data)

    global_min = log_data.min()
    global_max = log_data.max()

    bounds, ticks, pct_labels = get_axis_bounds_and_ticks_ln_pct(
        [global_min, global_max], padding=padding)

    fig, ax = plt.subplots()
    for i in range(log_data.shape[1]):
        kwargs = {'color': colors[i]} if colors is not None else {}
        ax.plot(x_data, log_data[:, i], **kwargs)

    ax.set_ylim(bounds)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{p:g}%' for p in pct_labels])
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)

    return fig, ax


def plot_ratios_shaded(x_data, ratio_data, x_axis_label='x_axis_label',
                       y_axis_label='y_axis_label', title='title', padding=0.05,
                       color=None):
    """Plot ratio data with shaded uncertainty bands and percentage-change y-axis.

    The number of columns in ratio_data determines the visual style:
        1 column:  solid line (identical to plot_ratios)
        3 columns: shaded band (cols 0–2) with median line (col 1)
        5 columns: two nested bands (outer cols 0–4, inner cols 1–3) with median (col 2)
        7 columns: thin lines at cols 0 & 6, two nested bands, median line (col 3)

    All elements (lines, bands) use the same color, distinguished by alpha/linewidth.

    Args:
        x_data: 1-D array-like of length N (x-axis values)
        ratio_data: array-like of ratios with shape (N,), (N,1), (N,3), (N,5), or (N,7)
        x_axis_label: label for x-axis
        y_axis_label: label for y-axis
        title: plot title
        padding: fractional padding for y-axis range (default 0.05)
        color: optional color for all plot elements (any matplotlib color spec).
            If None, uses the first color from matplotlib's default cycle.

    Returns:
        fig, ax: the matplotlib Figure and Axes objects
    """
    ratio_data = np.asarray(ratio_data, dtype=float)
    if ratio_data.ndim == 1:
        ratio_data = ratio_data.reshape(-1, 1)

    if ratio_data.shape[0] != len(x_data):
        raise ValueError(
            f"First dimension of ratio_data ({ratio_data.shape[0]}) "
            f"must match length of x_data ({len(x_data)})")

    M = ratio_data.shape[1]
    if M not in (1, 3, 5, 7):
        raise ValueError(
            f"ratio_data must have 1, 3, 5, or 7 columns, got {M}")

    log_data = np.log(ratio_data)

    global_min = log_data.min()
    global_max = log_data.max()

    bounds, ticks, pct_labels = get_axis_bounds_and_ticks_ln_pct(
        [global_min, global_max], padding=padding)

    fig, ax = plt.subplots()

    if color is None:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    if M == 1:
        ax.plot(x_data, log_data[:, 0], color=color)
    elif M == 3:
        ax.fill_between(x_data, log_data[:, 0], log_data[:, 2], color=color, alpha=0.3)
        ax.plot(x_data, log_data[:, 1], color=color)
    elif M == 5:
        ax.fill_between(x_data, log_data[:, 0], log_data[:, 4], color=color, alpha=0.15)
        ax.fill_between(x_data, log_data[:, 1], log_data[:, 3], color=color, alpha=0.3)
        ax.plot(x_data, log_data[:, 2], color=color)
    elif M == 7:
        ax.plot(x_data, log_data[:, 0], color=color, linewidth=0.5, alpha=0.7)
        ax.plot(x_data, log_data[:, 6], color=color, linewidth=0.5, alpha=0.7)
        ax.fill_between(x_data, log_data[:, 1], log_data[:, 5], color=color, alpha=0.15)
        ax.fill_between(x_data, log_data[:, 2], log_data[:, 4], color=color, alpha=0.3)
        ax.plot(x_data, log_data[:, 3], color=color)

    ax.set_ylim(bounds)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{p:g}%' for p in pct_labels])
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)

    return fig, ax
