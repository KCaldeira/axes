# Ratio-to-percent tick generation for matplotlib axes.
#
# Accepts raw ratio data (where 1.0 = no change), log-transforms internally,
# then computes buffered bounds and nicely-spaced tick marks labeled as
# percentage change.
#
# Standalone usage:
#     plot_min, plot_max, tick_locs, tick_labels = choose_ratio_percent_ticks(rmin, rmax)
#
# High-level usage:
#     bounds, ticks_vals, pct_labels = get_axis_bounds_and_ticks_ratio_pct(data, padding=0.05)
#     ax.set_ylim(bounds)
#     ax.set_yticks(ticks_vals)
#     ax.set_yticklabels([format_percent(p) for p in pct_labels])
#
# Compressed ratio axes (asinh / asinh2 of the log ratio), for data spanning
# far more range than a plain log axis can show:
#     bounds, ticks_vals, labels = get_axis_bounds_and_ticks_ratio_compressed(
#         data, scale=0.3, kind="asinh2")
#     ax.plot(x, ratio_axis_transform(np.log(ratios), 0.3, "asinh2"))
#     ax.set_ylim(bounds)
#     ax.set_yticks(ticks_vals)
#     ax.set_yticklabels(labels)

import math

import numpy as np

from asinh_transforms import asinh_transform, asinh_inverse, choose_ladder_ticks


def percent_to_log(p):
    """Convert percent change to natural log ratio."""
    return math.log(1 + p / 100.0)


def log_to_percent(x):
    """Convert natural log ratio to percent change."""
    return 100 * (math.exp(x) - 1)


def ratio_to_percent(r):
    """Convert a ratio to percent change.  ratio 1.0 = 0%, ratio 2.0 = +100%."""
    return 100 * (r - 1)


def format_percent(p):
    """Format percent labels cleanly.  Never rounds to -100%."""
    if abs(p) < 1e-12:
        return "0%"
    # Values between -100 and -99: show as -99.9…9%
    if -100 < p < -99:
        # number of 9s after the decimal = number of digits needed
        nines = -math.log10(100 + p)  # e.g. -99.9 -> 1, -99.99 -> 2
        decimals = max(1, math.ceil(nines - 1e-9))
        return f"{p:.{decimals}f}%"
    sign = "+" if p > 0 else ""
    if abs(p) >= 1:
        return f"{sign}{round(p):.0f}%"
    else:
        return f"{sign}{p:.2f}%"


def build_percent_candidates():
    """Build a library of 'nice' percent ticks.

    Negative side:
        -90, -98, -99, -99.5
        plus survival powers of 10 ( -99.9, -99.99, ... )

    Positive side:
        1-2-5 sequence (10%, 20%, 50%, 100%, 200%, 500%, ...)
    """
    perc = set()

    # basic negative ticks
    perc.update([-90, -98, -99, -99.5])

    # survival powers of 10: -99.9, -99.99, -99.999, ...
    for k in range(1, 8):
        s = 10 ** (-k)
        p = -100 * (1 - s)
        perc.add(p)

    # small and mid-range negative values
    perc.update([-80, -75, -70, -60, -50, -30, -20, -10, -5, -2, -1])

    # zero
    perc.add(0)

    # positive 1-2-3-5 sequence
    for scale in [1, 10, 100, 1000, 10000]:
        for base in [1, 2, 3, 5]:
            perc.add(base * scale)

    # keep values > -100
    perc = [p for p in perc if p > -100]

    return sorted(perc)


def _find_nearest_candidate(target, candidates_log):
    """Find the candidate in log space nearest to target."""
    idx = np.searchsorted(candidates_log, target)
    # Check the two nearest candidates
    best_idx = idx
    best_dist = float('inf')
    for i in [max(0, idx - 1), min(len(candidates_log) - 1, idx)]:
        dist = abs(candidates_log[i] - target)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def choose_ratio_percent_ticks(rmin, rmax, symmetric=False, buffer=0.05):
    """Choose nicely-spaced tick marks for a ratio axis with percent labels.

    Accepts raw ratio values (1.0 = no change), log-transforms internally,
    then finds boundary ticks from nice candidates and fills interior with
    7 total ticks proportionally allocated between negative and positive sides.

    Args:
        rmin: minimum ratio value (e.g. 0.5 for a 50% decline)
        rmax: maximum ratio value (e.g. 2.0 for a 100% increase)
        symmetric: if True, make bounds symmetric around zero (in log space)
        buffer: fractional buffer to add beyond data range (default 0.05)

    Returns:
        plot_min: adjusted minimum bound in log space
        plot_max: adjusted maximum bound in log space
        tick_locations: list of tick positions in log space
        tick_labels: list of formatted percentage strings
    """
    # Convert ratios to log space
    xmin = math.log(rmin)
    xmax = math.log(rmax)

    if xmax < xmin:
        xmin, xmax = xmax, xmin

    # ensure zero included if one-sided
    if xmin > 0:
        xmin = 0
    if xmax < 0:
        xmax = 0

    # Step 1: Compute buffered bounds
    if symmetric:
        y_extreme = max(xmax, -xmin)
        y_max_new = y_extreme * (1 + buffer)
        y_min_new = -y_max_new
    else:
        y_max_new = xmax * (1 + buffer)
        y_min_new = xmin * (1 + buffer)  # xmin is negative, so this makes it more negative

    # Build candidates in log space
    perc_candidates = build_percent_candidates()
    candidates = [(p, percent_to_log(p)) for p in perc_candidates]
    candidates.sort(key=lambda t: t[1])
    candidates_log = np.array([x for _, x in candidates])
    candidates_pct = [p for p, _ in candidates]

    # Step 2: Find boundary ticks (outermost nice ticks)
    # Strategy: look for a nice candidate in a window near the buffered bound.
    # If none found, snap to the nearest candidate (inward is fine — plot bounds
    # are set independently to cover the buffered data range).

    # Upper bound: look for nice candidate in [y_max_new * (1 - 2*buffer), y_max_new]
    upper_inner = y_max_new * (1 - 2 * buffer)
    upper_candidates = [(i, candidates_log[i]) for i in range(len(candidates_log))
                        if upper_inner <= candidates_log[i] <= y_max_new]
    if upper_candidates:
        max_tick_idx = upper_candidates[-1][0]  # pick the outermost one in range
    else:
        max_tick_idx = _find_nearest_candidate(y_max_new, candidates_log)

    # Lower bound: look for nice candidate in [y_min_new, y_min_new * (1 - 2*buffer)]
    lower_inner = y_min_new * (1 - 2 * buffer)  # closer to zero since y_min_new is negative
    lower_candidates = [(i, candidates_log[i]) for i in range(len(candidates_log))
                        if y_min_new <= candidates_log[i] <= lower_inner]
    if lower_candidates:
        min_tick_idx = lower_candidates[0][0]  # pick the outermost one (most negative)
    else:
        min_tick_idx = _find_nearest_candidate(y_min_new, candidates_log)

    min_tick = candidates_log[min_tick_idx]
    max_tick = candidates_log[max_tick_idx]

    # Step 3: Fill interior ticks (7 total)
    # 3 fixed: min_tick, 0, max_tick
    # 4 additional allocated proportionally between negative and positive sides
    full_span = max_tick - min_tick  # min_tick is negative, so this is the full range
    n_pos = round(max_tick / full_span * 4) if full_span > 0 else 2
    n_neg = 4 - n_pos

    targets = [min_tick]
    for k in range(1, n_neg + 1):
        targets.append(k / (n_neg + 1) * min_tick)
    targets.append(0.0)
    for k in range(1, n_pos + 1):
        targets.append(k / (n_pos + 1) * max_tick)
    targets.append(max_tick)

    # Snap each target to nearest candidate, deduplicate
    seen = set()
    ticks = []
    for target in targets:
        idx = _find_nearest_candidate(target, candidates_log)
        if idx not in seen:
            seen.add(idx)
            ticks.append((candidates_pct[idx], candidates_log[idx]))

    ticks.sort(key=lambda t: t[1])

    # Ensure zero is included
    if not any(abs(x) < 1e-12 for _, x in ticks):
        zero_idx = _find_nearest_candidate(0.0, candidates_log)
        if abs(candidates_log[zero_idx]) < 1e-12:
            ticks.append((0, 0.0))
            ticks.sort(key=lambda t: t[1])

    # Drop ticks that are too close together (< 5% of axis span).
    # Always keep zero as the anchor.
    total_span = ticks[-1][1] - ticks[0][1]
    min_gap = total_span * 0.05
    if min_gap > 0:
        filtered = [ticks[0]]
        for t in ticks[1:]:
            if abs(t[1]) < 1e-12:
                # Always keep zero; if previous tick is too close, remove it
                if abs(filtered[-1][1] - t[1]) < min_gap:
                    filtered[-1] = t
                else:
                    filtered.append(t)
            elif t[1] - filtered[-1][1] >= min_gap:
                filtered.append(t)
            # else: skip this tick (too close to previous kept tick)
        ticks = filtered

    # Plot bounds cover both the outermost ticks and the buffered data range
    plot_min = min(ticks[0][1], y_min_new)
    plot_max = max(ticks[-1][1], y_max_new)

    # The asymmetric tick-candidate library can pull plot_min/plot_max apart
    # (e.g. -80% snaps to ln(0.2)≈-1.609 while +300% snaps to ln(4)≈1.386).
    # When the caller asked for symmetric bounds — typically because they're
    # pairing this with a diverging colormap whose midpoint must land on
    # zero — widen to the symmetric extremum so RdBu's white sits at 0%.
    if symmetric:
        extremum = max(abs(plot_min), abs(plot_max))
        plot_min = -extremum
        plot_max = +extremum

    tick_locations = [x for _, x in ticks]
    tick_labels = [format_percent(p) for p, _ in ticks]

    return plot_min, plot_max, tick_locations, tick_labels


def get_axis_bounds_and_ticks_ratio_pct(data, padding=0.0, symmetric=False, buffer=0.05):
    """Calculate axis bounds and ticks for ratio data with percentage-change labels.

    Accepts raw ratio values (where 1.0 = no change, 2.0 = +100%, 0.5 = -50%),
    log-transforms internally, then computes tick positions in log space with
    corresponding percentage labels.

    Args:
        data: iterable of ratio values (e.g. [0.8, 1.5] meaning -20% to +50%).
            Often just [min_ratio, max_ratio].
        padding: fractional padding added to each side of the data range
            (in log space)
        symmetric: if True, make bounds symmetric around zero (in log space)
        buffer: fractional buffer for tick boundary selection (default 0.05)

    Returns:
        bounds: [min, max] in log space, suitable for ax.set_ylim
        ticks_vals: numpy array of tick positions in log space
        pct_labels: list of numeric percentage values at each tick
            (e.g. [-25, 0, 25]). Use with
            ax.set_yticklabels([format_percent(p) for p in pct_labels]).
    """
    finite_data = [x for x in data if np.isfinite(x)]
    # Convert ratios to log space
    log_data = [math.log(x) for x in finite_data if x > 0]
    xmin = min(log_data) if log_data else -0.1
    xmax = max(log_data) if log_data else 0.1
    span = xmax - xmin
    xmin -= span * padding
    xmax += span * padding

    # Convert padded log bounds back to ratios for choose_ratio_percent_ticks
    rmin = math.exp(xmin)
    rmax = math.exp(xmax)

    plot_min, plot_max, tick_locations, _ = choose_ratio_percent_ticks(
        rmin, rmax, symmetric=symmetric, buffer=buffer
    )

    # Add a small margin so outermost ticks aren't clipped by matplotlib
    margin = (plot_max - plot_min) * 0.02
    bounds = [plot_min - margin, plot_max + margin]
    ticks_vals = np.array(tick_locations)
    pct_labels = [log_to_percent(x) for x in tick_locations]

    return bounds, ticks_vals, pct_labels


# ==============================================================================
# Compressed ratio axes: arcsinh ("asinh") and double arcsinh ("asinh2")
# ==============================================================================
#
# A plain log-ratio axis (above) is linear in ln and so gives equal space to
# every factor of e. That fails when one series spans far more range than the
# rest, e.g. most series sit within +-0.4 ln while one reaches +-10 ln.
#
# The fix is to plot a compressed transform u of z, where z is still the
# natural-log ratio (see asinh_transforms):
#
#   "asinh"   u = asinh(z / scale)           tails ~ ln|z|
#   "asinh2"  u = asinh(asinh(z / scale))    tails ~ ln ln|z|
#
# With scale = 0.3 both kinds stay within ~4% of linear across +-10%, but
# asinh2 gives that band about twice the share of the axis. Both are odd in z,
# so +100% and -50% are equidistant from zero.
#
# The transform is applied to the *log ratio*, not to percent: percent is
# bounded below at -100% but unbounded above, so arcsinh-on-percent gives a
# badly lopsided axis. Ticks are still labeled in percent.


def ratio_axis_transform(x, scale, kind="asinh"):
    """Map natural-log ratios into compressed plotting space.

    For ratio data r, pass np.log(r). Linear for |x| << scale, compressed beyond.
    """
    return asinh_transform(x, scale, kind)


def ratio_axis_inverse(u, scale, kind="asinh"):
    """Inverse of `ratio_axis_transform`: plotting space back to log ratio."""
    return asinh_inverse(u, scale, kind)


def format_percent_wide(p):
    """Percent label with thousands separators, for axes reaching extreme ratios.

    Unlike `format_percent`, fractional magnitudes are never rounded to an
    integer, so -99.9, -99.99, ... stay distinguishable from each other and
    from -100%, which a ratio axis can approach but never reach.
    """
    if abs(p) < 1e-12:
        return "0%"
    sign = "+" if p > 0 else "-"
    magnitude = abs(p)
    if abs(magnitude - round(magnitude)) < 1e-9:
        return f"{sign}{round(magnitude):,d}%"
    # 10 significant digits: `:g` keeps only 6 and prints -99.99999 as "-100%".
    return f"{sign}{magnitude:.10g}%"


def collapse_candidates(max_nines=30):
    """Near-total-collapse ticks placed exactly in log space: [(ln ratio, label), ...].

    Survival fraction 10^-k (k = 3 .. max_nines) sits at ln = -k ln 10 and is
    labeled with its nines ("-99.9%", "-99.99%", ...) up to eight of them;
    deeper than that a row of nines is unreadable, so it is written as
    -(100 - 10^(2-k))%, i.e. via the percent that remains. Built from k
    rather than from a percent, because past ~11 nines a double-precision
    percent can no longer tell -99.999999999999% from -100%, while a
    compressed axis routinely holds collapses of e^-30 or more.
    """
    return [(-k * math.log(10.0),
             "-99." + "9" * (k - 2) + "%" if k <= 10 else rf"$-(100 - 10^{{{2 - k}}})\%$")
            for k in range(3, max_nines + 1)]


def compressed_ratio_ladder(max_nines=30, max_power=30, min_exponent=0):
    """Tick ladder for a compressed ratio axis: [(ln ratio, label), ...], exact in ln.

    Negative: -1, -2, -5, -10, -20, -30, -50, -70, -90%, then survival 10^-k
    as -99%, -99.9%, ... (via `collapse_candidates`). Positive: +1, +2, +5,
    +10, +20, +30, +50, +100, +200, +300, +500, +1,000%, then +10^k% for
    k = 4 .. max_power, placed at ln(1 + 10^(k-2)) and labeled as a power,
    because a row of zeros is as unreadable as a row of nines.

    `min_exponent < 0` adds a symmetric 1-2-5 sequence below 1% (e.g. -1 adds
    +-0.1, +-0.2, +-0.5%), for axes whose data sit well inside +-1%.
    """
    neg = [(math.log1p(-p / 100.0), f"-{p}%") for p in (1, 2, 5, 10, 20, 30, 50, 70, 90)]
    neg.append((-2 * math.log(10.0), "-99%"))
    pos = [(math.log1p(p / 100.0), f"+{p:,d}%")
           for p in (1, 2, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000)]
    pos += [(math.log1p(10.0 ** (k - 2)), rf"$+10^{{{k}}}\%$") for k in range(4, max_power + 1)]
    small = []
    for exponent in range(min_exponent, 0):
        for base in (1, 2, 5):
            p = base * 10.0 ** exponent
            small += [(math.log1p(p / 100.0), f"+{p:g}%"), (math.log1p(-p / 100.0), f"-{p:g}%")]
    return neg + collapse_candidates(max_nines) + pos + small


def choose_compressed_ratio_ticks(zmin, zmax, scale, kind="asinh", n_ticks=9, pad=0.03,
                                  symmetric=False, min_exponent=0):
    """Ticks and limits for a compressed (asinh/asinh2) ratio axis, labeled in percent.

    The outermost tick on each side is the first ladder rung at or beyond the
    data, and the axis is extended to it, so no data can run past the last
    label (where it would read as, e.g., a fall below -100%). Interior ticks
    are evenly spaced in transformed space and never collide; see
    `asinh_transforms.choose_ladder_ticks`.

    Args:
        zmin, zmax: data range in natural-log-ratio units (0 is always included)
        scale: transition from linear to compressed, in log-ratio units
        kind: "asinh" or "asinh2" (see asinh_transforms)
        n_ticks: target number of ticks (default 9)
        pad: margin beyond the outermost ticks, as a fraction of their span
        symmetric: if True, make the limits symmetric about 0 (e.g. for a
            diverging colormap whose midpoint must sit at 0%)
        min_exponent: pass a negative value for sub-percent rungs (see
            `compressed_ratio_ladder`)

    Returns:
        (plot_min, plot_max, tick_locations, tick_labels), all in transformed space
    """
    fwd = lambda z: float(ratio_axis_transform(z, scale, kind))
    rungs = [(fwd(z), label) for z, label in compressed_ratio_ladder(min_exponent=min_exponent)]
    return choose_ladder_ticks(fwd(zmin), fwd(zmax), rungs, n_ticks=n_ticks, pad=pad,
                               symmetric=symmetric, zero_label="0%")


def get_axis_bounds_and_ticks_ratio_compressed(data, scale, kind="asinh", padding=0.03,
                                               symmetric=False, n_ticks=9, min_exponent=0):
    """Axis bounds and percent-labeled ticks for ratio data on a compressed axis.

    Accepts raw ratio values (1.0 = no change). Plot the data with
    `ratio_axis_transform(np.log(ratios), scale, kind)`.

    Args:
        data: iterable of ratio values, often just [min_ratio, max_ratio]
        scale: transition from linear to compressed, in log-ratio units
            (0.3 is a reasonable start: nearly linear across +-10%)
        kind: "asinh" or "asinh2"
        padding: margin beyond the outermost ticks, as a fraction of their
            span (default 0.03). The outermost ticks already lie at or beyond
            the data, so the data range itself is never widened.
        symmetric: if True, make bounds symmetric around zero
        n_ticks: target number of ticks
        min_exponent: pass a negative value for sub-percent ticks

    Returns:
        bounds: [min, max] in transformed space, suitable for ax.set_ylim
        ticks_vals: numpy array of tick positions in transformed space
        tick_labels: list of percent label strings, for ax.set_yticklabels.
            Strings rather than numbers, because ticks such as
            -(100 - 10^-20)% cannot be represented as a float percent.
    """
    log_data = [math.log(x) for x in data if np.isfinite(x) and x > 0]
    zmin = min(log_data) if log_data else -0.1
    zmax = max(log_data) if log_data else 0.1

    plot_min, plot_max, tick_locations, tick_labels = choose_compressed_ratio_ticks(
        zmin, zmax, scale, kind, n_ticks=n_ticks, pad=padding, symmetric=symmetric,
        min_exponent=min_exponent)

    return [plot_min, plot_max], np.array(tick_locations), tick_labels
