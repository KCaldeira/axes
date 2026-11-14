# Compressing axis transforms: arcsinh ("asinh") and double arcsinh ("asinh2").
#
# Each kind is linear (slope 1/scale) for |z| << scale and compresses beyond:
#
#   "asinh"   u = asinh(z / scale)           tails ~ ln|z|      (singly logarithmic)
#   "asinh2"  u = asinh(asinh(z / scale))    tails ~ ln ln|z|   (doubly logarithmic)
#
# "asinh" suits data whose tails run a few orders of magnitude past `scale`.
# "asinh2" suits data whose tails run many orders of magnitude past the interesting
# range (e.g. a ratio axis that must hold both +1e12% and -99.99999%): near zero
# it behaves like asinh, but it gives the region around +-scale a much larger
# share of the axis. Both are smooth and odd. arcsinh is used rather than
# matplotlib's `symlog` because it is smooth everywhere: symlog is piecewise and
# has a visible kink where its linear and log regimes meet.
#
# The same transforms serve amount axes (z = the amount, see amount_arcsinh_ticks)
# and ratio axes (z = the natural-log ratio, see ratio_percent_ticks).
# To add a kind, add its (forward, inverse) pair to ASINH_TRANSFORMS.

import numpy as np

ASINH_TRANSFORMS = {
    "asinh": (lambda z, scale: np.arcsinh(z / scale),
              lambda u, scale: scale * np.sinh(u)),
    "asinh2": (lambda z, scale: np.arcsinh(np.arcsinh(z / scale)),
               lambda u, scale: scale * np.sinh(np.sinh(u))),
}

ASINH_KINDS = tuple(ASINH_TRANSFORMS)


def _check_kind(kind):
    if kind not in ASINH_TRANSFORMS:
        raise ValueError(f"kind must be one of {ASINH_KINDS}, got {kind!r}")


def asinh_transform(z, scale, kind="asinh"):
    """Map values into compressed plotting space. Accepts scalars or arrays."""
    _check_kind(kind)
    return ASINH_TRANSFORMS[kind][0](np.asarray(z, dtype=float), scale)


def asinh_inverse(u, scale, kind="asinh"):
    """Inverse of `asinh_transform`: plotting space back to the original value."""
    _check_kind(kind)
    return ASINH_TRANSFORMS[kind][1](np.asarray(u, dtype=float), scale)


def choose_ladder_ticks(u_lo, u_hi, rungs, n_ticks=9, pad=0.03, symmetric=False,
                        zero_label="0"):
    """Ticks and limits on a compressed axis, chosen from a ladder of nice values.

    Shared core of the compressed ratio axis (ratio_percent_ticks) and the
    "asinh2" amount axis (amount_arcsinh_ticks). Works in transformed space.

    The outermost tick on each side is the first rung at or beyond the data,
    and the axis is extended to it, so no data run past the last label. A side
    the data do not reach ends at the zero tick. Interior ticks aim at even
    spacing in transformed space: each side gets targets in proportion to its
    length, each target snaps to the nearest rung, and a rung closer than half
    the even spacing to an already-placed tick is dropped, so labels cannot
    collide. Padding is added only as a margin beyond the outer ticks, never by
    widening the data range: far out on a compressed axis a small step in
    transformed space is a huge step in the underlying value.

    Args:
        u_lo, u_hi: data range in transformed space (0 is always included)
        rungs: [(u, label), ...] nonzero candidate ticks in transformed space;
            `label` is returned as is, so it may be a string or a number
        n_ticks: target number of ticks
        pad: margin beyond the outermost ticks, as a fraction of their span
        symmetric: if True, make the range and limits symmetric about 0
        zero_label: label of the tick at 0

    Returns:
        (plot_min, plot_max, tick_locations, tick_labels)
    """
    if u_hi < u_lo:
        u_lo, u_hi = u_hi, u_lo
    u_lo, u_hi = min(u_lo, 0.0), max(u_hi, 0.0)
    if symmetric:
        u_hi = max(u_hi, -u_lo)
        u_lo = -u_hi
    neg = sorted([r for r in rungs if r[0] < 0], key=lambda r: -r[0])   # outward from 0
    pos = sorted([r for r in rungs if r[0] > 0], key=lambda r: r[0])
    zero = (0.0, zero_label)
    if u_lo == 0.0 and u_hi == 0.0:
        # All data at zero: show the innermost rung on each side.
        outer_lo, outer_hi = neg[0], pos[0]
    else:
        # Data past the end of the ladder get its last rung; the limits still cover them.
        outer_lo = next((r for r in neg if r[0] <= u_lo), neg[-1]) if u_lo < 0 else zero
        outer_hi = next((r for r in pos if r[0] >= u_hi), pos[-1]) if u_hi > 0 else zero
    span = outer_hi[0] - outer_lo[0]
    ticks = [zero] + [r for r in (outer_lo, outer_hi) if r is not zero]
    for side, outer in ((neg, outer_lo), (pos, outer_hi)):
        inner = [r for r in side if abs(r[0]) < abs(outer[0])]
        if not inner:
            continue
        n_side = max(round(n_ticks * abs(outer[0]) / span), 1)
        for k in range(1, n_side):
            target = k / n_side * outer[0]
            rung = min(inner, key=lambda r: abs(r[0] - target))
            if all(abs(rung[0] - t[0]) >= 0.5 * span / n_ticks for t in ticks):
                ticks.append(rung)
    ticks.sort(key=lambda t: t[0])
    margin = pad * span
    plot_min = min(outer_lo[0], u_lo) - margin
    plot_max = max(outer_hi[0], u_hi) + margin
    if symmetric:
        extremum = max(abs(plot_min), abs(plot_max))
        plot_min, plot_max = -extremum, extremum
    return plot_min, plot_max, [t[0] for t in ticks], [t[1] for t in ticks]
