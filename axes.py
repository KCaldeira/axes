# functions to determine axes bounds and tick marks
import numpy as np
from math import log10, floor


def get_axis_bounds_and_ticks(data, padding=0.0):
    """Calculate the bounds of an axis given the data and padding."""
    if not data:
        return (0, 1)  # default bounds if no data

    min_val = min(data)
    max_val = max(data)

    if abs(min_val) < 1e-6 * max_val or (min_val > 0 and max_val > 0 and min_val <=0.5 * max_val):
        min_val = 0  # set to zero if very close to zero or if min_val is positive and small compared to max_val
    if abs(max_val) < 1e-6 * abs(min_val) or (min_val < 0 and max_val < 0 and max_val >= 0.5 * min_val):
        max_val = 0  # set to zero if very close to zero

    range_val = max_val - min_val

    # Add padding to the bounds
    padding_amount = range_val * padding
    if min_val != 0:
        min_val -= padding_amount
    if max_val != 0:
        max_val += padding_amount

    range_val = max_val - min_val
    range_power = floor(log10(range_val))   
    range_mantissa = range_val / 10**range_power  # range mantissa should be between 1 and 10

    #examples:
    # if min_val = -0.2 and max_val = 0.8, range_val = 1, range_power = 0, range_mantissa = 1
    # if min_val = -0.2 and max_val = 1.3, range_val = 1.5, range_power = 0, range_mantissa = 1.5
    # if min_val = -37 and max_val = 42, range_val = 79, range_power = 1, range_mantissa = 7.9
    # if min_val = -312 and max_val = 421, range_val = 733, range_power = 2, range_mantissa = 7.33

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


_EXTRA_NICE_PCTS = {250, 150, 75, 25, -25, -75, -95, -98}


def round_pct_nice(pct):
    """Round a percentage-change value to a 'nice' number for axis labels.

    First rounds to the nearest integer and checks against a list of extra nice
    values (25, 75, 150, 250, -25, -75, -95, -98).  If matched, returns that value.
    Otherwise falls back to rounding to one significant digit.
    For pct <= -90: round to successive nines (-90, -99, -99.9, -99.99, ...).
    """
    if pct == 0:
        return 0.0
    rounded_int = round(pct)
    if rounded_int in _EXTRA_NICE_PCTS:
        return float(rounded_int)
    if pct > -90:
        # Round to 1 significant digit
        magnitude = 10 ** floor(log10(abs(pct)))
        return round(pct / magnitude) * magnitude
    # Deep negative: remainder from -100 is a power of 10
    remainder = 100 + pct  # e.g. pct=-95 -> remainder=5
    if remainder <= 0:
        return -100.0
    power = round(log10(remainder))
    return -(100 - 10 ** power)


def get_axis_bounds_and_ticks_ln_pct(data, padding=0.0):
    """Calculate axis bounds and ticks for data in log-ratio space, with percentage-change labels.

    Args:
        data: values in log-ratio space (log(n/d))
        padding: fractional padding to add to bounds

    Returns:
        bounds: [min_val, max_val] in log space
        ticks_vals: tick positions in log space
        pct_labels: percentage-change value at each tick, e.g. -50.0 means -50%
    """
    min_log = min(data)
    max_log = max(data)
    span = max_log - min_log
    min_log -= span * padding
    max_log += span * padding

    # Approximate spacing for ~6 ticks
    interval = (max_log - min_log) / 6

    # Build raw tick positions anchored at zero
    raw_ticks = [0.0]
    t = interval
    while t <= max_log + interval:
        raw_ticks.append(t)
        t += interval
    t = -interval
    while t >= min_log - interval:
        raw_ticks.append(t)
        t -= interval

    # Convert to nice percentages, deduplicate
    seen = set()
    pct_labels = []
    for t in raw_ticks:
        pct = round_pct_nice((np.exp(t) - 1) * 100)
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
