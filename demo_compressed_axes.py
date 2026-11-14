"""Demo: log vs asinh vs asinh2 axes, for ratio data and for amount data.

Ratio data: three groups that sit within tens of percent, plus one group whose
values run from near-total collapse to a many-orders-of-magnitude blow-up --
the case the compressed axes are built for.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from axes import (format_percent, get_axis_bounds_and_ticks_ratio_pct, get_axis_bounds_and_ticks_ratio_compressed,
                  ratio_axis_transform, get_axis_bounds_and_ticks_arcsinh, amount_to_arcsinh)

rng = np.random.default_rng(42)
os.makedirs('plots', exist_ok=True)

# --- Ratio data --------------------------------------------------------------
groups = ['A', 'B', 'C', 'Wild']
log_ratios = [rng.normal(0.05, 0.08, 200), rng.normal(-0.1, 0.1, 200),
              rng.normal(0.2, 0.15, 200), rng.normal(0.0, 8.0, 200)]
ratios = [np.exp(z) for z in log_ratios]
all_ratios = np.concatenate(ratios)
data_range = [all_ratios.min(), all_ratios.max()]
SCALE = 0.3   # in log-ratio units: nearly linear across about +-10%

fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
for ax, kind in zip(axs, ['log', 'asinh', 'asinh2']):
    if kind == 'log':
        bounds, ticks, pcts = get_axis_bounds_and_ticks_ratio_pct(data_range)
        labels = [format_percent(p) for p in pcts]
        to_axis = np.log
    else:
        bounds, ticks, labels = get_axis_bounds_and_ticks_ratio_compressed(
            data_range, SCALE, kind)
        to_axis = lambda r, kind=kind: ratio_axis_transform(np.log(r), SCALE, kind)
    for i, r in enumerate(ratios):
        ax.scatter(i + rng.uniform(-0.25, 0.25, r.size), to_axis(r), s=4, alpha=0.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylim(bounds)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_title(kind if kind == 'log' else f'{kind} (scale = {SCALE} ln)')
axs[0].set_ylabel('Change')
fig.suptitle('Ratio data: plain log axis vs compressed axes')
fig.savefig('plots/demo_compressed_ratio_axes.png', dpi=150)
print('Saved plots/demo_compressed_ratio_axes.png')

# --- Amount data -------------------------------------------------------------
x = np.linspace(0, 1, 1000)
y = np.sign(rng.normal(size=x.size)) * np.exp(rng.normal(0, 4, x.size))
amount_scale = 1.0

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
for ax, kind in zip(axs, ['asinh', 'asinh2']):
    bounds, ticks, amounts = get_axis_bounds_and_ticks_arcsinh(
        [y.min(), y.max()], scale=amount_scale, kind=kind)
    ax.scatter(x, amount_to_arcsinh(y, amount_scale, kind), s=3, alpha=0.5)
    ax.set_ylim(bounds)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{a:g}' for a in amounts])
    ax.set_title(f'{kind} (scale = {amount_scale:g})')
axs[0].set_ylabel('Amount')
fig.suptitle('Amount data: asinh vs asinh2')
fig.savefig('plots/demo_compressed_amount_axes.png', dpi=150)
print('Saved plots/demo_compressed_amount_axes.png')
