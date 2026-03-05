# axes

Utility functions for calculating axis bounds, tick marks, and plotting ratio data with percentage-change axes.

## Plotting Functions

### `plot_ratios(x_data, ratio_data, ..., colors=None)`

Plots ratio data (where 1.0 = no change) with a log-ratio y-axis labeled as percentage change. Each column of `ratio_data` is drawn as a separate line.

```python
import numpy as np
from axes import plot_ratios

x = np.arange(10)
ratios = np.column_stack([1 + 0.05 * x, 1 - 0.03 * x])
fig, ax = plot_ratios(x, ratios, x_axis_label='Year', y_axis_label='Change',
                      title='Two series', colors=['steelblue', 'darkorange'])
```

Optional `colors` argument accepts a list of matplotlib colors, one per column.

### `plot_ratios_shaded(x_data, ratio_data, ..., color=None)`

Plots ratio data with shaded uncertainty bands. The number of columns determines the visual style:

| Columns | Rendering |
|---------|-----------|
| 1 | Solid line |
| 3 | Shaded band + median line |
| 5 | Two nested bands (lighter outer, darker inner) + median line |
| 7 | Two nested bands + thin outer lines + median line |

All elements use a single color, distinguished by alpha and line width.

```python
import numpy as np
from axes import plot_ratios_shaded

x = np.linspace(0, 10, 100)
center = 1.0 + 0.2 * np.sin(x)
spread = 0.3 * x / x.max()
data5 = np.column_stack([center - 2*spread, center - spread, center,
                         center + spread, center + 2*spread])
fig, ax = plot_ratios_shaded(x, data5, title='Uncertainty bands', color='seagreen')
```

Optional `color` argument accepts any matplotlib color spec (default: first color from the default cycle).

## Axis Utility Functions

### `get_axis_bounds_and_ticks(data, padding=0.1)`

Computes axis bounds and evenly spaced "nice" tick positions for numerical data. Automatically selects tick spacing based on the data range and anchors ticks at zero when the range spans zero.

```python
from axes import get_axis_bounds_and_ticks

bounds, ticks = get_axis_bounds_and_ticks([-0.2, 0.8], padding=0.1)
```

### `get_axis_bounds_and_ticks_ln_pct(data, padding=0.1)`

For data in log-ratio space (`log(n/d)`), computes axis bounds and tick positions in log space with percentage-change labels. Useful when you want to plot log-ratio data but label the axis with percentage changes (e.g., -50%, 0%, +100%).

Tick labels are rounded to "nice" percentages via `round_pct_nice`.

```python
import numpy as np
from axes import get_axis_bounds_and_ticks_ln_pct

data = [np.log(0.5), np.log(3)]  # -50% to +200%
bounds, ticks, pct_labels = get_axis_bounds_and_ticks_ln_pct(data, padding=0.0)
# pct_labels: [-70, -60, -30, 0.0, 50, 100, 200]
```

### `round_pct_nice(pct)`

Rounds a percentage-change value to a "nice" number for axis labels:

- **0 to -90% and all positive values**: rounds to one significant digit (121 -> 100, 0.123 -> 0.1, -45 -> -40)
- **Below -90%**: rounds to successive nines (-95 -> -90, -99.3 -> -99, -99.95 -> -99.9)

## Demo Scripts

- `demo_plot.py` — generates demo PNGs for `plot_ratios()` with 1, 3, 5, and 7 columns
- `demo_plot_shaded.py` — generates demo PNGs for `plot_ratios_shaded()` with 1, 3, 5, and 7 columns

## Requirements

- Python 3
- NumPy
- Matplotlib
