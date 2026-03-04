# axes

Utility functions for calculating axis bounds and tick marks for plots.

## Functions

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

## Requirements

- Python 3
- NumPy
