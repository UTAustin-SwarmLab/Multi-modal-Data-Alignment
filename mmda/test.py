from pathlib import Path

import numpy as np

dir_path = Path(
    "/nas/timeseries/timeseries_synthesis/sameep_store/btc/split_fresh_large_120"
)

a = np.load(dir_path / "test_continuous_conditions.npy")
print(a.shape)
