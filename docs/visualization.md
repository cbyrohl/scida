# Visualization

!!! info

    If you want to run the code below, consider downloading the [demo data](supported_datasets/tng.md#demo-data) or use the [TNGLab](supported_datasets/tng.md#tnglab) online.

## Creating plots

As we often use large datasets, we need to be careful with the amount of data we plot.
Generally, we reduce the data by either selecting a subset or reducing it prior to plotting.
For example, we can select a subset of particles by applying a cut on a given field.

```python title="Selecting a subset of particles"
from scida import load
import matplotlib.pyplot as plt

ds = load("./snapdir_030")
dens = ds.data["PartType0"]["Density"][:10000].compute()  # (1)!
temp = ds.data["PartType0"]["Temperature"][:10000].compute()
plt.plot(dens, temp, "o", markersize=0.1)
plt.xscale("log")
plt.yscale("log")
plt.show()
```

1. Note the subselection of the first 10000 particles and conversion to a numpy array. Replace this operation with a meaninguful selection operation (e.g. a certain spatial region selection).

Instead of subselection, we sometimes want to visualize all of the data. We can do so by first applying reduction operations using dask. A common example would be a 2D histogram.

```python title="2D histograms"
import dask.array as da
import numpy as np
from scida import load
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ds = load("./snapdir_030")
dens10 = da.log10(ds.data["PartType0"]["Density"].to("Msun/kpc^3").magnitude)
temp10 = da.log10(ds.data["PartType0"]["Temperature"].to("K").magnitude)

bins = [np.linspace(1, 12, 44 + 1), np.linspace(3.5, 8, 45 + 1)]
hist, xedges, yedges = da.histogram2d(dens10, temp10, bins=bins)
hist, xedges, yedges = hist.compute(), xedges.compute(), yedges.compute()
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.imshow(hist.T, origin="lower", extent=extent, aspect="auto", cmap="Greys",
          norm=LogNorm())
plt.xlabel(r"$\log_{10}$(density/(M$_\odot$/kpc$^3$))")
plt.ylabel(r"$\log_{10}$(temperature/K)")
plt.show()
```



## Interactive visualization

!!! info
    This example requires the `holoviews`, `datashader` and `bokeh` packages installed.
    Make sure that these [holoviews examples](https://holoviews.org/getting_started/index.html) work before continuing.


We can do interactive visualization with holoviews. For example, we can create a scatter plot of the particle positions.

```python
import holoviews as hv
import holoviews.operation.datashader as hd
import datashader as dshdr
from scida import load

ds = load("./snapdir_030")
ddf = ds.data["PartType0"].get_dataframe(["Coordinates0", "Coordinates1", "Masses"])  # (1)!

hv.extension("bokeh")
shaded = hd.datashade(hv.Points(ddf, ["Coordinates0", "Coordinates1"]), cmap="viridis", interpolation="linear",
                      aggregator=dshdr.sum("Masses"), x_sampling=5, y_sampling=5)
hd.dynspread(shaded, threshold=0.9, max_px=50).opts(bgcolor="black", xaxis=None, yaxis=None, width=500, height=500)
```

1. Visualization operations in holowview primarily run with dataframes, which we thus need to create using this wrapper for given fields.


![type:video](./videos/datashader_tng50.webm){: style='width: 100%; height: 400px; align: center;'}
