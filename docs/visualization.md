# Visualization
## Creating plots

As we often use large datasets, we need to be careful with the amount of data we plot.
Generally, we reduce the data by either selecting a subset or reducing it prior to plotting.
For example, we can select a subset of particles by applying a cut on a given field.

```python title="Selecting a subset of particles"
from astrodask import load
import matplotlib.pyplot as plt
ds = load("TNG50-1_snapshot")
dens = ds.data["PartType0"]["Density"][:10000].compute() # (1)!
temp = ds.data["PartType0"]["Temperature"][:10000].compute()
plt.plot(dens, temp, "o", markersize=0.1)
plt.show()
```

1. Note the subselection of the first 10000 particles and conversion to a numpy array. Replace this operation with a meaninguful selection operation (e.g. a certain spatial region selection).

Instead of subselection, we sometimes want to visualize all of the data. We can do so by first applying reduction operations using dask. A common example would be a 2D histogram.

```python title="2D histograms"
import dask.array as da
from astrodask import load
import matplotlib.pyplot as plt
ds = load("TNG50-1_snapshot")
dens = ds.data["PartType0"]["Density"]
temp = ds.data["PartType0"]["Temperature"]
hist, xedges, yedges = da.histogram2d(dens, temp, bins=100)
hist = hist.compute()
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(hist.T, origin="lower", extent=extent, aspect="auto")
plt.show()
```



## Interactive visualization

We can do interactive visualization with holoviews. For example, we can create a scatter plot of the particle positions.


```python
import holoviews as hv
import holoviews.operation.datashader as hd
import datashader as dshdr
from astrodask import load

ds = load("TNG50-1_snapshot")
ddf = ds.data["PartType0"].get_dataframe(["Coordinates0", "Coordinates1", "Masses"]) # (1)!

hv.extension("bokeh")
shaded = hd.datashade(hv.Points(ddf, ["Coordinates0", "Coordinates1"]), cmap="viridis", interpolation="linear",
                     aggregator=dshdr.sum("Masses"), x_sampling=5, y_sampling=5)
hd.dynspread(shaded, threshold=0.9, max_px=50).opts(bgcolor="black", xaxis=None, yaxis=None, width=500, height=500)
```

1. Visualization operations in holowview primarily run with dataframes, which we thus need to create using this wrapper for given fields.


![type:video](./videos/datashader_tng50.webm){: style='width: 100%; height: 400px; align: center;'}
