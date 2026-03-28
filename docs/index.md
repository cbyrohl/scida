---
template: home.html
hide:
  - navigation
  - toc
  - footer
---

<section class="hero" markdown>
<div class="hero-body" markdown>
# scida

**Scalable analysis for large astrophysical datasets**
{ .hero-tagline }

Process cosmological simulations and observational data with dask-powered
parallel computing and automatic physical units.
{ .hero-subtitle }

<div class="hero-buttons">
  <a href="install/" class="md-button md-button--primary">Get Started</a>
  <a href="https://github.com/cbyrohl/scida" class="md-button">View on GitHub</a>
</div>

</div>
</section>

<section class="features-section" markdown>
<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### :material-lightning-bolt: One-Line Loading { .feature-title }

Load any supported dataset with a single function call. Automatic type detection selects the right handler.

```python
import scida
ds = scida.load("snapshot_099.hdf5")
```
</div>

<div class="feature-card" markdown>
### :material-chart-scatter-plot: Dask-Powered { .feature-title }

Scale from your laptop to HPC clusters. All data is loaded as lazy dask arrays, computed only when needed.

```python
masses = ds.data["PartType0"]["Masses"]
total = masses.sum().compute()  # runs in parallel
```
</div>

<div class="feature-card" markdown>
### :material-scale-balance: Physical Units { .feature-title }

Automatic unit support via pint. Easily compare results across different simulation codes and observational surveys with consistent physical units.

```python
ds = scida.load("snapshot_099.hdf5", units=True)
coords = ds.data["PartType0"]["Coordinates"]
coords_kpc = coords.to("kpc")
```
</div>

<div class="feature-card" markdown>
### :material-database-outline: Multiple Formats { .feature-title }

Native support for HDF5, zarr, and FITS. Works with AREPO, SWIFT, GIZMO, Gadget, and more out of the box.

[See supported datasets :material-arrow-right:](supported_data.md)
</div>

</div>
</section>
