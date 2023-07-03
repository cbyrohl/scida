# Series
In the tutorial section, we have only considered individual data sets. 
Often data sets are given in a series (e.g. multiple snapshots of a simulation, multiple exposures in a survey). 
Loading this as a series provides convenient access to all contained objects.

``` pycon
>>> from scida import load
>>> series = load("TNGvariation_simulation") #(1)!
```

1. Pass the base path of the simulation. 

We can now access the individual data sets from the series object:

``` pycon
>>> series[0] #(1)!
```

1. Alias for 'series.datasets[0]'

Depending on the available metadata, we can select data sets by these.

For example, cosmological simulations usually have information about their redshift:

``` pycon
>>> snp = series.get_dataset(redshift=2.0)
>>> snp.header["Redshift"]
2.0020281392528516
```


