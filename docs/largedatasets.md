# Handling Large Data Sets


!!! info

    If you want to run the code below, you need access to or download the full [TNG](https://www.tng-project.org) simulation dataset.
    The easiest way to access all TNG data sets is to use the [TNGLab](https://www.tng-project.org/data/lab/), which supports [scida](https://www.tng-project.org/data/forum/topic/742/scida-analysis-toolkit-example-within-tng-lab/).

Until now, we have applied our framework to a very small simulation.
However, what if we are working with a very large data set
(like the TNG50-1 cosmological simulation, which has $2160^3$ particles, $512$ times more than TNG50-4)?

## Preparation
First, we need to inform scida/dask about the resources it is allowed to use or allocate.
By default, scida would use all available memory and CPU cores, which is not always desired, particularly on shared systems such as HPC clusters or [TNGLab](https://www.tng-project.org/data/lab/).

scida provides a convenient `init_resources()` function that set up reasonable local resource defaults.
**Call `scida.init_resources()` immediately after importing scida** and before any other dask operations .
For TNGLab, we impose a default memory limit of 4GB via this mechanism. Monitor memory usage via the [dask dashboard](https://docs.dask.org/en/latest/dashboard.html).


## Starting simple: computing in chunks

First, we can still run the same calculation as above, and it will "just work" (hopefully).

This is because Dask has many versions of common algorithms and functions
which work on "blocks" or "chunks" of the data, which split up the large array into smaller arrays.
Work is needed on each chunk, after which the final answer is assembled.

Importantly, in our case above, even if the `mass` array above does not fit into memory,
the `mass.sum().compute()` will chunk the operation up in a way that the task can be calculated.

```pycon
>>> from scida import load
>>> sim = load("TNG50-1")
>>> ds = sim.get_dataset(99)
```

Before we start, let's enable a progress indicator from dask
(note that this will only work for local schedulers, see next section):

``` pycon
>>> from dask.diagnostics import ProgressBar
>>> ProgressBar().register()
```

Let's benchmark this operation on our local machine.

```pycon
>>> %time ds.data["PartType0"]["Masses"].sum().compute()
[########################################] | 100% Completed | 194.28 s
CPU times: user 12 s, sys: 16.2 s, total: 28.2 s
Wall time: 3min 16s
52722.6796875 code_mass
```

## More advanced: computing in parallel

Rather than sequentially calculating large tasks, we can also run the computation in parallel.

To do so different advanced dask schedulers are available.
Here, we use the most straight forward [distributed scheduler](https://docs.dask.org/en/latest/how-to/deploy-dask/single-distributed.html).

### Manual LocalCluster setup

If you need fine-grained control beyond what `scida.init_resources()` provides, you can manually configure the Local distributed scheduler:

```pycon
>>> from dask.distributed import Client, LocalCluster
>>> cluster = LocalCluster(n_workers=16, threads_per_worker=1,
                           memory_limit="4GB",
                           dashboard_address=":8787")
>>> client = Client(cluster)
>>> client
```

This is our client. We can access the scheduler on specified dashboard port to investigate its state.

We can now perform the same operations, but it is performed in a distributed manner, in parallel.

One significant advantage is that (even when using only a single node) individual workers will load just the subsets of data they need to work on, meaing that I/O operations become parallel.

Note: after creating a `Client()`, all calls to `.compute()` will automatically use this scheduler and its set of workers.

```pycon
>>> %time ds.data["PartType0"]["Masses"].sum().compute()
CPU times: user 5.11 s, sys: 1.42 s, total: 6.53 s
Wall time: 24.7 s

52722.6796875 code_mass
```

The progress bar, we could use for the default scheduler (before initializing `LocalCluster`),
is unavailable for the distributed scheduler.
However, we can still view the progress of this task as it executes using its status dashboard
(as a webpage in a new browser tab or within [jupyter lab](https://github.com/dask/dask-labextension)).
You can find it by clicking on the "Dashboard" link above.
If running this notebook server remotely, e.g. on a login node of a HPC cluster,
you may have to change the '127.0.0.1' part of the address to be the same machine name/IP.

### Running a SLURMCluster

If you are working with HPC resources, such as compute clusters with common schedulers (e.g. SLURM),
check out [Dask-Jobqueue](https://jobqueue.dask.org/en/latest/) to automatically batch jobs spawning dask workers.

Below is an example using the SLURMCluster.
We configure the job and node resources before submitting the job via the `scale()` method.

``` pycon
>>> from dask.distributed import Client
>>> from dask_jobqueue import SLURMCluster
>>> cluster = SLURMCluster(queue='p.large', cores=72, memory="500 GB",
>>>                        processes=36,
>>>                        scheduler_options={"dashboard_address": ":8811"})
>>> cluster.scale(jobs=1)  # submit 1 job for 1 node
>>> client = Client(cluster)

>>> from scida import load
>>> sim = load("TNG50-1")
>>> ds = sim.get_dataset(99)
>>> %time ds.data["PartType0"]["Masses"].sum().compute()
CPU times: user 1.27 s, sys: 152 ms, total: 1.43 s
Wall time: 21.4 s
>>> client.shutdown()
```

The SLURM job will be killed by invoking `client.shutdown()` or if the spawning python process or ipython kernel dies.
Make sure to properly handle exceptions, particularly in active jupyter notebooks, as allocated nodes might otherwise
idle and not be cleaned up.
