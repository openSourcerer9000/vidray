#!/usr/bin/env python3
"""
...and I will advertise it!

Manages launching a local Dask cluster with a configurable number of workers
and threads per worker.
"""

from dask.distributed import Client, progress
from subprocess import Popen

def Cliente(n_workers: int = 6, threads_per_worker: int = 1, **kwargz) -> Client:
    """
    Launches a local Dask cluster with the specified number of workers and threads per worker.
    Prints the dashboard link and opens it in Windows Explorer.

    :param n_workers: Number of worker processes.
    :param threads_per_worker: Number of threads per worker process.
    :param kwargz: Additional kwargs for dask.distributed.Client.
    :return: The launched Dask client.
    """
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, **kwargz)
    print(f"Dask client dashboard at: {client.dashboard_link}")
    # Open the dashboard in Windows Explorer (Windows only)
    Popen(f'explorer "{client.dashboard_link}"')
    return client
