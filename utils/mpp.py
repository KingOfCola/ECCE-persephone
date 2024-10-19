# -*-coding:utf-8 -*-
"""
@File    :   mpp.py
@Time    :   2024/10/09 14:11:54
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Multiprocessing utilities
"""

# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap

    From https://stackoverflow.com/a/57364423/25980698
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap
