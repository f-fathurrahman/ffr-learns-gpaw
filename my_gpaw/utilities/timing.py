
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import sys
import time
import math

import numpy as np
from ase.utils.timing import Timer

import my_gpaw.mpi as mpi


class NullTimer:
    """Compatible with Timer and StepTimer interfaces.  Does nothing."""
    def __init__(self):
        pass

    def print_info(self, calc):
        pass

    def start(self, name):
        pass

    def stop(self, name=None):
        pass

    def get_time(self, name):
        return 0.0

    def write(self, out=sys.stdout):
        pass

    def write_now(self, mark=''):
        pass

    def add(self, timer):
        pass

    def __call__(self, name):
        return self

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


nulltimer = NullTimer()


class DebugTimer(Timer):
    def __init__(self, print_levels=1000, comm=mpi.world, txt=sys.stdout):
        Timer.__init__(self, print_levels)
        ndigits = 1 + int(math.log10(comm.size))
        self.srank = '%0*d' % (ndigits, comm.rank)
        self.txt = txt

    def start(self, name):
        Timer.start(self, name)
        abstime = time.time()
        t = self.timers[tuple(self.running)] + abstime
        self.txt.write('T%s >> %15.8f %s (%7.5fs) started\n'
                       % (self.srank, abstime, name, t))

    def stop(self, name=None):
        if name is None:
            name = self.running[-1]
        abstime = time.time()
        t = self.timers[tuple(self.running)] + abstime
        self.txt.write('T%s << %15.8f %s (%7.5fs) stopped\n'
                       % (self.srank, abstime, name, t))
        Timer.stop(self, name)


class ParallelTimer(DebugTimer):
    """Like DebugTimer but writes timings from all ranks.

    Each rank writes to timings.<rank>.txt.  Also timings.metadata.txt
    will contain information about the parallelization layout.  The idea
    is that the output from this timer can be used for plots and to
    determine bottlenecks in the parallelization.

    See the tool gpaw-plot-parallel-timings."""
    def __init__(self, prefix='timings', flush=False):
        ndigits = len(str(mpi.world.size - 1))
        ranktxt = '%0*d' % (ndigits, mpi.world.rank)
        fname = '%s.%s.txt' % (prefix, ranktxt)
        txt = open(fname, 'w', buffering=1 if flush else -1)
        DebugTimer.__init__(self, comm=mpi.world, txt=txt)
        self.prefix = prefix

    def print_info(self, calc):
        """Print information about parallelization into a file."""
        fd = open('%s.metadata.txt' % self.prefix, 'w')
        DebugTimer.print_info(self, calc)
        wfs = calc.wfs

        # We won't have to type a lot if everyone just sends all their numbers.
        myranks = np.array([wfs.world.rank, wfs.kd.comm.rank,
                            wfs.bd.comm.rank, wfs.gd.comm.rank])
        allranks = None
        if wfs.world.rank == 0:
            allranks = np.empty(wfs.world.size * 4, dtype=int)
        wfs.world.gather(myranks, 0, allranks)
        if wfs.world.rank == 0:
            for itsranks in allranks.reshape(-1, 4):
                fd.write('r=%d k=%d b=%d d=%d\n' % tuple(itsranks))
        fd.close()


class HPMTimer(Timer):
    """HPMTimer requires installation of the IBM BlueGene/P HPM
    middleware interface to the low-level UPC library. This will
    most likely only work at ANL's BlueGene/P. Must compile
    with GPAW_HPM macro in customize.py. Note that HPM_Init
    and HPM_Finalize are called in _gpaw.c and not in the Python
    interface. Timer must be called on all ranks in node, otherwise
    HPM will hang. Hence, we only call HPM_start/stop on a list
    subset of timers."""

    top_level = 'GPAW.calculator'  # HPM needs top level timer
    compatible = ['Initialization', 'SCF-cycle']

    def __init__(self):
        Timer.__init__(self)
        from _gpaw import hpm_start, hpm_stop
        self.hpm_start = hpm_start
        self.hpm_stop = hpm_stop
        hpm_start(self.top_level)

    def start(self, name):
        Timer.start(self, name)
        if name in self.compatible:
            self.hpm_start(name)

    def stop(self, name=None):
        Timer.stop(self, name)
        if name in self.compatible:
            self.hpm_stop(name)

    def write(self, out=sys.stdout):
        Timer.write(self, out)
        self.hpm_stop(self.top_level)


class CrayPAT_timer(Timer):
    """Interface to CrayPAT API. In addition to regular timers,
    the corresponding regions are profiled by CrayPAT. The gpaw-python has
    to be compiled under CrayPAT.
    """

    def __init__(self, print_levels=4):
        Timer.__init__(self, print_levels)
        from _gpaw import craypat_region_begin, craypat_region_end
        self.craypat_region_begin = craypat_region_begin
        self.craypat_region_end = craypat_region_end
        self.regions = {}
        self.region_id = 5  # leave room for regions in C

    def start(self, name):
        Timer.start(self, name)
        if name in self.regions:
            id = self.regions[name]
        else:
            id = self.region_id
            self.regions[name] = id
            self.region_id += 1
        self.craypat_region_begin(id, name)

    def stop(self, name=None):
        Timer.stop(self, name)
        id = self.regions[name]
        self.craypat_region_end(id)
