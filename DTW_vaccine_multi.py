#!/usr/bin/env python

import datetime
import glob
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import click
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
from matplotlib import rcParams
from scipy import signal, stats

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance.subsequence.dtw import subsequence_alignment
from ont_fast5_api.fast5_interface import get_fast5_file

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


def find_sim(ref, fast5_path, verbose=True, cutoff=20000):
    """
    The major function for finding subsequence matches in raw nanopore signals.
    """

    # create results data.frame (pandas)
    results = pd.DataFrame(
        columns=['read_id', 'distance', 'startidx', 'endidx'])

    with get_fast5_file(fast5_path, mode="r") as f5:  # open fast5

        now = datetime.datetime.now()
        print("##### processing file: " + fast5_path + "  " + str(now))

        temp_data = {}
        for num, read in enumerate(f5.get_reads(), start=1):
            raw_data = read.get_raw_data(scale=True)

            # get only first  {cutoff} signals
            raw_data = np.array(
                raw_data[1:cutoff]).astype(float)

            # apply savitzky golay fiter
            raw_data = signal.savgol_filter(raw_data, 51, 3)

            # zscore normalize signals
            raw_data = stats.zscore(raw_data)

            # get read_id
            read_id = read.read_id

            # perform alignment using sequence_alignment from dtaidistance:
            # use_C - use C capabilities (faster) if compiled and available
            seq_align = subsequence_alignment(ref, raw_data, use_c=True)

            # get best match from alignment
            match = seq_align.best_match()

            # get start and end of alignment
            startidx, endidx = match.segment

            # if start and end are not close to each other
            if (endidx > startidx+10):
                # calculate distance using fast option
                distance = dtw.distance_fast(ref, raw_data[startidx:endidx])
                if (verbose):
                    print(str(num) + ": " + read_id + "dist: " +
                          str(distance) + "      " + str(os.getpid()))
            else:
                distance = 1000

            # add results to pandas data.frame
            # TODO: This has to be refactored to use dict.

            temp_data[num] = {
                'read_id': read_id,
                'distance': distance,
                'startidx': startidx,
                'endidx': endidx
            }

    results = pd.DataFrame.from_dict(
        temp_data, orient="index",
        columns=['read_id', 'distance', 'startidx', 'endidx']
    )

    # # output results of a given chnk to file in /tmp
    # temp_output = '/tmp/' + \
    #     os.path.basename(fast5_path) + '_' + str(os.getpid()) + '_DTW.tsv'
    # results.to_csv(temp_output, sep="\t")

    fin_now = datetime.datetime.now()

    print(
        "***** finished file: " + fast5_path + "  " +
        str(fin_now) + " (started: " + str(now) + ")"
    )

    return results


@ click.command()
@ click.option('--inpath', '-i', help='The input fast5 directory path')
@ click.option('--ref_signal', '-r', help='reference signal')
@ click.option('--shift_signal', '-s', default=0, help='shift reference signal by number of points')
@ click.option('--output', '-o', help='output file')
@ click.option('--threads', '-t', default=1, help='parallel threads to use')
@ click.option('--verbose', '-v', is_flag=True, default=False, help='Be verbose?')
def main(inpath, ref_signal, output, shift_signal, threads, verbose):

    # load reference signal
    ref_sig = np.loadtxt(ref_signal, dtype="float")
    # get only first 5000 signal points
    ref_sig = ref_sig[1+shift_signal:5000+shift_signal].astype(float)
    # apply savitzky golay fiter
    ref_sig = signal.savgol_filter(ref_sig, 51, 3)
    ref_sig = stats.zscore(ref_sig)  # zscore normalize signals
    print("Succesfully read reference signal")
    if (shift_signal > 0):
        print("Reference signal was shifted by " +
              str(shift_signal) + " data points")
    # create pandas data.rame for results
    fin_results = pd.DataFrame(
        columns=['read_id', 'distance', 'startidx', 'endidx'])

    futures = []  # initialize futures

    # get fast5 files from input path
    files = list(glob.glob(inpath + '/**/*.fast5', recursive=True))
    # files = []
    # for fileNM in glob.glob(inpath + '/**/*.fast5', recursive=True):
    #     # print(fileNM)
    #     files.append(fileNM)

    # start processes pool (futures)
    with ProcessPoolExecutor(max_workers=threads) as pool:
        results = list(
            pool.map(find_sim, repeat(ref_sig),
                     files, repeat(verbose))
        )

    # produce and save final results
    fin_results = pd.concat(results)
    fin_results.to_csv(output, sep="\t")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("ERROR. You have to provide arguments.\n")
        main.main(['--help'])
    else:
        main()
