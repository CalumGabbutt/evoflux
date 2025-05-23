#!/usr/bin/env python3

"""
Copyright 2025 The Institute of Cancer Research.

Licensed under a software academic use license provided with this software package (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: https://github.com/CalumGabbutt/evoflux
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""

import pandas as pd
import evoflux.evoloo as el
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Calculate leave-one-out')
    parser.add_argument('datafile', type=str,
                        help='path to csv containing beta values')
    parser.add_argument('patientinfofile', type=str,
                        help='path to csv containing patientinfo')
    parser.add_argument('outputdir', type=str, default='~', 
                        help='path to folder in which to store output')
    parser.add_argument('sample', type=str,
                        help='samplename of beta array (must be a col in datafile & index in patientinfo)')
    parser.add_argument('-Smin', default=10**2,
                        help='minimum allowed tumour size (default 10^2)')
    parser.add_argument('-Smax', default=10**9,
                        help='maximum allowed tumour size (default 10^9)')
    parser.add_argument('-NSIM', default=None,
                        help='Number of simulated fCpG loci per run (default:10e5)')
    parser.add_argument('-mode', default='neutral', 
                        help='Model evolutionary mode (default:neutral)')     
    parser.add_argument('-Ncores', default=None, 
                        help='Number of cores to use (default:cpu_count())')   
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help = 'overwrite an existing LL file, if it exists')

    # Execute the parse_args() method
    args = parser.parse_args()

    datafile = args.datafile
    patientinfofile = args.patientinfofile
    outputdir = args.outputdir
    sample = args.sample
    Smin = float(args.Smin)
    Smax = float(args.Smax)
    NSIM = args.NSIM
    mode = args.mode
    Ncores = args.Ncores
    overwrite = args.overwrite

    outsamplesdir = os.path.join(outputdir, sample)

    beta_values = pd.read_csv(datafile, index_col = 0)
    patientinfo = pd.read_csv(patientinfofile, index_col = 0) 

    y = beta_values[sample].dropna().values
    T = patientinfo.loc[sample, 'AGE_SAMPLING']

    rho = patientinfo.loc[sample, 'PURITY_TUMOR_CONSENSUS'] / 100

    if rho < 0.6:
        print("""
              Purity is assumed to be a percentage and EVOFLUx works best with 
              high purity samples, ensure you haven't given purity as a fraction! 
              """)

    if NSIM is None:
        NSIM = 100000
        print(f'{NSIM} samples per stochastic run')
    else:
        NSIM = int(NSIM)

    inference = el.calculate_loo(y, T, outsamplesdir, sample,
                    rho=rho, Smin=Smin, Smax=Smax,
                    NSIM=NSIM, mode=mode, 
                    Ncores = Ncores, overwrite=overwrite)

if __name__ == "__main__":
    main()