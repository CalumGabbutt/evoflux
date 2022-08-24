#!/usr/bin/env python3

import pandas as pd
from flipflopblood import flipflopblood as fb
import os
import joblib

import argparse

def main():
    parser = argparse.ArgumentParser(description='Run flipflopblood Bayesian inference.')
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
    parser.add_argument('-nlive', default=300, dest='nlive', type=int,
                        help='number of live points in dynesty sampler (default:300)')
    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose')
    parser.add_argument('-thetamean', default=3.0, type=float,
                        help='prior mean growth rate (default:3.0)')
    parser.add_argument('-thetastd', default=2.0, type=float,
                        help='prior standard deviation on growth rate (default:2.0)')
    parser.add_argument('-muscale', default=0.05, type=float,
                        help='scale of methylation rate (default:0.05)')
    parser.add_argument('-gammascale', default=0.05, type=float,
                        help='scale of methylation rate (default:0.05)')
    parser.add_argument('-NSIM', default=None,
                        help='Number of simulated fCpG loci per run (default:len(y))')
    parser.add_argument('-dlogz', default=0.5, type = float,
                        help='dlogz stopping criteria (default:0.5)')   

    # Execute the parse_args() method
    args = parser.parse_args()

    datafile = args.datafile
    patientinfofile = args.patientinfofile
    outputdir = args.outputdir
    sample = args.sample
    Smin = float(args.Smin)
    Smax = float(args.Smax)
    nlive = int(args.nlive)
    verbose = args.verbose
    thetamean=float(args.thetamean)
    thetastd=float(args.thetastd)
    muscale=float(args.muscale)
    gammascale=float(args.gammascale)
    NSIM = args.NSIM
    dlogz = float(dlogz)

    outsamplesdir = os.path.join(outputdir, sample)
    outsamples = os.path.join(outsamplesdir, f'{sample}_posterior.pkl')

    os.makedirs(outsamplesdir, exist_ok=True)

    beta_values = pd.read_csv(datafile, index_col = 0)
    patientinfo = pd.read_csv(patientinfofile, index_col = 0) 

    beta = beta_values[sample].dropna().values
    T = patientinfo.loc[sample, 'Age']

    rho = patientinfo.loc[sample, 'Purity.Methylation(B-cell.proportion)']

    if NSIM is None:
        NSIM = len(beta)
        print(f'{NSIM} samples per stochastic run')
    else:
        NSIM = int(NSIM)

    res = fb.run_inference(beta, 
                            T, 
                            rho=rho,
                            Smin=Smin,
                            Smax=Smax,
                            nlive=nlive, 
                            thetamean=thetamean, 
                            thetastd=thetastd,
                            muscale=muscale, 
                            gammascale=gammascale, 
                            NSIM=NSIM,
                            verbose=verbose,
                            dlogz=dlogz)

    with open(outsamples, 'wb') as f:
        joblib.dump(res, f)

if __name__ == "__main__":
    main()