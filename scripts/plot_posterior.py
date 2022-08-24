#!/usr/bin/env python3

import pandas as pd
from flipflopblood import flipflopblood as fb
import os
import joblib
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import numpy as np
import arviz as az
import corner

import argparse

def main():
    parser = argparse.ArgumentParser(description='Plot posterior')
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

    # Execute the parse_args() method
    args = parser.parse_args()

    datafile = args.datafile
    patientinfofile = args.patientinfofile
    outputdir = args.outputdir
    sample = args.sample
    Smin = float(args.Smin)
    Smax = float(args.Smax)
    thetamean=float(args.thetamean)
    thetastd=float(args.thetastd)
    muscale=float(args.muscale)
    gammascale=float(args.gammascale)
    NSIM = args.NSIM

    outsamplesdir = os.path.join(outputdir, sample)
    outsamples = os.path.join(outsamplesdir, f'{sample}_posterior.pkl')

    beta_values = pd.read_csv(datafile, index_col = 0)
    patientinfo = pd.read_csv(patientinfofile, index_col = 0) 

    y = beta_values[sample].dropna().values
    T = patientinfo.loc[sample, 'T']
    N = len(y)

    rho = patientinfo.loc[sample, 'Purity.Methylation(B-cell.proportion)']

    scales = [thetamean, thetastd, muscale, gammascale]

    if NSIM is None:
        NSIM = len(y)
        print(f'{NSIM} samples per stochastic run')
    else:
        NSIM = int(NSIM)

    with open(outsamples, 'rb') as f:
        res = joblib.load(f)

    samples =  dynesty.utils.resample_equal(res.samples, softmax(res.logwt))
    labels = ["theta", "tau_rel", "mu", "gamma", "nu_rel", "zeta_rel",
                "contam_meth", "delta", "eta", "kappa"]

    df = pd.DataFrame(samples, columns = labels)
    df.to_csv(os.path.join(outsamplesdir, f'{sample}_posterior.csv'), index = False)


    fig, axes = plt.subplots(ndims, figsize=(10, 7), sharex=True)
    for i in range(ndims):
        ax = axes[i]
        ax.plot(samples[:, i], "k", alpha=0.3)
        ax.set_xlim(0, np.shape(samples)[0])
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig(os.path.join(outsamplesdir, f'{sample}_trace.png'), dpi=600)


    fig, axes = dyplot.traceplot(res, show_titles=True,
                                trace_cmap='viridis', connect=True,
                                connect_highlight=range(5), labels=labels)
    plt.tight_layout()
    plt.savefig(os.path.join(outsamplesdir, f'{sample}_traceplot.png'), dpi=600)
    plt.close()

    # plot dynesty cornerplot
    fig, ax = dyplot.cornerplot(res, color='blue', show_titles=True,
                            max_n_ticks=3, quantiles=None, labels=labels)
    plt.tight_layout()
    plt.savefig(os.path.join(outsamplesdir, f'{sample}_cornerplot.png'), dpi=600)
    plt.close()

    ndims = np.shape(df)[1]

    # Make the base corner plot
    figure = corner.corner(df.values, bins=7, smooth=1, labels=df.columns)
    # Extract the axes
    axes = np.array(figure.axes).reshape((ndims, ndims))
    plt.tight_layout()
    plt.savefig(os.path.join(outsamplesdir, f'{patient}_pairs.png'), dpi=600)


    Ndraws = np.shape(samples)[0]
    y_hat = np.empty((1, Ndraws, N))
    LL = np.empty((1, Ndraws, N))

    for i in range(Ndraws):

        (theta_sample, tau_rel_sample, mu_sample, gamma_sample,
        nu_rel_sample, zeta_rel_sample,
        contam_meth_sample, delta_sample, eta_sample, kappa_sample) = samples[i, :]

        nu_sample = nu_rel_sample * mu_sample
        zeta_sample = zeta_rel_sample * gamma_sample

        LL[0, i, :] = fb.loglikelihood_perpoint(theta_sample, rho, tau_rel_sample, 
                                mu_sample, gamma_sample, nu_rel_sample, 
                                zeta_rel_sample, contam_meth_sample, 
                                delta_sample, eta_sample, kappa_sample, y, T, 
                                Smin, Smax, NSIM)

        betaProb = fb.stochastic_growth(theta_sample, rho, T * tau_rel_sample, 
                                    mu_sample, gamma_sample, nu_sample, zeta_sample,
                                    contam_meth_sample, T, NSIM)


        y_hat[0, i, :] = fb.add_noise(betaProb, delta_sample, eta_sample, kappa_sample)

    fig, ax = plt.subplots()      
    plt.hist(y, np.linspace(0, 1, 101), density=True, alpha=0.4, linewidth=0) 
    plt.hist(np.ravel(y_hat), np.linspace(0, 1, 101), density=True, alpha=0.4, linewidth=0) 
    plt.legend(("Data", "Posterior predictive"))
    plt.xlabel("Fraction methylated")
    plt.ylabel("Probability density")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(outsamplesdir, f"{sample}_posterior_predictive.png"), dpi = 600)
    plt.close()

    posterior = samples[np.newaxis, :, :]

    inference = az.from_dict(
        posterior = {labels[i]:posterior[:, :, i] for i in range(ndims)},
        observed_data={'y':y},
        posterior_predictive={'y_hat':y_hat},
        constant_data={"rho":rho,
                        "T":T,
                        "Smin":Smin,
                        "Smax":Smax,
                        "NSIM":NSIM},
        sample_stats={'lp':np.sum(LL, axis=2)},
        log_likelihood={"log_likelihood":LL}
    )     

    # netcdf = az.to_netcdf(inference, os.path.join(outsamplesdir, f'{sample}_posterior.nc'))

    pairs = az.plot_pair(inference) 
    plt.savefig(os.path.join(outsamplesdir, f'{sample}_az_pairs.png'), dpi=600)
    plt.close()

    trace = az.plot_trace(inference) 
    plt.savefig(os.path.join(outsamplesdir, f'{sample}_az_trace.png'), dpi=600)
    plt.close()

    loo_pit = az.plot_loo_pit(inference, y='y', y_hat='y_hat')
    plt.savefig(os.path.join(outsamplesdir, f'{sample}_loo_pit.png'), dpi=600)
    plt.close()

    prior = np.array([fb.prior_transform(np.random.rand(ndims), scales) for i in range(10000)])

    fig, axes = plt.subplots(1, np.shape(samples)[1], figsize = (16, 4))
    for i, var in enumerate(labels):
        axes[i].hist(samples[:, i], bins=11, alpha=0.4, density=True)
        sns.kdeplot(prior[:, i], ax = axes[i])
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('')
    axes[0].set_ylabel('Probability Density')
    plt.tight_layout()
    plt.savefig(os.path.join(outsamplesdir, f"{sample}_posterior_shrinkage.png"), dpi = 600)
    plt.close()

    theta = df["theta"]

    fig, ax = plt.subplots()
    plt.hist(theta, bins=np.linspace(0, 10, 101), alpha=0.4, density=True)
    plt.hist(prior[:, 0], bins=np.linspace(0, 10, 101), alpha=0.4, density=True)
    plt.legend(("Posterior", "Prior"))
    ax.set_xlim([0, 10])
    plt.ylabel('Probability Density')
    plt.xlabel('Theta')
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(outsamplesdir, f"{sample}_theta_shrinkage.png"), dpi = 600)
    plt.close()

if __name__ == "__main__":
    main()