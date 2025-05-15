"""
EVOFLUx is © 2025, Calum Gabbutt

EVOFLUx is published and distributed under the Academic Software License v1.0 (ASL).

EVOFLUx is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.

You should have received a copy of the ASL along with this program; if not, email Calum Gabbutt at calum.gabbutt@icr.ac.uk. It is also published at https://github.com/gabor1/ASL/blob/main/ASL.md.

You may contact the original licensor at calum.gabbutt@icr.ac.uk.
"""

import pandas as pd
from evoflux import evoflux as ev
import os
import joblib
from joblib import delayed, Parallel
from multiprocess import cpu_count
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import numpy as np
import arviz as az
import corner

def plot_posterior_predictive(y, df, constants, mode, outsamplesdir, sample, Ncores=1,
                              save = True):

    generate_data_wrapper = lambda params: ev.generate_data(params, constants, mode)

    if Ncores > 1:
        y_hat = np.vstack(Parallel(n_jobs=Ncores)(delayed(generate_data_wrapper)(s) for s in df.values))
    else:
        y_hat = np.vstack([generate_data_wrapper(s) for s in df.values])

    fig, ax = plt.subplots()      
    plt.hist(y, np.linspace(0, 1, 101), density=True, alpha=0.4, linewidth=0) 
    plt.hist(np.ravel(y_hat), np.linspace(0, 1, 101), density=True, alpha=0.4, linewidth=0) 
    plt.legend(("Data", "Posterior predictive"))
    plt.xlabel("Fraction methylated")
    plt.ylabel("Probability density")
    plt.title(sample)
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(outsamplesdir, f"{sample}_posterior_predictive.png"), dpi = 600)
        plt.close()

def plot_prior_shrinkage(df, scales, mode, outsamplesdir, sample, save = True):

    if mode.lower() == 'neutral':
        prior_function = lambda flat_prior: ev.prior_transform_neutral(flat_prior, scales)
    elif mode.lower() == 'subclonal':
        prior_function = lambda flat_prior: ev.prior_transform_subclonal(flat_prior, scales)
    elif mode.lower() == 'independent':
        prior_function = lambda flat_prior: ev.prior_transform_independent(flat_prior, scales)
    else:
        raise ValueError("mode must be one of 'neutral', 'subclonal' or 'independent")
    
    ndims = df.shape[1]
    prior = np.array([prior_function(np.random.rand(ndims)) for i in range(10000)])

    fig, axes = plt.subplots(1, ndims, figsize = (16, 4))
    for i, var in enumerate(df.columns):
        axes[i].hist(df.values[:, i], bins=11, alpha=0.4, density=True)
        sns.kdeplot(prior[:, i], ax = axes[i])
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('')
    axes[0].set_ylabel('Probability Density')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(outsamplesdir, f"{sample}_posterior_shrinkage.png"), dpi = 600)
        plt.close()

def plot_trace(res, outsamplesdir, sample, labels, save = True):
    fig, axes = dyplot.traceplot(res, show_titles=True,
                                trace_cmap='viridis', connect=True,
                                connect_highlight=range(5), labels=labels)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(outsamplesdir, f'{sample}_traceplot.png'), dpi=600)
        plt.close()

def plot_cornerplot(res, outsamplesdir, sample, labels, save = True):
    # plot dynesty cornerplot
    fig, ax = dyplot.cornerplot(res, color='blue', show_titles=True,
                            max_n_ticks=3, quantiles=None, labels=labels)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(outsamplesdir, f'{sample}_cornerplot.png'), dpi=600)
        plt.close()

def plot_corner(df, outsamplesdir, sample, save = True):
    # Make the base corner plot
    figure = corner.corner(df.values, bins=7, smooth=1, labels=df.columns)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(outsamplesdir, f'{sample}_pairs.png'), dpi=600)
        plt.close()

def plot_all(
    y, 
    T, 
    outsamplesdir,
    sample,
    rho=1.0,
    Smin=10**2,
    Smax=10**9,
    thetamean=3.0, 
    thetastd=2.0,
    muscale=0.05, 
    gammascale=0.05, 
    NSIM=None,
    mode = 'neutral',
    Ncores = None,
):

    if NSIM is None:
        NSIM = len(y)

    if Ncores is None:
        Ncores = cpu_count()
    else:
        Ncores = int(Ncores)

    # set the std of the halfnormal priors on lam, mu, gamma
    scales = [thetamean, thetastd, muscale, gammascale]
    constants = [rho, T, Smin, Smax, NSIM] 

    outsamples = os.path.join(outsamplesdir, f'{sample}_posterior.pkl')
    with open(outsamples, 'rb') as f:
        res = joblib.load(f)

    df = ev.extract_posterior(res, mode, outsamplesdir, sample)
    labels = df.columns.to_list()

    plot_trace(res, outsamplesdir, sample, labels)
    plot_cornerplot(res, outsamplesdir, sample, labels)
    plot_corner(df, outsamplesdir, sample)

    plot_posterior_predictive(y, df, constants, mode, outsamplesdir, sample, Ncores)
    plot_prior_shrinkage(df, scales, mode, outsamplesdir, sample)
