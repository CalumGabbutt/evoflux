"""
Copyright 2025 The Institute of Cancer Research.

Licensed under a software academic use license provided with this software package (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: https://github.com/CalumGabbutt/evoflux
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""

import pandas as pd
from evoflux import evoflux as ev
import os
import joblib
from joblib import delayed, Parallel
from multiprocess import cpu_count
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
import arviz as az
import itertools


def calculate_loglikelihood(y, res_samples, constants, mode, outsamplesdir, 
                            sample, Ncores=1, overwrite=False):

    loglfile = os.path.join(outsamplesdir, f'{sample}_loglikelihood.csv')

    if os.path.exists(loglfile) and not overwrite:
        print('Log-likelihood file already exists')
        logl = pd.read_csv(loglfile, header = None).values
    else:
        if os.path.exists(loglfile):
            print('Overwriting existing log-likelihood file')
        else:
            print("Log-likelihood file doesn't exist")  

        loglikelihood_wrapper = lambda params: ev.loglikelihood_perpoint(y, params, constants, mode)

        print('Calculating new LL matrix')
        if Ncores > 1:
            print(f'Running in parallel with { Ncores} cores')
            logl = np.vstack(Parallel(n_jobs=Ncores)(delayed(loglikelihood_wrapper)(s) for s in res_samples))
        else:
            print('Running in serial')
            logl = np.vstack([loglikelihood_wrapper(s) for s in res_samples])

        np.savetxt(loglfile, logl, delimiter=',') 

    return logl

def generate_yhat(posterior, constants, mode, Ncores=1):
    generate_data_wrapper = lambda params: ev.generate_data(params, constants, mode)

    if Ncores > 1:
        yhat = np.vstack(Parallel(n_jobs=Ncores)(delayed(generate_data_wrapper)(s) for s in posterior.values))
    else:
        yhat = np.vstack([generate_data_wrapper(s) for s in posterior.values])

    return yhat

def calculate_posterior_idx(posterior, res_samples):
    idx = np.zeros(posterior.shape[0], dtype = int)
    for i in range(posterior.shape[0]):
        posterior_i = posterior.iloc[i].values[np.newaxis, :]
        distances = cdist(posterior_i, res_samples)
        idx[i] = np.argmin(distances)

    return idx

def calculate_loo(
    y, 
    T, 
    outsamplesdir,
    sample,
    rho=1.0,
    Smin=10**2,
    Smax=10**9,
    NSIM=None,
    mode = 'neutral',
    Ncores = None,
    overwrite = False,
    save = True
):
    N = len(y)

    if NSIM is None:
        NSIM = 100000

    if Ncores is None:
        Ncores = cpu_count()
    else:
        Ncores = int(Ncores)

    constants = [rho, T, Smin, Smax, NSIM] 

    outsamples = os.path.join(outsamplesdir, f'{sample}_posterior.pkl')
    with open(outsamples, 'rb') as f:
        res = joblib.load(f)

    res_samples = res.samples

    logl = calculate_loglikelihood(y, res_samples, constants, mode,
                                outsamplesdir, sample, Ncores, overwrite)
    
    posterior = ev.extract_posterior(res, mode, outsamplesdir, sample, 
                                     overwrite)
    ndims = posterior.shape[1]
    pos = posterior.values[np.newaxis, :, :]    
    idx = calculate_posterior_idx(posterior, res_samples)
    
    # Ndraws = np.shape(res_samples)[0]
    Ndraws = posterior.shape[0]
    y_hat = np.empty((1, Ndraws, N))
    LL = np.empty((1, Ndraws, N))

    LL[0, :, :] = np.vstack([logl[i, :] for i in idx])
    y_hat[0, :, :] = generate_yhat(posterior, [rho, T, Smin, Smax, N], 
                                   mode, Ncores)

    pos = posterior.values[np.newaxis, :, :]

    inference = az.from_dict(
                    posterior = {posterior.columns[i]:pos[:, :, i] for i in range(ndims)},
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

    if save:
        netcdf = az.to_netcdf(inference, os.path.join(outsamplesdir, f'{sample}_fit.nc'))

    data_loo = az.loo(inference, pointwise=True)
    khat = az.plot_khat(data_loo, show_bins=True)
    if save:
        plt.savefig(os.path.join(outsamplesdir, f'{sample}_khat.png'), dpi=300)
        plt.close()

    return inference

def load_inference(inference_path):
    return az.from_netcdf(inference_path)

def model_selection(inference_list, outputdir, sample, labels=None, save=True):

    if labels is not None:
        if len(inference_list) != len(labels):
            raise ValueError(f"""
                             Expected the same number of labels as inference objects,
                             instead {len(labels)} were passed.
                             """)

    pairs = list(itertools.combinations(inference_list, 2))

    for pair in pairs:
        inference1, inference2 = pair
        if any(inference1.observed_data['y'].values != inference2.observed_data['y'].values):
            raise ValueError("The observed data in the two inference objects is different!")
    
    y = inference_list[0].observed_data['y'].values

    fig, ax = plt.subplots()      
    plt.hist(y, np.linspace(0, 1, 101), density=True, alpha=0.4, linewidth=0) 
    for inference in inference_list:
        plt.hist(np.ravel(inference.posterior_predictive['y_hat'].values), 
                np.linspace(0, 1, 101), density=True, alpha=0.4, linewidth=0) 
    if labels is not None:
        legend = ["Data"] + list(labels)
        plt.legend(legend)
    plt.xlabel("Fraction methylated")
    plt.ylabel("Probability density")
    plt.title(sample)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(outputdir, f"{sample}_compare_pospred.png"), 
                    dpi = 600)
        plt.close()

    if labels is None:
        compare_dict = {f"model {i}":inference for i, inference in enumerate(inference_list)}
    else:
        compare_dict = {labels[i]:inference for i, inference in enumerate(inference_list)}

    model_compare = az.compare(compare_dict, ic="loo", method="BB-pseudo-BMA", scale="log")

    if save:
        model_compare.to_csv(os.path.join(outputdir, f'{sample}_model_compare.csv'))

    az.plot_compare(model_compare)
    if save:
        plt.savefig(os.path.join(outputdir, f'{sample}_model_elpd.png'), dpi=600)
        plt.close()

    return model_compare