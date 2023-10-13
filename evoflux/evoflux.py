from scipy import stats, linalg
import numpy as np
import pandas as pd
import os
from scipy.special import logsumexp, gammaln, logit, softmax
from time import time
import dynesty
from dynesty import NestedSampler
from dynesty.results import print_fn
from multiprocess import Pool, cpu_count
import joblib

def generate_next_timepoint(m, k, w, mu, gamma, nu, zeta, S, dt, rng=None):
    """
    Simulates the transitions between the homozygous demtheylated, heterozygous
    and homozygous methylated states in a time step dt in a pool of S cells.

    Arguments:
        m: number of homozygous methylated cells - array of the ints
        k: number of heterozygous methylated cells - array of the ints
        w: number of homozygous demethylated cells - array of the ints
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu: rate to transition from heterozygous to homozygous methylated
            - float >= 0
        zeta: rate to transition from heterozygous to homozygous demethylated
            - float >= 0         
        S: total number of cells all(m + k + w == S) - int
        dt: time step - float > 0
        rng: np.random.default_rng() object, Optional
    Returns:
        Updated m, k, w after transitions have occurred
    """

    if rng is None:
        rng = np.random.default_rng()

    NSIM = len(m)

    # Use sequential rounds of binomial sampling to calculate how many cells
    # transition between each state
    m_to_k, k_out, w_to_k = rng.binomial(
                                    n = (m, k, w), 
                                    p = np.tile([2*gamma*dt, 
                                        (nu + zeta)*dt, 2*mu*dt], [NSIM, 1]).T)

    k_to_m = rng.binomial(n=k_out, p = np.repeat(nu / (nu + zeta), NSIM))

    m = m - m_to_k + k_to_m
    k = k - k_out + m_to_k + w_to_k
    w = S - m - k

    return (m, k, w)

def multinomial_rvs(counts, p, rng=None):
    """
    Simulate multinomial sampling of D dimensional probability distribution

    Arguments:
        counts: number of draws from distribution - int or array of the 
                ints (N)
        p: probability  - array of the floats (D, N)
        rng: np.random.default_rng() object, Optional
    Returns:
        Multinomial sample
    """

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(counts, (np.ndarray)):
        counts = np.full(p[0, ...].shape, counts)

    out = np.zeros(np.shape(p), dtype=int)
    ps = np.cumsum(p[::-1, ...], axis=0)[::-1, ...]
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0

    for i in range(p.shape[0]-1):
        binsample = rng.binomial(counts, condp[i, ...])
        out[i, ...] = binsample
        counts -= binsample

    out[-1, ...] = counts

    return out

def initialise_cancer(tau, mu, gamma, nu, zeta, NSIM, rng=None):
    """
    Initialise a cancer, assigning fCpG states assuming fCpGs are homozygous 
    at t=0

    Arguments:
        tau: age when population began expanding exponentially - float
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu: rate to transition from heterozygous to homozygous methylated
            - float >= 0
        zeta: rate to transition from heterozygous to homozygous demethylated
            - float >= 0         
        NSIM: number of fCpG loci to simulate - int
        rng: np.random.default_rng() object, Optional
    Returns:
        m_cancer, k_cancer, w_cancer: number of homo meth, hetet meth and 
                homo unmeth cells in the population - np.array[int]
    """

    if rng is None:
        rng = np.random.default_rng()

    # assume fCpG's are homozygous methylated at t=0
    mkw = np.zeros((3, NSIM), dtype = int)
    idx = np.arange(NSIM)
    np.random.shuffle(idx)
    mkw[0, idx[:NSIM//2]] = 1
    mkw[2, idx[NSIM//2:]] = 1

    # generate distribution of fCpG loci when population begins growing 
    # at t=tau
    RateMatrix = np.array([[-2*gamma, nu, 0], 
                            [2*gamma, -(nu+zeta), 2*mu], 
                            [0, zeta, -2*mu]])

    ProbStates = linalg.expm(RateMatrix * tau) @ mkw

    m_cancer, k_cancer, w_cancer = multinomial_rvs(1, ProbStates, rng)

    return m_cancer, k_cancer, w_cancer

def grow_cancer(m_cancer, k_cancer, w_cancer, S_cancer_i, S_cancer_iPlus1, rng):
    """
    Grow a cancer, assigning fCpG states according to a multinomial ditribution

    Arguments:
        m_cancer, k_cancer, w_cancer: number of homo meth, hetet meth and 
                homo unmeth cells in the population - np.array[int]
        S_cancer_i: number of cells at time t - int = m_cancer + k_cancer + w_cancer
        S_cancer_iPlus1: number of cells at time t+dt - int >= S_cancer_i
        rng: np.random.default_rng() object, Optional
    Returns:
        Updated m_cancer, k_cancer, w_cancer
    """

    if rng is None:
        rng = np.random.default_rng()

    if S_cancer_iPlus1 - S_cancer_i > 0:
        prob_matrix = np.stack((m_cancer, k_cancer, w_cancer)) / S_cancer_i
        growth = multinomial_rvs(S_cancer_iPlus1 - S_cancer_i, prob_matrix, rng)

        m_cancer += growth[0, :]
        k_cancer += growth[1, :]
        w_cancer += growth[2, :]

    return m_cancer, k_cancer, w_cancer

def stochastic_growth(theta, tau, mu, gamma, nu, zeta, T, NSIM):
    """
    Simulate the methylation distribution of fCpG loci for an exponentially 
    growing well-mixed population evolving neutrally

    Arguments:
        theta: exponential growth rate of population - float
        tau: age when population began expanding exponentially - float < T
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu: rate to transition from heterozygous to homozygous methylated
            - float >= 0
        zeta: rate to transition from heterozygous to homozygous demethylated
            - float >= 0         
        T: patient's age - float
        NSIM: number of fCpG loci to simulate - int
    Returns:
        betaCancer: fCpG methylation fraction distribution - np.array[float]
    """

    # calculate the time step so all transition probabilities are <= 10%
    dt_max = 0.1 / np.max((
        2*gamma, 
        2*mu,
        2*nu,
        2*zeta,
        theta)
    )
    
    # calculate deterministic exponential growth population size
    n = int((T-tau) / dt_max) + 2  # Number of time steps.
    t = np.linspace(tau, T, n) 
    dt = t[1] - t[0]
    S_cancer = np.exp(theta * (t-tau)).astype(int)

    if np.any(S_cancer < 0):
        raise(OverflowError('overflow encountered for S_cancer'))

    rng = np.random.default_rng()

    # generate distribution of fCpG loci when population begins growing 
    # at t=tau, assuming fCpG's are homozygous methylated at t=0
    m_cancer, k_cancer, w_cancer = initialise_cancer(tau, mu, gamma, nu, zeta, 
                                                     NSIM, rng)

    # simulate changes to methylation distribution by splitting the process 
    # into 2 phases, an exponential growth phase and a methylation transition 
    # phase
    for i in range(len(t)-1):
        m_cancer, k_cancer, w_cancer = grow_cancer(m_cancer, k_cancer,
                                                    w_cancer, S_cancer[i], 
                                                    S_cancer[i+1], rng)

        m_cancer, k_cancer, w_cancer = generate_next_timepoint(m_cancer, 
                                                    k_cancer, w_cancer, 
                                                    mu, gamma, nu, zeta,
                                                    S_cancer[i+1], dt, rng)


    with np.errstate(divide='raise', over='raise'):
        betaCancer = (k_cancer + 2*m_cancer) / (2*S_cancer[-1])

    return betaCancer



def stochastic_growth_subclonal(theta1, theta2, tau1, tau2, mu,
                                gamma, nu, zeta, T, NSIM):
    """
    Simulate the methylation distribution of fCpG loci for an exponentially 
    growing well-mixed population evolving with an expanding subclone

    Arguments:
        theta1: exponential growth rate of base population - float
        theta2: exponential growth rate of fitter subclone - float
        tau1: age when population began expanding exponentially - float < T
        tau2: age when subclone began expanding - float < T, > tau1
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu: rate to transition from heterozygous to homozygous methylated
            - float >= 0
        zeta: rate to transition from heterozygous to homozygous demethylated
            - float >= 0         
        T: patient's age - float
        NSIM: number of fCpG loci to simulate - int
    Returns:
        betaCancer: fCpG methylation fraction distribution - np.array[float]
    """

    dt_max1 = 0.1 / np.max((
        2*gamma, 
        2*mu,
        2*nu,
        2*zeta,
        theta1)
    )

    dt_max2 = 0.1 / np.max((
        2*gamma, 
        2*mu,
        2*nu,
        2*zeta,
        theta2)
    )
    
    n1, n2 = int((tau2-tau1) / dt_max1) + 2, int((T-tau2) / dt_max2) + 2  # Number of time steps.
    t1, t2 = np.linspace(tau1, tau2, n1), np.linspace(tau2, T, n2) 
    dt1, dt2 = t1[1] - t1[0], t2[1] - t2[0]
    S_cancer = np.exp(theta1 * (t1-tau1)).astype(int)
    S_cancer1 = np.exp(theta1 * (t2-tau1)).astype(int)
    S_cancer2 = np.exp(theta2 * (t2-tau2)).astype(int)

    if np.any(S_cancer1 <= 0):
        raise OverflowError("The number of cells has overflowed")  

    rng = np.random.default_rng()

    # generate distribution of fCpG loci when population begins growing 
    # at t=tau, assuming fCpG's are homozygous methylated at t=0
    m_cancer1, k_cancer1, w_cancer1 = initialise_cancer(tau1, mu, gamma, nu, zeta, 
                                                     NSIM, rng)

    # simulate changes to methylation distribution by splitting the process 
    # into 2 phases, an exponential growth phase and a methylation transition 
    # phase
    for i in range(len(t1)-1):
        m_cancer1, k_cancer1, w_cancer1 = grow_cancer(m_cancer1, k_cancer1,
                                                    w_cancer1, S_cancer[i], 
                                                    S_cancer[i+1], rng)

        m_cancer1, k_cancer1, w_cancer1 = generate_next_timepoint(m_cancer1, 
                                                    k_cancer1, w_cancer1, 
                                                    mu, gamma, nu, zeta,
                                                    S_cancer[i+1], dt1, rng)

    prob_matrix = np.stack((m_cancer1, k_cancer1, w_cancer1)) / S_cancer[-1]
    m_cancer2, k_cancer2, w_cancer2 = multinomial_rvs(1, prob_matrix, rng)

    for i in range(len(t2)-1):
        m_cancer1, k_cancer1, w_cancer1 = grow_cancer(m_cancer1, k_cancer1,
                                            w_cancer1, S_cancer1[i], 
                                            S_cancer1[i+1], rng)


        m_cancer2, k_cancer2, w_cancer2 = grow_cancer(m_cancer2, k_cancer2,
                                            w_cancer2, S_cancer2[i], 
                                            S_cancer2[i+1], rng)

        m_cancer1, k_cancer1, w_cancer1 = generate_next_timepoint(m_cancer1, 
                                                    k_cancer1, w_cancer1, 
                                                    mu, gamma, nu, zeta,
                                                    S_cancer1[i+1], dt2, rng)

        m_cancer2, k_cancer2, w_cancer2 = generate_next_timepoint(m_cancer2, 
                                                    k_cancer2, w_cancer2, 
                                                    mu, gamma, nu, zeta,
                                                    S_cancer2[i+1], dt2, rng)

    with np.errstate(divide='raise', over='raise'):
        betaCancer = ((k_cancer1 + k_cancer2 + 2*m_cancer1 + 2*m_cancer2) / 
                        (2*S_cancer1[-1] + 2*S_cancer2[-1]))

    return betaCancer

def stochastic_growth_independent(theta1, theta2, tau1, tau2, mu1, gamma1, 
                        nu1, zeta1, mu2, gamma2, nu2, zeta2, T, NSIM):
    """
    Simulate the methylation distribution of fCpG loci for 2 independent
    exponentially growing well-mixed populations

    Arguments:
        theta1: exponential growth rate of base population - float
        theta2: exponential growth rate of fitter second cancer - float
        tau1: age when population began expanding exponentially - float < T
        tau2: age when second cancer began expanding - float < T, > tau1
        mu1: rate to transition from homozygous demethylated to heterozygous 
                in cancer 1 - float >= 0
        gamma1: rate to transition from homozygous methylated to heterozygous
                in cancer 1  - float >= 0
        nu1: rate to transition from heterozygous to homozygous methylated
                in cancer 1 - float >= 0
        zeta1: rate to transition from heterozygous to homozygous demethylated
                in cancer 1 - float >= 0  
        mu2: rate to transition from homozygous demethylated to heterozygous 
                in cancer 2 - float >= 0
        gamma2: rate to transition from homozygous methylated to heterozygous
                in cancer 2  - float >= 0
        nu2: rate to transition from heterozygous to homozygous methylated
                in cancer 2 - float >= 0
        zeta2: rate to transition from heterozygous to homozygous demethylated
                in cancer 2 - float >= 0  
        T: patient's age - float
        NSIM: number of fCpG loci to simulate - int
    Returns:
        betaCancer: fCpG methylation fraction distribution - np.array[float]
    """

    dt_max1 = 0.1 / np.max((
        2*gamma1, 
        2*mu1,
        2*nu1,
        2*zeta1,
        theta1)
    )

    dt_max2 = 0.1 / np.max((
        2*gamma1, 
        2*mu1,
        2*nu1,
        2*zeta1,
        2*gamma2, 
        2*mu2,
        2*nu2,
        2*zeta2,
        theta2)
    )
    
    n1, n2 = int((tau2-tau1) / dt_max1) + 2, int((T-tau2) / dt_max2) + 2  # Number of time steps.
    t1, t2 = np.linspace(tau1, tau2, n1), np.linspace(tau2, T, n2) 
    dt1, dt2 = t1[1] - t1[0], t2[1] - t2[0]
    S_cancer = np.exp(theta1 * (t1-tau1)).astype(int)
    S_cancer1 = np.exp(theta1 * (t2-tau1)).astype(int)
    S_cancer2 = np.exp(theta2 * (t2-tau2)).astype(int)

    if np.any(S_cancer1 <= 0):
        raise OverflowError("The number of cells has overflowed")  

    rng = np.random.default_rng()

    # generate distribution of fCpG loci when population begins growing 
    # at t=tau, assuming fCpG's are homozygous methylated at t=0
    m_cancer1, k_cancer1, w_cancer1 = initialise_cancer(tau1, mu1, gamma1, nu1, 
                                                    zeta1, NSIM, rng)

    # simulate changes to methylation distribution by splitting the process 
    # into 2 phases, an exponential growth phase and a methylation transition 
    # phase
    for i in range(len(t1)-1):
        m_cancer1, k_cancer1, w_cancer1 = grow_cancer(m_cancer1, k_cancer1,
                                                    w_cancer1, S_cancer[i], 
                                                    S_cancer[i+1], rng)

        m_cancer1, k_cancer1, w_cancer1 = generate_next_timepoint(m_cancer1, 
                                                    k_cancer1, w_cancer1, 
                                                    mu1, gamma1, nu1, zeta1,
                                                    S_cancer[i+1], dt1, rng)

    # generate distribution of fCpG loci when population begins growing 
    # at t=tau, assuming fCpG's are homozygous methylated at t=0
    m_cancer2, k_cancer2, w_cancer2 = initialise_cancer(tau2, mu2, gamma2, nu2,
                                                    zeta2, NSIM, rng)

    # simulate changes to methylation distribution by splitting the process 
    # into 2 phases, an exponential growth phase and a methylation transition 
    # phase
    for i in range(len(t2)-1):
        m_cancer1, k_cancer1, w_cancer1 = grow_cancer(m_cancer1, k_cancer1,
                                            w_cancer1, S_cancer1[i], 
                                            S_cancer1[i+1], rng)


        m_cancer2, k_cancer2, w_cancer2 = grow_cancer(m_cancer2, k_cancer2,
                                            w_cancer2, S_cancer2[i], 
                                            S_cancer2[i+1], rng)

        m_cancer1, k_cancer1, w_cancer1 = generate_next_timepoint(m_cancer1, 
                                                    k_cancer1, w_cancer1, 
                                                    mu1, gamma1, nu1, zeta1,
                                                    S_cancer1[i+1], dt2, rng)

        m_cancer2, k_cancer2, w_cancer2 = generate_next_timepoint(m_cancer2, 
                                                    k_cancer2, w_cancer2, 
                                                    mu2, gamma2, nu2, zeta2,
                                                    S_cancer2[i+1], dt2, rng)

    with np.errstate(divide='raise', over='raise'):
        betaCancer = ((k_cancer1 + k_cancer2 + 2*m_cancer1 + 2*m_cancer2) / 
                        (2*S_cancer1[-1] + 2*S_cancer2[-1]))

    return betaCancer

def simulate_contamination(betaCancer, rho, betaContam):
    """
    Simulate the effect of impurity on the distribution of fCpG methylation
    """
    return rho * betaCancer + (1 - rho) * betaContam

def lognormal_convert_params(mu, sigma):
    """
    Convert mean/std parameterization of a beta distribution to the ones 
    scipy supports
    """

    if np.any(mu <= 0):
        raise Exception("mu must be greater than 0")
    elif np.any(sigma <= 0) :
        raise Exception("sigma must be greater than 0")
    
    log_mean = np.log(mu) - 0.5 * np.log1p(sigma**2 / mu**2)
    log_std = np.sqrt(np.log1p(sigma**2 / mu ** 2))

    return log_mean, log_std

def lognormal_pdf(x, mu, sigma, **kwargs):
    """
    Calculate the pdf of a lognormal distribution with mean/std specified
    """
    log_mean, log_std = lognormal_convert_params(mu, sigma)
    return stats.lognorm.pdf(x, log_std, scale=np.exp(log_mean), **kwargs)

def lognormal_ppf(x, mu, sigma, **kwargs):
    """
    Calculate the ppf of a lognormal distribution with mean/std specified
    """
    log_mean, log_std = lognormal_convert_params(mu, sigma)
    return stats.lognorm.ppf(x, log_std, scale=np.exp(log_mean), **kwargs)

def lognormal_rvs(mu, sigma, **kwargs):
    """
    Generate random samples from a lognormal distribution with mean/std 
    specified
    """
    log_mean, log_std = lognormal_convert_params(mu, sigma)
    return stats.lognorm.rvs(log_std, scale=np.exp(log_mean), **kwargs)

def truncnormal_convert_params(mean, std, lb=0.0, ub=1.0):
    """
    Convert mean/std parameterization of a truncated normal distribution
    to the ones scipy supports

    """

    if np.isfinite(lb):
        a = (lb - mean) / std
    else:
        a = lb
    
    if np.isfinite(ub):
        b = (ub - mean) / std
    else: 
        b = ub

    return a, b


def truncnormal_ppf(y, mean, std, lb=0.0, ub=1.0):
    """
    Calculate the ppf of a truncated normal distribution with mean/std
    specified
    """
    a, b = truncnormal_convert_params(mean, std, lb, ub)

    return stats.truncnorm.ppf(y, a, b, loc=mean, scale=std)

def beta_convert_params(mu, kappa):
    """
    Convert mean/dispersion parameterization of a beta distribution to the ones
    scipy supports
    """

    if np.any(kappa <= 0):
        raise Exception("kappa must be greater than 0")
    elif np.any(mu <= 0) or np.any(mu >= 1):
        raise Exception("mu must be between 0 and 1")
    
    alpha = kappa * mu 
    beta = kappa * (1- mu)

    return alpha, beta

def beta_rvs(mean, kappa, **kwargs):
    """
    Generate random samples from a beta distribution with mean/dispersion
    specified
    """
    alpha, beta = beta_convert_params(mean, kappa)

    return stats.beta.rvs(alpha, beta, **kwargs)

def rescale_beta(beta, delta, eta):
    """
    Linear transform of beta values from between 0 and 1 to between delta and 
    eta
    """
    return (eta - delta) * beta + delta

def add_noise(beta, delta, eta, kappa):
    """
    Rescale distribution to lie between delta and eta and add beta distributed 
    noise
    """
    beta_rescale = rescale_beta(beta, delta, eta)
 
    return beta_rvs(beta_rescale, kappa)

def beta_lpdf(y, alpha, beta):
    """
    Calculate the logpdf of N datapoints originating from Z possible peaks
    with shape parameters alpha and beta respectively
    """
    N = len(y)
    Z = len(alpha)
    lpk = np.empty((Z, N))

    # Check that alpha and beta are the same shape, if not raise exception
    if len(alpha)!= len(beta):
        raise Exception("alph and beta must be the same shape")

    # Loop through each of the possible Z peaks and calculate the
    # log-pdf of the nth beta value in y[N] for the zth beta distribution
    for z in range(Z):
        # Precompute the log-gamma constants
        lgamma_alpha = gammaln(alpha[z])
        lgamma_beta = gammaln(beta[z])
        lgamma_alphaplusbeta = gammaln(alpha[z] + beta[z])

        lpk[z, :] = ((alpha[z] - 1)*np.log(y) + (beta[z] - 1) * np.log1p(-y) - 
            lgamma_alpha - lgamma_beta + lgamma_alphaplusbeta)

    return lpk

def loglikelihood_perpoint_neutral(y, params, constants):
    """
    Estimates the loglikelhood value per datapoint y assuming a neutral model

    Arguments:
        y: fCpG methyaltion data - array of floats
        theta: exponential growth rate of population - float
        rho: tumour purity - float
        tau_rel: relativeage when population began expanding exponentially 
                - float < 1
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu_rel: relative rate to transition from heterozygous to homozygous 
            methylated compared to mu - float >= 0
        zeta_rel: relative rate to transition from heterozygous to homozygous 
            demethylated compared to gamma - float >= 0  
        betaContam: average methylation of non-tumour cells - float
        delta: offset from zero - float
        eta: offset from 1 - float
        kappa: dispersion of beta distributed noise - float > 0
        T: patient's age - float
        Smin: minimum allowed tumour size - float
        Smax: maximum allowed tumour size - float
        NSIM: number of fCpG loci to simulate - int
    Returns:
        Log likelihood per data point at a given set of parameters
    """
    (theta, tau_rel, mu, gamma, nu_rel, zeta_rel, betaContam, delta,
                                        eta, kappa) = params
    rho, T, Smin, Smax, NSIM = constants

    tau = T * tau_rel
    nu = nu_rel * mu
    zeta = zeta_rel * gamma

    # # calucalate p(z| mu, gamma)

    Stot = np.exp(theta * (T-tau))
    if Stot >= np.iinfo(int).max:
        raise OverflowError("The number of cells will overflow")

    betaCancer = stochastic_growth(theta, tau, mu, gamma, 
                                nu, zeta, T, NSIM)
    betaProb = simulate_contamination(betaCancer, rho, betaContam)

    betaScaled = rescale_beta(betaProb, delta, eta)
    hist, bin_edges = np.histogram(betaScaled, np.linspace(0, 1, 101))
    betaCentres = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

    # transform the mean and shape parameters to the ones scipy supports
    alpha, beta = beta_convert_params(betaCentres, kappa)

    # calculate log(p(y|z))
    lpk = beta_lpdf(y, alpha, beta)

    # calculate penalisation that limits the final tumour size to realistic ranges
    logPenalise = logsumexp([-Smin / Stot, 
                                -Smax / Stot], 
                            b=[1, -1])

    # this is a biased estimator of the LL due to having a finite NSIM, so 
    # we must correct for that  
    logBias = (0.5 + 0.005 * kappa ** (4/3)) / NSIM

    # p(y| mu, gamma) = sum(p(y|z)*p(z| mu, gamma))
    # on the log scale this requires logsumexp 
    logl = (logsumexp(lpk, axis = 0, b = hist[:, np.newaxis]) + 
            logPenalise - np.log(NSIM) + logBias)

    return logl

def loglikelihood_perpoint_subclonal(y, params, constants):
    """
    Estimates the loglikelhood value per datapoint y assuming an independent model

    Arguments:
        y: fCpG methyaltion data - array of floats
        theta: exponential growth rate of population - float
        rho: tumour purity - float
        tau_rel: relativeage when population began expanding exponentially 
                - float < 1
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu_rel: relative rate to transition from heterozygous to homozygous 
            methylated compared to mu - float >= 0
        zeta_rel: relative rate to transition from heterozygous to homozygous 
            demethylated compared to gamma - float >= 0  
        betaContam: average methylation of non-tumour cells - float
        delta: offset from zero - float
        eta: offset from 1 - float
        kappa: dispersion of beta distributed noise - float > 0
        T: patient's age - float
        Smin: minimum allowed tumour size - float
        Smax: maximum allowed tumour size - float
        NSIM: number of fCpG loci to simulate - int
    Returns:
        Log likelihood per data point at a given set of parameters
    """

    (theta1, theta2_rel, frac, tau1_rel, mu, gamma, nu_rel, zeta_rel, 
    betaContam, delta, eta, kappa) = params
    rho, T, Smin, Smax, NSIM = constants

    tau1 = T * tau1_rel
    theta2 = theta2_rel * theta1
    tau2 = T - (T - tau1) * theta1 / theta2 - logit(frac) / theta2

    if tau2 <= tau1:
        raise ValueError("tau2 must be after tau1")
    elif tau2 >= T:
        raise ValueError("tau2 must be before T")
    
    Stot = np.exp(theta2 * (T-tau2)) + np.exp(theta1 * (T-tau1))
    if Stot >= np.iinfo(int).max:
        raise OverflowError("The number of cells will overflow")

    nu = nu_rel * mu
    zeta = zeta_rel * gamma

    # # calucalate p(z| mu, gamma)

    betaCancer = stochastic_growth_subclonal(theta1, theta2, tau1, tau2, mu,
                                gamma, nu, zeta, T, NSIM)
    betaProb = simulate_contamination(betaCancer, rho, betaContam)

    betaScaled = rescale_beta(betaProb, delta, eta)
    hist, bin_edges = np.histogram(betaScaled, np.linspace(0, 1, 101))
    betaCentres = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

    # transform the mean and shape parameters to the ones scipy supports
    alpha, beta = beta_convert_params(betaCentres, kappa)

    # calculate log(p(y|z))
    lpk = beta_lpdf(y, alpha, beta)

    # calculate penalisation that limits the final tumour size to realistic ranges
    logPenalise = logsumexp([-Smin / Stot, 
                                -Smax / Stot], 
                            b=[1, -1])

    # calculate the bias due to having a finite NSIM
    logBias = (0.5 + 0.005 * kappa ** (4/3)) / NSIM - np.log(NSIM)

    # p(y| mu, gamma) = sum(p(y|z)*p(z| mu, gamma))
    # on the log scale this requires logsumexp 
    logl = logsumexp(lpk, axis = 0, b = hist[:, np.newaxis]) + logPenalise + logBias

    return logl

def loglikelihood_perpoint_independent(y, params, constants):
    """
    Estimates the loglikelhood value per datapoint y assuming an independent model

    Arguments:
        y: fCpG methyaltion data - array of floats
        theta: exponential growth rate of population - float
        rho: tumour purity - float
        tau_rel: relativeage when population began expanding exponentially 
                - float < 1
        mu1: rate to transition from homozygous demethylated to heterozygous
            in the first cancer - float >= 0
        gamma1: rate to transition from homozygous methylated to heterozygous
            in the first cancer - float >= 0
        nu1_rel: relative rate to transition from heterozygous to homozygous 
            methylated compared to mu in the first cancer - float >= 0
        zeta1_rel: relative rate to transition from heterozygous to homozygous 
            demethylated compared to gamma in the first cancer - float >= 0 
        clock_rel: relative rate of epigenetic switching rates in the second 
            cancer compared to the first - float >= 0
        betaContam: average methylation of non-tumour cells - float
        delta: offset from zero - float
        eta: offset from 1 - float
        kappa: dispersion of beta distributed noise - float > 0
        T: patient's age - float
        Smin: minimum allowed tumour size - float
        Smax: maximum allowed tumour size - float
        NSIM: number of fCpG loci to simulate - int
    Returns:
        Log likelihood per data point at a given set of parameters
    """

    (theta1, theta2_rel, frac, tau1_rel, mu1, gamma1, nu1_rel, zeta1_rel, 
    clock_rel, betaContam, delta, eta, kappa) = params
    rho, T, Smin, Smax, NSIM = constants

    tau1 = T * tau1_rel
    theta2 = theta2_rel * theta1
    tau2 = T - (T - tau1) * theta1 / theta2 - logit(frac) / theta2

    if tau2 <= tau1:
        raise ValueError("tau2 must be after tau1")
    elif tau2 >= T:
        raise ValueError("tau2 must be before T")
    
    Stot = np.exp(theta2 * (T-tau2)) + np.exp(theta1 * (T-tau1))
    if Stot >= np.iinfo(int).max:
        raise OverflowError("The number of cells will overflow")

    nu1 = nu1_rel * mu1
    zeta1 = zeta1_rel * gamma1

    mu2 = clock_rel * mu1
    gamma2 = clock_rel * gamma1
    nu2 = clock_rel * nu1
    zeta2 = clock_rel * zeta1

    # # calucalate p(z| mu, gamma)

    betaCancer = stochastic_growth_independent(theta1, theta2, tau1, tau2, mu1, gamma1, 
                        nu1, zeta1, mu2, gamma2, nu2, zeta2, T, NSIM)
    betaProb = simulate_contamination(betaCancer, rho, betaContam)
    
    betaScaled = rescale_beta(betaProb, delta, eta)
    hist, bin_edges = np.histogram(betaScaled, np.linspace(0, 1, 101))
    betaCentres = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

    # transform the mean and shape parameters to the ones scipy supports
    alpha, beta = beta_convert_params(betaCentres, kappa)

    # calculate log(p(y|z))
    lpk = beta_lpdf(y, alpha, beta)

    # calculate penalisation that limits the final tumour size to realistic ranges
    logPenalise = logsumexp([-Smin / Stot, 
                                -Smax / Stot], 
                            b=[1, -1])

    # calculate the bias due to having a finite NSIM
    logBias = (0.5 + 0.005 * kappa ** (4/3)) / NSIM - np.log(NSIM)

    # p(y| mu, gamma) = sum(p(y|z)*p(z| mu, gamma))
    # on the log scale this requires logsumexp 
    logl = logsumexp(lpk, axis = 0, b = hist[:, np.newaxis]) + logPenalise + logBias

    return logl

def loglikelihood_perpoint(y, params, constants, mode):
    if mode.lower() == 'neutral':
        try:
            logl = loglikelihood_perpoint_neutral(y, params, constants)
        except (ValueError, OverflowError, RuntimeWarning, FloatingPointError) as e:
            logl = np.full(len(y), -np.inf)

    elif mode.lower() == 'subclonal':
        try:
            logl = loglikelihood_perpoint_subclonal(y, params, constants)
        except (ValueError, OverflowError, RuntimeWarning, FloatingPointError) as e:
            logl = np.full(len(y), -np.inf)
    elif mode.lower() == 'independent':
        try:
            logl = loglikelihood_perpoint_independent(y, params, constants)
        except (ValueError, OverflowError, RuntimeWarning, FloatingPointError) as e:
            logl = np.full(len(y), -np.inf)
    else:
        raise ValueError("mode must be one of 'neutral', 'subclonal' or 'independent")

    return logl

def loglikelihood(y, params, constants, mode):
    try:
        LL = np.sum(loglikelihood_perpoint(y, params, constants, mode))
    except (ValueError, OverflowError, RuntimeWarning) as e:
        LL = -np.inf

    return LL

def generate_data_neutral(params, constants):
    """
    Simulates a set of fCpG methylation values, y 

    Arguments:
        params: parameters to calculate LL at - array of floats
        constants: additional paramaters - array
    Returns:
        fCpG methyaltion data
    """
    theta, tau_rel, mu, gamma, nu_rel, zeta_rel, betaContam, delta, eta, kappa = params
    rho, T, Smin, Smax, NSIM = constants

    try:
        tau = T * tau_rel
        nu = nu_rel * mu
        zeta = zeta_rel * gamma

        betaCancer = stochastic_growth(theta, tau, mu, gamma, 
                                    nu, zeta, T, NSIM)
        betaProb = simulate_contamination(betaCancer, rho, betaContam)

        y = add_noise(betaProb, delta, eta, kappa)

    except (ValueError, OverflowError, FloatingPointError) as e:
        y = np.full(NSIM, -1)

    return y

def generate_data_subclonal(params, constants):
    """
    Simulates a set of fCpG methylation values, y 

    Arguments:
        params: parameters of model - array of floats
        constants: additional paramaters - array
    Returns:
        fCpG methyaltion data
    """
    (theta1, theta2_rel, frac, tau1_rel, mu, gamma, nu_rel, zeta_rel, 
    betaContam, delta, eta, kappa) = params
    rho, T, Smin, Smax, NSIM = constants

    try:
        tau1 = T * tau1_rel
        theta2 = theta2_rel * theta1
        tau2 = T - (T - tau1) * theta1 / theta2 - logit(frac) / theta2

        nu = nu_rel * mu
        zeta = zeta_rel * gamma

        betaCancer = stochastic_growth_subclonal(theta1, theta2, tau1, tau2, mu,
                                gamma, nu, zeta, T, NSIM)
        betaProb = simulate_contamination(betaCancer, rho, betaContam)

        y = add_noise(betaProb, delta, eta, kappa)
        
    except (ValueError, OverflowError, FloatingPointError) as e:
        y = np.full(NSIM, -1)

    return y

def generate_data_independent(params, constants):
    """
    Simulates a set of fCpG methylation values, y 

    Arguments:
        params: parameters of model - array of floats
        constants: additional paramaters - array
    Returns:
        fCpG methyaltion data
    """
    (theta1, theta2_rel, frac, tau1_rel, mu1, gamma1, nu1_rel, zeta1_rel, 
    clock_rel, betaContam, delta, eta, kappa) = params
    rho, T, Smin, Smax, NSIM = constants

    try:
        tau1 = T * tau1_rel
        theta2 = theta2_rel * theta1
        tau2 = T - (T - tau1) * theta1 / theta2 - logit(frac) / theta2

        nu1 = nu1_rel * mu1
        zeta1 = zeta1_rel * gamma1

        mu2 = clock_rel * mu1
        gamma2 = clock_rel * gamma1
        nu2 = clock_rel * nu1
        zeta2 = clock_rel * zeta1

        betaCancer = stochastic_growth_independent(theta1, theta2, tau1, tau2, mu1, gamma1, 
                        nu1, zeta1, mu2, gamma2, nu2, zeta2, T, NSIM)
        betaProb = simulate_contamination(betaCancer, rho, betaContam)

        y = add_noise(betaProb, delta, eta, kappa)
        
    except (ValueError, OverflowError, FloatingPointError) as e:
        y = np.full(NSIM, -1)

    return y

def generate_data(params, constants, mode):
    if mode.lower() == 'neutral':
        y_hat = generate_data_neutral(params, constants)

    elif mode.lower() == 'subclonal':
        y_hat = generate_data_subclonal(params, constants)

    elif mode.lower() == 'independent':
        y_hat = generate_data_independent(params, constants)

    else:
        raise ValueError("mode must be one of 'neutral', 'subclonal' or 'independent")

    return y_hat
    
def prior_transform_neutral(flat_prior, scales):
    # priors for parameters [theta, tau_rel, mu, gamma, nu_rel, 
    #                   zeta_rel, betaContam, delta, eta, kappa]
    prior = np.empty(np.shape(flat_prior))
    thetamean, thetastd, muscale, gammascale = scales

    # priors on rho and tau_rel
    prior[0] = lognormal_ppf(flat_prior[0], thetamean, thetastd)
    prior[1] = stats.beta.ppf(flat_prior[1], 2, 2)

    # priors on mu, gamma
    prior[2:4] = truncnormal_ppf(flat_prior[2:4], 
                            np.array([0, 0]), 
                            np.array([muscale, gammascale]), ub=np.inf)

    # priors on nu_rel, zeta_rel
    prior[4:6] = lognormal_ppf(flat_prior[4:6], 
                            np.array([1.0, 1.0]), 
                            np.array([0.7, 0.7]))

    # prior on betaContam
    prior[6] = stats.beta.ppf(flat_prior[6], 2, 2)

    # priors on delta, eta
    prior[7:9] = stats.beta.ppf(flat_prior[7:9], 
                            np.array([5, 95]),
                            np.array([95, 5]))

    # priors on kappa     
    prior[9] = lognormal_ppf(flat_prior[9], 100, 30)
    
    return prior

def prior_transform_subclonal(flat_prior, scales):
    # priors for parameters [theta1, theta2_rel, frac, tau1_rel, mu, 
    # gamma, nu_rel, zeta_rel, betaContam, delta, eta, kappa]
    prior = np.empty(np.shape(flat_prior))
    thetamean, thetastd, muscale, gammascale = scales

    # priors on rho and tau_rel
    prior[0] = lognormal_ppf(flat_prior[0], thetamean, thetastd)
    prior[1] = truncnormal_ppf(flat_prior[1], 1, 1,
                                lb=1.0, ub=np.inf)
    prior[2:4] = stats.beta.ppf(flat_prior[2:4], 2, 2)

    # priors on mu, gamma
    prior[4:6] = truncnormal_ppf(flat_prior[4:6], 
                            np.array([0, 0]), 
                            np.array([muscale, gammascale]), ub=np.inf)

    # priors on nu_rel, zeta_rel
    prior[6:8] = lognormal_ppf(flat_prior[6:8], 
                            np.array([1.0, 1.0]), 
                            np.array([0.7, 0.7]))


    # prior on betaContam
    prior[8] = stats.beta.ppf(flat_prior[8], 2, 2)

    # priors on delta, eta
    prior[9:11] = stats.beta.ppf(flat_prior[9:11], 
                            np.array([5, 95]),
                            np.array([95, 5]))

    # priors on kappa     
    prior[11] = lognormal_ppf(flat_prior[11], 100, 30)
    
    return prior

def prior_transform_independent(flat_prior, scales):
    # priors for parameters [theta1, theta2_rel, frac, tau1_rel, mu1, 
    # gamma1, nu1_rel, zeta1_rel, clock_rel, betaContam, delta, eta, kappa]
    prior = np.empty(np.shape(flat_prior))
    thetamean, thetastd, muscale, gammascale = scales

    # priors on rho and tau_rel
    prior[0] = lognormal_ppf(flat_prior[0], thetamean, thetastd)
    prior[1] = truncnormal_ppf(flat_prior[1], 1, 1,
                                lb=1.0, ub=np.inf)
    prior[2:4] = stats.beta.ppf(flat_prior[2:4], 2, 2)

    # priors on mu, gamma
    prior[4:6] = truncnormal_ppf(flat_prior[4:6], 
                            np.array([0, 0]), 
                            np.array([muscale, gammascale]), ub=np.inf)

    # priors on nu_rel, zeta_rel
    prior[6:8] = lognormal_ppf(flat_prior[6:8], 
                            np.array([1.0, 1.0]), 
                            np.array([0.7, 0.7]))

    # priors on clock_rel
    prior[8] = lognormal_ppf(flat_prior[8], 1.0, 0.7)

    # prior on betaContam
    prior[9] = stats.beta.ppf(flat_prior[9], 2, 2)

    # priors on delta, eta
    prior[10:12] = stats.beta.ppf(flat_prior[10:12], 
                            np.array([5, 95]),
                            np.array([95, 5]))

    # priors on kappa     
    prior[12] = lognormal_ppf(flat_prior[12], 100, 30)
    
    return prior

def prior_transform(flat_prior, scales, mode):

    if mode.lower() == 'neutral':
        prior = prior_transform_neutral(flat_prior, scales)

    elif mode.lower() == 'subclonal':
        prior = prior_transform_subclonal(flat_prior, scales)

    elif mode.lower() == 'independent':
        prior = prior_transform_independent(flat_prior, scales)

    else:
        raise ValueError("mode must be one of 'neutral', 'subclonal' or 'independent")

    return prior

def number_params(mode):
    number_params = {'neutral':10,
                    'subclonal':12,
                    'independent':13}
    
    if mode in number_params.keys():
        ndims = number_params[mode]
    else:
        raise ValueError("mode must be one of 'neutral', 'subclonal' or 'independent")
    
    return ndims

def extract_posterior(res, mode, outsamplesdir, sample, overwrite =False):
    posteriorfile = os.path.join(outsamplesdir, f'{sample}_posterior.csv')
    if os.path.exists(posteriorfile) and not overwrite:
        df = pd.read_csv(posteriorfile)

    else:
        samples =  dynesty.utils.resample_equal(res.samples, 
                                                softmax(res.logwt))

        if mode.lower() == 'neutral':
            labels = ["theta", "tau_rel", "mu", "gamma", "nu_rel", "zeta_rel",
                        "betaContam", "delta", "eta", "kappa"]

        elif mode.lower() == 'subclonal':
            labels = ["theta1", "theta2_rel", "frac", "tau1_rel", "mu", 
                        "gamma", "nu_rel", "zeta_rel", "betaContam", "delta", 
                        "eta", "kappa"]
        elif mode.lower() == 'independent':
            labels = ["theta1", "theta2_rel", "frac", "tau1_rel", "mu", 
                        "gamma", "nu_rel", "zeta_rel", "clock_rel", "betaContam", 
                        "delta", "eta", "kappa"]
        else:
            raise ValueError("mode must be one of 'neutral', 'subclonal' or 'independent")

        df = pd.DataFrame(samples, columns = labels)
        df.to_csv(posteriorfile, index = False)

    return df

def run_inference(
    y, 
    T, 
    outsamples,
    rho=1.0,
    Smin=10**2,
    Smax=10**9,
    nlive=300, 
    thetamean=3.0, 
    thetastd=2.0,
    muscale=0.05, 
    gammascale=0.05, 
    NSIM=None,
    verbose=False,
    dlogz=0.5,
    mode = 'neutral',
    sample_meth = 'auto',
    Ncores = None,
):

    if NSIM is None:
        NSIM = len(y)
        print(f'{NSIM} samples per stochastic run')
    else:
        NSIM = int(NSIM)

    if Ncores is None:
        Ncores = cpu_count()
    else:
        Ncores = int(Ncores)

    # set the std of the halfnormal priors on lam, mu, gamma
    scales = [thetamean, thetastd, muscale, gammascale]

    constants = [rho, T, Smin, Smax, NSIM] 
    # create dummy functions which takes only the params as an argument to pass
    # to dynesty
    ndims = number_params(mode)
    loglikelihood_function = lambda params: loglikelihood(y, params, constants, mode) 
    prior_function = lambda flat_prior: prior_transform(flat_prior, scales, mode)
    
    if sample_meth not in ['unif', 'rwalk', 'rslice', 'auto']:
        raise ValueError("sample_meth must be one of 'unif', 'rwalk', 'rslice' or 'auto'")
    
    t0 = time()

    DUMP_EVERY_N = 50
    LOG_EVERY_N = 20
    print(f'Performing Dynesty sampling with {Ncores} threads')
    with Pool() as pool:
        sampler = NestedSampler(loglikelihood_function, prior_function, ndims,
                                bound='multi', sample=sample_meth, nlive=nlive, 
                                pool=pool, queue_size=Ncores)

        # continue sampling from where we left off
        ncall = sampler.ncall  # internal calls
        nit = sampler.it  # internal iteration
        for it, results in enumerate(sampler.sample(dlogz=dlogz)):
            # split up our results
            (worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar,
            h, nc, worst_it, boundidx, bounditer, eff, delta_logz) = results
            # add number of function calls
            ncall += nc
            nit += 1

            if (nit % LOG_EVERY_N) == 0:
                # print results
                if verbose:
                    print_fn(results, nit, ncall, dlogz=dlogz)

            if (nit % DUMP_EVERY_N) == 0:
                res = sampler.results

                with open(outsamples, 'wb') as f:
                    joblib.dump(res, f)

        # add the remaining live points back into our final results 
        # (they are removed from our set of dead points each time we start sampling)
        for it2, results in enumerate(sampler.add_live_points()):
            # split up results
            (worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar,
            h, nc, worst_it, boundidx, bounditer, eff, delta_logz) = results
            # print results
            print_fn(results, nit, ncall, add_live_it=it2+1, dlogz=dlogz)

    with open(outsamples, 'wb') as f:
        joblib.dump(res, f)

    res = sampler.results

    t1 = time()
    timesampler = int(t1-t0)

    print("\nTime taken to run Dynesty is {} seconds".format(timesampler))

    print(res.summary())

    return res

