"""
EVOFLUx is © 2025, Calum Gabbutt

EVOFLUx is published and distributed under the Academic Software License v1.0 (ASL).

EVOFLUx is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.

You should have received a copy of the ASL along with this program; if not, email Calum Gabbutt at calum.gabbutt@icr.ac.uk. It is also published at https://github.com/gabor1/ASL/blob/main/ASL.md.

You may contact the original licensor at calum.gabbutt@icr.ac.uk.
"""

import numpy as np
from scipy.special import logit
from scipy.sparse import csc_matrix, diags
import time
import pandas as pd
import os
import zipfile
from zipfile import ZipFile

EVOFLUX_DIR = os.path.dirname(os.path.realpath(__file__))

def read_largest_tsv_from_zip(zip_filepath):
    """Reads the largest .tsv file from a ZIP archive using ZipFile.

    Args:
        zip_filepath: Path to the ZIP file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if no .tsv file is found.
    """
    try:
        with ZipFile(zip_filepath, 'r') as zip_ref:
            largest_file = None
            largest_size = 0

            for file_info in zip_ref.infolist():  # Iterate through files in the zip
                if file_info.filename.endswith('.tsv') and file_info.file_size > largest_size:
                    largest_file = file_info
                    largest_size = file_info.file_size

            if largest_file:
                with zip_ref.open(largest_file) as f:  # Open the largest file
                    df = pd.read_csv(f, sep='\t', index_col = 0) # Specify the separator if needed.
                    return df
            else:
                print("No .tsv file found in the zip archive.")
                return None

    except FileNotFoundError:
        print(f"Error: ZIP file not found at {zip_filepath}")
        return None
    except zipfile.BadZipFile:
        print(f"Error: Invalid ZIP file at {zip_filepath}")
        return None
    except Exception as e: # Catch any other potential errors
        print(f"An unexpected error occurred: {e}")
        return None


def process_data(beta_all, sampleinfo, array=None):
    """
    Preprocess DNA methylation beta values by filtering cross-reactive probes
    and optionally restricting to probes on the 450K or EPIC arrays,
    excluding sex chromosomes.

    Parameters
    ----------
    beta_all : pd.DataFrame
        DataFrame of beta values (rows: CpG sites, columns: samples).
    
    sampleinfo : pd.DataFrame
        DataFrame of sample metadata. If it contains a 'SAMPLE_TRAINING_MODEL' 
        boolean column, only training samples will be used.
    
    array : str or None, optional
        If specified, restrict to probes from a specific array:
        - '450K' : use only sites from the Illumina 450K array
        - 'EPIC' : use only sites from the Illumina EPIC array
        - None   : use all available sites (after cross-reactive removal)

    Returns
    -------
    beta_candidate : pd.DataFrame
        Filtered beta matrix restricted to training samples and selected probes.
    
    sampleinfo_training : pd.DataFrame
        Subset of sampleinfo for training samples (or all if no training flag).
    """

    # Remove rows with NA entries
    beta_all = beta_all.dropna()

    # Remove known cross-reactive probes
    crossreactive = pd.read_csv(
        os.path.join(EVOFLUX_DIR, 'data', '13059_2016_1066_MOESM1_ESM.csv'),
        index_col=0
    )
    beta_all = beta_all.loc[beta_all.index.difference(crossreactive.index), :]

    # Filter probes based on array type and exclude sex chromosomes
    if array is None:
        good_sites = beta_all.index

    elif array == '450K':
        manifest = pd.read_csv(
            os.path.join(EVOFLUX_DIR, 'data', 'HumanMethylation450_15017482_v1-2.csv'),
            index_col=0, low_memory=False
        )
        manifest = manifest.loc[manifest.index.intersection(beta_all.index), :]
        good_sites = manifest.loc[~manifest['CHR'].str.contains('X|Y'), :].index

    elif array == 'EPIC':
        manifest = pd.read_csv(
            os.path.join(EVOFLUX_DIR, 'data', 'MethylationEPIC_v-1-0_B4'),
            index_col=0, low_memory=False
        )
        manifest = manifest.loc[manifest.index.intersection(beta_all.index), :]
        good_sites = manifest.loc[~manifest['CHR'].str.contains('X|Y'), :].index

    else:
        raise ValueError("Expected array to be: None, '450K', or 'EPIC'")

    # Subset beta values to good CpG sites
    beta_all = beta_all.loc[good_sites]

    # Subset to training samples if indicated in metadata
    if 'SAMPLE_TRAINING_MODEL' in sampleinfo.columns:
        sampleinfo_training = sampleinfo.loc[sampleinfo['SAMPLE_TRAINING_MODEL']]
        beta_candidate = beta_all.loc[:, sampleinfo_training.index]
    else:
        sampleinfo_training = sampleinfo
        beta_candidate = beta_all.loc[:, sampleinfo_training.index]

    return beta_candidate, sampleinfo_training


def calculate_stats(beta_candidate, sampleinfo_training):
    """
    Calculate summary statistics to identify fluctuating CpGs across or within cell types.

    If multiple CELL_TYPEs are present in the dataset, the function computes the standard
    deviation and distance from 0.5 separately for each group and returns the average
    across groups. If only one CELL_TYPE is present, or if the 'CELL_TYPE' column is
    missing entirely, the statistics are computed across all samples.

    Parameters
    ----------
    beta_candidate : pd.DataFrame
        DataFrame of beta values for candidate CpG sites (rows: CpGs, columns: samples).
    
    sampleinfo_training : pd.DataFrame
        DataFrame containing sample metadata. Should include a 'CELL_TYPE' column
        and row indices matching the columns of `beta_candidate`. If 'CELL_TYPE'
        is absent, statistics will be computed over all samples.

    Returns
    -------
    avg_stds : pd.Series
        Mean standard deviation of beta values for each CpG, either across CELL_TYPEs or overall.
    
    avg_distance_from_05 : pd.Series
        Mean absolute deviation of beta values from 0.5 for each CpG, either across CELL_TYPEs or overall.
    """
    stds = pd.DataFrame({}, index=beta_candidate.index)
    distance_from_05 = pd.DataFrame({}, index=beta_candidate.index)

    if 'CELL_TYPE' in sampleinfo_training.columns:
        sample_groups = sampleinfo_training['CELL_TYPE'].unique()
    else:
        sample_groups = []

    if len(sample_groups) > 1:
        for sample_group in sample_groups:
            samples = sampleinfo_training.loc[
                sampleinfo_training['CELL_TYPE'] == sample_group, :]
            beta_samples = beta_candidate[samples.index]

            patient_std = np.std(beta_samples, axis=1)
            stds[sample_group] = patient_std

            distance_from_05[sample_group] = np.abs(np.mean(beta_samples, axis=1) - 0.5)

        avg_stds = stds.mean(axis=1)
        avg_distance_from_05 = distance_from_05.mean(axis=1)

    else:
        # Either no CELL_TYPE column or only one group
        beta_samples = beta_candidate[sampleinfo_training.index]
        avg_stds = pd.Series(np.std(beta_samples, axis=1), index=beta_candidate.index)
        avg_distance_from_05 = pd.Series(np.abs(np.mean(beta_samples, axis=1) - 0.5), index=beta_candidate.index)

    return avg_stds, avg_distance_from_05

def construct_cosine_similarity_graph(X, k=5):
    """
    Construct a symmetric k-nearest neighbor graph using cosine similarity.

    Each sample is connected to its k nearest neighbors (including itself) based on 
    cosine similarity. The resulting graph is binary (edges are set to 1) and symmetrized 
    by taking the element-wise maximum with its transpose.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix where each row is a sample and each column is a feature.
        The function normalizes each row to unit length before computing similarity.

    k : int, default=5
        Number of nearest neighbors to use for constructing the graph (including self).

    Returns
    -------
    W : scipy.sparse.csc_matrix, shape (n_samples, n_samples)
        Sparse, symmetric affinity matrix where W[i, j] = 1 if sample j is among the 
        k nearest neighbors of sample i or vice versa.
    """

    n_samples = X.shape[0]

    # Normalize each row to unit norm
    X_normalized = np.linalg.norm(X, axis=1)
    X = X / np.maximum(X_normalized[:, np.newaxis], 1e-12)

    # Compute cosine similarity matrix
    D_cosine = np.dot(X, X.T)

    # Sort similarities in descending order and get top (k+1) indices
    idx = np.argsort(-D_cosine, axis=1)
    idx_k = idx[:, :k+1]

    # Construct triplet edge list 
    row_idx = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
    col_idx = np.ravel(idx_k, order='F')  
    data = np.ones_like(row_idx, dtype=np.float64)

    # Build sparse affinity matrix
    W = csc_matrix((data, (row_idx, col_idx)), shape=(n_samples, n_samples))

    # Symmetrize by taking the element-wise maximum
    W = W.maximum(W.T)

    return W

def laplacian_score(X, W):
    """
    Compute the Laplacian Score for each feature in X given a sparse affinity matrix W.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix, where each row is a sample and each column is a feature.

    W : scipy.sparse.spmatrix, shape (n_samples, n_samples)
        Sparse symmetric affinity matrix (e.g., from k-NN cosine similarity).

    Returns
    -------
    scores : np.ndarray, shape (n_features,)
        Laplacian scores for each feature. Lower scores indicate more informative features.
    """
    # Degree matrix D: diagonal with row sums of W
    D = diags(W.sum(axis=1).A1)

    # Graph Laplacian L = D - W
    L = D - W

    # Center features w.r.t. the degree matrix (optional, but improves numerical
    # stability)
    XT = X.T  # features x samples
    D1 = D.diagonal()[:, np.newaxis]
    D_sum = D1.sum()
    f_mean = (X.T @ D1).flatten() / D_sum  # degree-weighted mean
    f_centered = XT - f_mean[:, np.newaxis]

    # Numerator: f^T L f (feature × Laplacian × feature)
    numerator = np.sum(f_centered * (L @ f_centered.T).T, axis=1)

    # Denominator: f^T D f (feature × degree × feature)
    denominator = np.sum(f_centered * (D @ f_centered.T).T, axis=1)

    # Avoid divide-by-zero
    denominator = np.maximum(denominator, 1e-12)

    scores = numerator / denominator
    return scores

def calculate_laplacian_score(beta_candidate):
    """
    Calculate the Laplacian Score for each CpG site based on its M-value representation.

    The Laplacian Score is a feature selection metric that identifies features
    (here, CpG sites) that best preserve local manifold structure of the data.
    This is useful for filtering out CpGs which exhibit cell-type specific methyaltion 

    Parameters
    ----------
    beta_candidate : pd.DataFrame
        A DataFrame of beta values (rows: CpG sites, columns: samples).
        Values must be in the open interval (0, 1), as the logit transform is applied.

    Returns
    -------
    pd.Series
        A Series of Laplacian Scores for each CpG, indexed by CpG ID.
        Lower scores indicate more informative features.
    """

    # Convert beta values to M-values using the logit transform
    Mvalues = logit(beta_candidate).transpose()  # shape: (samples, CpGs)

    # Construct the affinity (similarity) matrix W between samples
    time1 = time.time()
    W = construct_cosine_similarity_graph(Mvalues.values)
    time2 = time.time()
    print(f'Constructing the affinity matrix took {time2 - time1:.3f} secs')

    # Compute the Laplacian score for each CpG (feature)
    time1 = time.time()
    lap_feature = laplacian_score(Mvalues.values, W=W)
    time2 = time.time()
    print(f'Calculating the Laplacian feature score took {time2 - time1:.3f} secs')

    # Return scores indexed by CpG site
    return pd.Series(lap_feature, index=beta_candidate.index)

def filter_for_fcpgs(beta_candidate, lap_feature, avg_stds, 
                     avg_distance_from_05, lap_feature_thresh=None,
                     avg_stds_thresh=0.15, avg_distance_from_05_thresh=0.1):
    """
    Filter CpG sites to identify fluctuating CpGs (fCpGs) based on
    variability, Laplacian feature score, and distance from 0.5.

    This function applies thresholds to:
    - Standard deviation (to ensure sufficient fluctuation),
    - Laplacian score (to remove cell-specific CpGs),
    - Mean distance from 0.5 (to remove constitutively hypo/hypermethylated sites).

    Parameters
    ----------
    beta_candidate : pd.DataFrame
        DataFrame of beta values (rows: CpG sites, columns: samples) for 
            candidate sites.
    
    lap_feature : pd.Series
        Laplacian feature score for each CpG. Lower values indicate more 
            informative features.
    
    avg_stds : pd.Series
        Average standard deviation across CELL_TYPEs or samples for each CpG.
    
    avg_distance_from_05 : pd.Series
        Average absolute deviation from 0.5 for each CpG.

    lap_feature_thresh : float, optional
        Threshold for Laplacian score. If None, uses the 95th percentile (i.e., 
            keeps bottom 5%).
    
    avg_stds_thresh : float, default=0.15
        Minimum required standard deviation to retain a CpG.

    avg_distance_from_05_thresh : float, default=0.1
        Maximum allowed distance from 0.5 (i.e., filter out constitutive sites).

    Returns
    -------
    beta_fcpgs : pd.DataFrame
        Filtered beta matrix containing only fCpGs that passed all thresholds.
    """

    # If no Laplacian threshold provided, default to top 5% most informative
    if lap_feature_thresh is None:
        lap_feature_thresh = np.percentile(lap_feature, 95)

    # Apply all filters
    beta_fcpgs = beta_candidate.loc[
                        (avg_stds > avg_stds_thresh) &
                        (lap_feature > lap_feature_thresh) &
                        (avg_distance_from_05 < avg_distance_from_05_thresh)
                    ]

    print(beta_fcpgs.shape)

    return beta_fcpgs



