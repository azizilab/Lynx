import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2, ttest_rel
from patsy import dmatrix

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import get_cluster_dynamics


# -----------------------------------------------
#  Statistical tests for cell-cell interaction
# -----------------------------------------------

def test_cci_target(df_int, df_abun, cluster_labels, alpha=0.05, alternative='greater'):
    """
    Perform paired t-tests between interaction and abundance scores w.r.t target cell type
    for each sender cluster, with optional one-sided test and FDR correction.
    """
    df_int_norm = df_int.copy()
    df_abun_norm = df_abun.copy()

    stats = []
    for sender in cluster_labels:
        t_stat, p_val = ttest_rel(
            df_int_norm[sender], df_abun_norm[sender], alternative=alternative
        )
        stats.append((sender, t_stat, p_val))
    results = pd.DataFrame(stats, columns=["cluster", "t_stat", "p_value"])

    # FDR correction
    try:
        reject, qvals, _, _ = multipletests(results["p_value"], alpha=alpha, method="fdr_bh")
        results["q_value"] = qvals
        results["significant"] = reject
    except ZeroDivisionError:
        print("Warning: Invalid p-values for FDR correction.")
        results["q_value"] = np.nan
        results["significant"] = False

    return results.sort_values("q_value")


def test_cci(adata, cci_df, cluster_labels, cluster_key='cell_type'):
    r"""Post-hoc paired t-test for significant CCI against cell-type abundance"""
    n_clusters = len(cluster_labels)
    cci_summary = pd.DataFrame(
        adata.obsm['omega'].copy(),
        index=adata.obs_names,
        columns=cluster_labels
    )
    # cci_summary = cci_summary / cci_summary.values.sum(axis=1, keepdims=True)  # Normalize to proportions

    abundance_summary = pd.DataFrame(
        adata.obsm['abundance'].copy(),
        index=adata.obs_names,
        columns=cluster_labels,  
    )
    
    mask = np.zeros((n_clusters, n_clusters))
    qval = np.zeros_like(cci_df.values)
    for i, target in enumerate(cluster_labels):
        if np.any(target in adata.obs[cluster_key].values):
            cci_dynamics = get_cluster_dynamics(
                adata, cci_summary, cluster_key=cluster_key,
                target_cell_type=target, n_bins=50, show_fig=False,
            )
            abun_dynamics = get_cluster_dynamics(
                adata, abundance_summary, cluster_key=cluster_key,
                target_cell_type=target, n_bins=50, show_fig=False,
            )
            test_res = test_cci_target(
                cci_dynamics, abun_dynamics, 
                cluster_labels=cci_dynamics.columns[1:], 
                alternative='greater'
            )

            for _, row in test_res.iterrows():
                j = pd.Index(cluster_labels).get_loc(row['cluster'])
                if row['significant'] and not np.isnan(row['q_value']):
                    mask[j, i] = 1  # (row: sender, col: receiver)
                    qval[j, i] = -np.log10(row["q_value"]) 

    qval_df = pd.DataFrame(
        qval,
        index=cci_df.index,
        columns=cci_df.columns 
    )

    return cci_df*mask, qval_df


# ----------------------------------------------------
#  statistic tests for trajectory / sex association
# ----------------------------------------------------

def get_feature(df, feature, meta_cols=['sample_id', 'sex', 't', 'zone']):
    r"""Get binned expression of specific feature along w/ metadata"""
    assert feature in df.columns
    feature_df = df[[col for col in meta_cols if col in df.columns]].copy()
    feature_df['expression'] = df[feature].copy()
    return feature_df


def get_bspline(df, dof=5, degree=3):
    assert 't' in df, "Please infer trajectory first"
    
    bs_basis = dmatrix(
        f"bs(t, df={dof}, degree={degree}, include_intercept=False)",
        df, return_type='dataframe'
    )
    spline_cols = ['spline_'+str(i+1) for i in range(bs_basis.shape[1]-1)]
    df[spline_cols] = bs_basis.iloc[:, 1:]  # Drop the intercept term
    return df, spline_cols


def likelihood_ratio_test(full_model, reduced_model, dof):
    LRT_stat = -2 * (reduced_model.llf - full_model.llf)
    pval = chi2.sf(LRT_stat, df=dof)
    return pval


def test_trajectory(df, dof=5, degree=3, likelihood='gaussian'):
    r"""
    Tests if gene expression significantly affected by 
        (1). trajectory (t); (2). sex
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with expression, sample_id, sex, and t columns
    dof : int
        Degrees of freedom for B-spline basis
    degree : int
        Degree of B-spline
    likelihood : str
        'gaussian' for mixed linear models (default)
        'gamma' for Gamma GLM
    """
    assert 't' in df, "Please infer trajectory first"
    assert 'sex' in df and 'sample_id' in df, \
        'Please append sex & sample metadata'
    
    if likelihood not in ['gaussian', 'gamma']:
        raise NotImplementedError(f"Likelihood '{likelihood}' not implemented. Use 'gaussian' or 'gamma'.")
    
    df, spline_terms = get_bspline(df, dof, degree)
    formula1 = " + ".join(spline_terms)
    formula2 = formula1 + " + sex"
    formula3 = f"({formula1}) * sex"
    
    # Fitting multiple models, select w/ the lowest BIC
    if likelihood == 'gaussian':
        null_model = sm.OLS(df['expression'].values, np.ones(len(df))).fit()
        trajectory_model = smf.mixedlm("expression ~ "+formula1, df, groups=df['sample_id']).fit(reml=False)
        sex_model = smf.mixedlm("expression ~ "+formula2, df, groups=df['sample_id']).fit(reml=False)
        interact_model = smf.mixedlm("expression ~ "+formula3, df, groups=df['sample_id']).fit(reml=False)
    
    elif likelihood == 'gamma':
        gamma_family = sm.families.Gamma(link=sm.families.links.Log())
        cov_struct = sm.cov_struct.Exchangeable()
        null_model = smf.gee("expression ~ 1", groups=df['sample_id'], data=df, family=gamma_family, cov_struct=cov_struct).fit()
        trajectory_model = smf.gee("expression ~ "+formula1, groups=df['sample_id'], data=df, family=gamma_family, cov_struct=cov_struct).fit()
        sex_model = smf.gee("expression ~ "+formula2, groups=df['sample_id'], data=df, family=gamma_family, cov_struct=cov_struct).fit()
        interact_model = smf.gee("expression ~ "+formula3, groups=df['sample_id'], data=df, family=gamma_family, cov_struct=cov_struct).fit()


    # Model selection (BIC)
    is_trajectory_feature = 0   
    is_interact_feature = 0
    sex_coeff = 0.
    sex_pval = 1.

    BICs = [null_model.bic, trajectory_model.bic, sex_model.bic, interact_model.bic]

    if np.argmin(BICs) == 0:
        best_model = null_model   # stationary dynamics
    else:
        is_trajectory_feature = 1
        if np.argmin(BICs) == 1:
            best_model = trajectory_model
        elif np.argmin(BICs) == 2:
            best_model = sex_model
        else:
            best_model = interact_model
            is_interact_feature = 1
    pred_expr = np.maximum(best_model.predict(), 0)  # Clip to non-negative

    if 'sex[T.M]' in best_model.pvalues:
        sex_pval = sex_model.pvalues['sex[T.M]']
        sex_coeff = sex_model.params['sex[T.M]']

    return pred_expr, (is_trajectory_feature, is_interact_feature, sex_coeff, sex_pval)


def get_test_associations(df, dof=5, degree=3, alpha=.05, likelihood='gaussian'):
    r"""Get test statistics for both trajectory & sex across features
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features, sample_id, sex, and t columns
    dof : int
        Degrees of freedom for B-spline basis
    degree : int  
        Degree of B-spline
    alpha : float
        Significance level for FDR correction
    likelihood : str
        'gaussian' for mixed linear models (default)
        'gamma' for Gamma GLM
    """
    features = df.columns[df.dtypes.apply(lambda x: x.kind in 'if')]  # skip covariate columns
    features = features[:-1]
    fitted_df = df[features].copy()
    fitted_df['sample_id'] = df['sample_id'].copy()
    fitted_df['sex'] = df['sex'].copy()

    pbar = tqdm(range(len(features)))
    cols = ['trajectory_feature', 'interact_feature', 'coeff.sex', 'pval.sex', 'adj-pval.sex']
    test_assoc = np.zeros((len(features), 5))
    test_assoc[:, -1] = np.nan

    for i in pbar:
        feature = df.columns[i]
        feature_df = get_feature(df, feature)
        res = test_trajectory(feature_df, dof, degree, likelihood=likelihood)
        fitted_df[feature] = res[0]
        test_assoc[i, :-1] = res[1]
        pbar.set_description('Feature {0}/{1}'.format(i+1, len(features)))

    # FDR control
    test_assoc_df = pd.DataFrame(test_assoc, index=features, columns=cols)
    pvals = test_assoc_df['pval.sex'].values
    indices = np.where(~np.isnan(pvals))[0]
    test_assoc_df['adj-pval.sex'][indices] = multipletests(pvals[indices], method='fdr_bh')[1]
    
    test_assoc_df['interact_feature'][test_assoc_df['adj-pval.sex'] >= alpha] = 0
    test_assoc_df['interact_feature'] = test_assoc_df['interact_feature'].astype(np.uint8)
    test_assoc_df['trajectory_feature'] = test_assoc_df['trajectory_feature'].astype(np.uint8)
    
    return fitted_df, test_assoc_df

