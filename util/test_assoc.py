import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from patsy import dmatrix

def get_feature(df, feature, meta_cols=['t', 'sample_id', 'gender']):
    r"""Get binned expression of specific feature along w/ metadata"""
    assert feature in df.columns
    assert np.all([col in df.columns for col in meta_cols])

    feature_df = df[meta_cols].copy()
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


def test_trajectory(df, dof=5, degree=3):
    r"""
    Tests if gene expression is significantly changing along pseudotime.
    
     - Full model:r"$ expr_i(t) \sim \beta_0 + f(\gamma_i(t)) + \beta_1 \cdot gender + b_i + \epsilon_i" 
     - Reduced model: r"$ expr_i(t) \sim \beta_0 + \beta_1 \cdot gender + b_i + \epsilon_i" 
    
    Null Hypothesis (H0): Expression is stationary (constant) over pseudotime.
    Alternative Hypothesis (H1): Expression changes over pseudotime.
    """
    assert 't' in df, "Please infer trajectory first"
    assert 'gender' in df and 'sample_id' in df, \
        'Please append gender & sample metadata'
    
    df, spline_cols = get_bspline(df, dof, degree)
    spline_terms = " + ".join(spline_cols)

    # Full model
    full_model = smf.mixedlm(
        f"expression ~ {spline_terms} + gender", 
        df, groups=df['sample_id']
    ).fit()

    # Reduced model
    reduced_model = smf.mixedlm(
        "expression ~ gender", 
        df, groups=df['sample_id']
    ).fit()

    # Omnibus test
    pval = likelihood_ratio_test(full_model, reduced_model, dof=len(spline_cols))
    return pval


def test_gender(df, dof=5, degree=3):
    r"""
    Tests if gene expression is significantly altered by gender.

    - Model: r"$ expr_i(t) \sim \beta_0 + f(\gamma_i(t)) + \beta_1 \cdot gender + b_i + \epsilon_i" 

    Null Hypothesis (H0): Expression is not different between males and females.
    Alternative Hypothesis (H1): Expression is different between genders.
    """
    assert 't' in df, "Please infer trajectory first"
    assert 'gender' in df and 'sample_id' in df, \
        'Please append gender & sample metadata'
    
    df, spline_cols = get_bspline(df, dof, degree)
    spline_terms = " + ".join(spline_cols)

    model = smf.mixedlm(
        f"expression ~ {spline_terms} + gender",
        df, groups=df['sample_id']
    ).fit()

    # extract gender-specific coefficient
    pval = model.pvalues['gender[T.M]']
    coeff = model.params['gender[T.M]']
    return pval, coeff 


def test_gender_trajectory_interaction(df, dof=5, degree=3):
    r"""
    Tests if expression patterne over trajectory significantly differs
    by gender (i.e. trajectory dependes on gender)

    Null Hypothesis (H0): Expression follows the same pseudotime trajectory for both genders.
    Alternative Hypothesis (H1): Expression patterns differ along pseudotime for each gender.
    """
    assert 't' in df, "Please infer trajectory first"
    assert 'gender' in df and 'sample_id' in df, \
        'Please append gender & sample metadata'
    
    df, spline_cols = get_bspline(df, dof, degree)

    # Full model
    interaction_terms = " + ".join([f"{col} * gender" for col in spline_cols])
    full_model = smf.mixedlm(
        f"expression ~ {interaction_terms}",
        df, groups=df['sample_id']
    ).fit()

    # Reduced model
    spline_terms = " + ".join(spline_cols)
    reduced_model = smf.mixedlm(
        f"expression ~ {spline_terms} + gender",
        df, groups=df['sample_id']
    ).fit()

    # Omnibus test
    pval = likelihood_ratio_test(full_model, reduced_model, dof=len(spline_cols))
    return pval


def get_test_associations(df, dof=5, degree=3):
    r"""Get test statistics for both trajectory & gender across features"""
    features = df.columns[:-3]  # skip 't', 'sample_id' & 'gender'
    pbar = tqdm(range(len(features)))
    res = np.zeros((len(features), 4))

    for i in pbar:
        feature_df = get_feature(df, feature=df.columns[i])
        pval_t = test_trajectory(feature_df, dof, degree)
        pval_gender, coeff = test_gender(feature_df, dof, degree)
        pval_interact = test_gender_trajectory_interaction(feature_df, dof, degree)

        pvals_adj = multipletests(
            [pval_t, pval_gender, pval_interact], 
            method='fdr_bh'
        )[1]
        res[i, :-1] = pvals_adj
        res[i, -1] = coeff

        pbar.set_description('Feature {0}/{1}'.format(i+1, len(features)))

    cols = ['pval.t', 'pval.gender', 'pval.interact', 'gender.coeff']
    res_df = pd.DataFrame(res, index=features, columns=cols)  
    return res_df