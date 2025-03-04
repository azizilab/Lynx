import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from patsy import dmatrix

def get_feature(df, feature, meta_cols=['t', 'sample_id', 'sex']):
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
    
     - Full model:r"$ expr_i(t) \sim \beta_0 + f(\gamma_i(t)) + \beta_1 \cdot sex + b_i + \epsilon_i" 
     - Reduced model: r"$ expr_i(t) \sim \beta_0 + \beta_1 \cdot sex + b_i + \epsilon_i" 
    
    Null Hypothesis (H0): Expression is stationary (constant) over pseudotime.
    Alternative Hypothesis (H1): Expression changes over pseudotime.
    """
    assert 't' in df, "Please infer trajectory first"
    assert 'sex' in df and 'sample_id' in df, \
        'Please append sex & sample metadata'
    
    df, spline_cols = get_bspline(df, dof, degree)
    spline_terms = " + ".join(spline_cols)

    # Full model
    full_model = smf.mixedlm(
        f"expression ~ {spline_terms} + sex", 
        df, groups=df['sample_id']
    ).fit()

    # Reduced model
    reduced_model = smf.mixedlm(
        "expression ~ sex", 
        df, groups=df['sample_id']
    ).fit()

    # Omnibus test
    pval = likelihood_ratio_test(full_model, reduced_model, dof=len(spline_cols))
    return pval


def test_sex(df, dof=5, degree=3):
    r"""
    Tests if gene expression is significantly altered by sex.

    - Model: r"$ expr_i(t) \sim \beta_0 + f(\gamma_i(t)) + \beta_1 \cdot sex + b_i + \epsilon_i" 

    Null Hypothesis (H0): Expression is not different between males and females.
    Alternative Hypothesis (H1): Expression is different between sexs.
    """
    assert 't' in df, "Please infer trajectory first"
    assert 'sex' in df and 'sample_id' in df, \
        'Please append sex & sample metadata'
    
    df, spline_cols = get_bspline(df, dof, degree)
    spline_terms = " + ".join(spline_cols)

    model = smf.mixedlm(
        f"expression ~ {spline_terms} + sex",
        df, groups=df['sample_id']
    ).fit()

    # extract sex-specific coefficient
    pval = model.pvalues['sex[T.M]']
    coeff = model.params['sex[T.M]']
    return pval, coeff 


def test_sex_trajectory_interaction(df, dof=5, degree=3):
    r"""
    Tests if expression pattern over trajectory significantly differs
    by sex (i.e. trajectory dependes on sex)

    Null Hypothesis (H0): Expression follows the same pseudotime trajectory for both sexs.
    Alternative Hypothesis (H1): Expression patterns differ along pseudotime for each sex.
    """
    assert 't' in df, "Please infer trajectory first"
    assert 'sex' in df and 'sample_id' in df, \
        'Please append sex & sample metadata'
    
    df, spline_cols = get_bspline(df, dof, degree)

    # Full model
    interaction_terms = " + ".join([f"{col} * sex" for col in spline_cols])
    full_model = smf.mixedlm(
        f"expression ~ {interaction_terms}",
        df, groups=df['sample_id']
    ).fit()

    # Reduced model
    spline_terms = " + ".join(spline_cols)
    reduced_model = smf.mixedlm(
        f"expression ~ {spline_terms} + sex",
        df, groups=df['sample_id']
    ).fit()

    # Omnibus test
    pval = likelihood_ratio_test(full_model, reduced_model, dof=len(spline_cols))
    return pval


def get_test_associations(df, dof=5, degree=3):
    r"""Get test statistics for both trajectory & sex across features"""
    features = df.columns[:-3]  # skip 't', 'sample_id' & 'sex'
    pbar = tqdm(range(len(features)))
    res = np.zeros((len(features), 4))

    for i in pbar:
        feature_df = get_feature(df, feature=df.columns[i])
        pval_t = test_trajectory(feature_df, dof, degree)
        pval_sex, coeff = test_sex(feature_df, dof, degree)
        pval_interact = test_sex_trajectory_interaction(feature_df, dof, degree)

        pvals_adj = multipletests(
            [pval_t, pval_sex, pval_interact], 
            method='fdr_bh'
        )[1]
        res[i, :-1] = pvals_adj
        res[i, -1] = coeff

        pbar.set_description('Feature {0}/{1}'.format(i+1, len(features)))

    cols = ['pval.t', 'pval.sex', 'pval.interact', 'sex.coeff']
    res_df = pd.DataFrame(res, index=features, columns=cols)  
    return res_df