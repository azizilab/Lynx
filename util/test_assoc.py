import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from patsy import dmatrix


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


def test_trajectory(df, dof=5, degree=3):
    r"""
    Tests if gene expression significantly affected by 
        (1). trajectory (t); (2). sex
    """
    assert 't' in df, "Please infer trajectory first"
    assert 'sex' in df and 'sample_id' in df, \
        'Please append sex & sample metadata'
    
    df, spline_terms = get_bspline(df, dof, degree)

    # Fitting multiple models
    sex_model = smf.mixedlm("expression ~ sex", df, groups=df['sample_id']).fit(reml=False)

    formula1 = " + ".join(spline_terms)
    trajectory_model = smf.mixedlm("expression ~ "+formula1, df, groups=df['sample_id']).fit(reml=False)

    formula2 = formula1+" + sex"
    additive_model = smf.mixedlm("expression ~ "+formula2, df, groups=df['sample_id']).fit(reml=False)

    formula3 = formula2+" + "+":sex + ".join(spline_terms) + ":sex"
    interact_model = smf.mixedlm("expression ~ "+formula3, df, groups=df['sample_id']).fit(reml=False)

    # Model selection (BIC)
    is_trajectory_feature = 0   
    is_interact_feature = 0
    sex_coeff = 0.
    sex_pval = np.nan

    BICs = [sex_model.bic, trajectory_model.bic, sex_model.bic, interact_model.bic]
    if np.argmin(BICs) == 0:
        best_model = sex_model
    else:
        is_trajectory_feature = 1
        if np.argmin(BICs) == 1:
            best_model = trajectory_model
        elif np.argmin(BICs) == 2:
            best_model = additive_model
        else:
            best_model = interact_model
            is_interact_feature = 1

    pred_expr = best_model.predict()
            
    if 'sex[T.M]' in best_model.pvalues:
        sex_pval = sex_model.pvalues['sex[T.M]']
        sex_coeff = sex_model.params['sex[T.M]']

    return pred_expr, (is_trajectory_feature, is_interact_feature, sex_coeff, sex_pval)


def get_test_associations(df, dof=5, degree=3, alpha=.05):
    r"""Get test statistics for both trajectory & sex across features"""
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
        res = test_trajectory(feature_df, dof, degree)
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
