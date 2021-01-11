import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from numpy import interp
import rbo
from collections import OrderedDict
import pima_utils as pm


def get_logit(prob, eps=1e-16):
    return np.log2((prob+eps)/(1-prob+eps))


def get_column_categories(table, sort=False):
    cats = OrderedDict()
    for kk,vv in table.dtypes.items():
        if vv is np.dtype('O'):
            cats[kk] = table[kk].fillna('').unique().tolist()
            if sort:
                cats[kk] = sorted(cats[kk])
        else:
            cats[kk] = []
    return cats


def hier_col_name_generator(categories):
    for cl, vv in categories.items():
        if len(vv) > 0:
            for cat in vv:
                yield '{}-{}'.format(cl, cat) if len(vv) > 0 else vv
        else:
            yield cl


def predict(model, data):
    """
    Model output (predicted) probabilities.
    Wrapper for predict_proba function in scikit-learn models.
    When a model does not have a predict_proba use predict interface.
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(data)
        if probs.shape[1] == 2:
            probs = probs[:, 1].ravel()
        else:
            probs = probs.ravel()
    else:
        probs = np.array(model.predict(data))
    return probs


def zero_pad(df, length=None):
    """
    Given a MAgEC dataframe (indexed with timepoint and case) zero-pad all features/columns
    """
    x = list()
    y = list()
    z = list()

    # assert 'timepoint' and 'case' exist in either index or columns
    assert 'timepoint' in df.index.names, "mising 'timepoint' from index"
    assert 'case' in df.index.names, "mising 'case' from index"

    if length is None:
        length = len(df.index.get_level_values('timepoint').unique())

    # use all features except for 'label', 'case' and 'timepoint'
    series_cols = list(set(df.columns) - {'case', 'timepoint', 'label'})

    for idx, fname in df.groupby(level='case', group_keys=False):

        if 'label' in df.columns:
            y_data = np.array(fname['label'].values[0])
            y.append(y_data)

        tmp = fname[series_cols].astype(float).values  # get all features as matrix of floats
        x_data = np.zeros([length, tmp.shape[1]])  # prepare zero pad matrix
        x_data[:tmp.shape[0], :] = tmp  # zero pad
        x.append(x_data)
        # format for pandas dataframe with columns containg time-series
        series = [[x_data[i, j] for i in range(x_data.shape[0])] for j in range(x_data.shape[1])]

        if 'label' in df.columns:
            z.append(pd.Series(series + [idx, y_data],
                               index=series_cols + ['case', 'label']))
        else:
            z.append(pd.Series(series + [idx],
                               index=series_cols + ['case']))
    x = np.array(x)
    y = np.array(y)
    z = pd.DataFrame.from_records(z)

    return x, y, z


def slice_series(target_data, tt, reverse=True):
    if reverse:
        df = target_data.loc[target_data.index.get_level_values('timepoint') >= tt]
    else:
        df = target_data.loc[target_data.index.get_level_values('timepoint') <= tt]
    return df


def static_prediction(model, target_data, score_preprocessing,
                      timepoint, var_name, epsilons, label='orig', baseline=None):
    idx = target_data.index.get_level_values('timepoint') == timepoint
    if label == 'orig':
        df = target_data.loc[idx].copy()
    elif label == 'perturb':
        df = target_data.loc[idx].copy()
        if baseline is None:
            if type(epsilons[var_name]) is list and len(epsilons[var_name]) == 2:
                new_val = (df.loc[:, var_name] == epsilons[var_name][0]).astype(int)
                new_val = new_val.multiply(epsilons[var_name][1]) + (1-new_val).multiply(epsilons[var_name][0])
            elif type(epsilons[var_name]) is list:
                raise ValueError('epsilon value can only be a scalar or have 2 values (binary)')
            else:
                new_val = epsilons[var_name]
            df.loc[:, var_name] = new_val
        else:
            if type(epsilons[var_name]) is list and len(epsilons[var_name]) == 2:
                new_val = (df.loc[:, var_name] == epsilons[var_name][0]).astype(int)
                new_val = new_val.multiply(epsilons[var_name][1]) + (1-new_val).multiply(epsilons[var_name][0])
            elif type(epsilons[var_name]) is list:
                raise ValueError('epsilon value can only be a scalar or have 2 values (binary)')
            else:
                tmp = df.loc[:, var_name]
                new_val = tmp - tmp * baseline
            df.loc[:, var_name] = new_val
    else:
        raise ValueError("label must be either 'orig' or' 'perturb")
    probs = predict(model, df)
    logits = score_preprocessing(probs)
    df_cols = df.columns
    df['probs_{}'.format(label)] = probs
    df['logit_{}'.format(label)] = logits
    df = df.drop(df_cols, axis=1)
    return df


def series_prediction(model, target_data, score_preprocessing,
                      timepoint, reverse, pad, var_name, epsilons,
                      label='orig', baseline=None):
    if label == 'orig':
        df = target_data.copy()
    elif label == 'perturb':
        df = target_data.copy()
        idx = df.index.get_level_values('timepoint') == timepoint
        if baseline is None:
            if type(epsilons[var_name]) is list and len(epsilons[var_name]) == 2:
                new_val = (df.loc[:, var_name] == epsilons[var_name][0]).astype(int)
                new_val = new_val.multiply(epsilons[var_name][1]) + (1-new_val).multiply(epsilons[var_name][0])
            elif type(epsilons[var_name]) is list:
                raise ValueError('epsilon value can only be a scalar or have 2 values (binary)')
            else:
                new_val = epsilons[var_name]
            df.loc[idx, var_name] = new_val  # perturb to new value
        else:
            if type(epsilons[var_name]) is list and len(epsilons[var_name]) == 2:
                new_val = (df.loc[:, var_name] == epsilons[var_name][0]).astype(int)
                new_val = new_val.multiply(epsilons[var_name][1]) + (1-new_val).multiply(epsilons[var_name][0])
            elif type(epsilons[var_name]) is list:
                raise ValueError('epsilon value can only be a scalar or have 2 values (binary)')
            else:
                tmp = df.loc[:, var_name]
                new_val = tmp - tmp * baseline
            df.loc[idx, var_name] = new_val
    else:
        raise ValueError("label must be either 'orig' or' 'perturb")
    df = slice_series(df, timepoint, reverse=reverse)
    x_series, _, df_vector = zero_pad(df, length=pad)
    df_cols = list(set(df_vector.columns) - {'case'})
    probs = predict(model, x_series)
    logits = score_preprocessing(probs)
    df_vector['probs_{}'.format(label)] = probs
    df_vector['logit_{}'.format(label)] = logits
    df_vector['timepoint'] = timepoint
    df_vector = df_vector.set_index(['case', 'timepoint'])
    df_vector = df_vector.drop(df_cols, axis=1)
    return df_vector.loc[df_vector.index.get_level_values('timepoint') == timepoint]


def z_perturbation(model, target_data,
                   score_preprocessing=get_logit,
                   score_comparison=lambda x_baseline, x: x - x_baseline,
                   sort_categories=True,
                   categories=None,
                   features=None,
                   binary=None,
                   timepoint_level='timepoint',
                   epsilon_value=0,
                   reverse=True,
                   timeseries=False,
                   baseline=None):
    '''
    Main method for computing a MAgEC. Assumes 'scaled/normalized' features in target data.
        Supporting 2 types of variables:
        - numeric / floats
        - binary / boolean
        Default score_comparison subtracts perturbed output from original.
        For a binary classification task, where 1 denotes a "bad" outcome, a good perturbation
        is expected to result in a negative score_comparison (assuming monotonic score_preprocessing).
    :param model:
    :param target_data:
    :param score_preprocessing:
    :param score_comparison:
    :param sort_categories:
    :param categories:
    :param features:
    :param binary:
    :param timepoint_level:
    :param epsilon_value:
    :param reverse:
    :param timeseries:
    :param baseline: whether to compute baseline MAgECS, None as default, 0.01 for 1% perturbation
    :return:
    '''
    # assert 'timepoint' and 'case' exist in either index or columns
    assert 'timepoint' in target_data.index.names, "mising 'timepoint' from index"
    assert 'case' in target_data.index.names, "mising 'case' from index"

    timepoints = list(sorted(target_data.index.get_level_values(timepoint_level).unique()))
    if reverse:
        timepoints = list(reversed(timepoints))

    if features is None:
        features = target_data.columns.unique()
    else:
        features = np.asarray(features)

    if binary is None:
        binary = target_data.apply(lambda x: len(np.unique(x)), ) <= 2
        binary = binary[binary].index.tolist()

    epsilons = dict()
    for var_name in features:
        if var_name in binary:
            epsilons[var_name] = target_data[var_name].unique().tolist()
            # epsilons[var_name] = target_data[var_name].value_counts().idxmax()  # most frequent value
        else:
            epsilons[var_name] = epsilon_value

    if categories is None:
        categories = get_column_categories(target_data[features], sort=sort_categories)

    prob_deltas_per_cell = pd.DataFrame(index=target_data.index,
                                        columns=pd.Index(hier_col_name_generator(categories),
                                                         name='features'))

    for tt in timepoints:

        # print("Timepoint {}".format(tt))

        if not timeseries:
            base = static_prediction(model,
                                     target_data,
                                     score_preprocessing,
                                     tt,
                                     var_name=None,
                                     epsilons=None,
                                     label='orig')
        else:
            base = series_prediction(model,
                                     target_data,
                                     score_preprocessing,
                                     tt,
                                     reverse,
                                     len(timepoints),
                                     var_name=None,
                                     epsilons=None,
                                     label='orig')

        for var_name in features:

            if not timeseries:
                # predict for perturbed data
                perturb = static_prediction(model,
                                            target_data,
                                            score_preprocessing,
                                            tt,
                                            var_name=var_name,
                                            epsilons=epsilons,
                                            label='perturb',
                                            baseline=baseline)
            else:
                # predict for perturbed data
                perturb = series_prediction(model,
                                            target_data,
                                            score_preprocessing,
                                            tt,
                                            reverse,
                                            len(timepoints),
                                            var_name=var_name,
                                            epsilons=epsilons,
                                            label='perturb',
                                            baseline=baseline)
            # logits
            logit_orig = base['logit_orig']
            logit_perturb = perturb['logit_perturb']
            logit_diff = score_comparison(logit_orig, logit_perturb)
            # store
            idx = target_data.index.get_level_values('timepoint') == tt
            prob_deltas_per_cell.loc[idx, var_name] = logit_diff
            prob_deltas_per_cell.loc[idx, 'perturb_{}_prob'.format(var_name)] = perturb['probs_perturb']
            prob_deltas_per_cell.loc[idx, 'orig_prob'] = base['probs_orig']

    return prob_deltas_per_cell.astype(float)


def m_prefix(magecs, feature, model_name=None):
    """
    Given a feature (e.g. BMI) and a magecs dataframe extract prefix (model_name).
    """
    prefix = 'm'
    for c in magecs.columns:
        splits = c.split('_')
        if len(splits) > 1 and feature == '_'.join(splits[1:]):
            prefix = splits[0]
            if model_name is not None:
                assert prefix == model_name
            break
    return prefix


def create_magec_col(model_name, feature):
    return model_name + '_' + feature


def case_magecs(model, data, epsilon_value=0, model_name=None,
                reverse=True, timeseries=False, baseline=None):
    """
    Compute MAgECs for every 'case' (individual row/member table).
    Use all features in data to compute MAgECs.
    NOTE 1: we prefix MAgECs with model_name.
    NOTE 2: we postfix non-MAgECs, such as 'perturb_<FEAT>_prob' with model_name.
    """
    magecs = z_perturbation(model, data,
                            epsilon_value=epsilon_value,
                            reverse=reverse,
                            timeseries=timeseries,
                            baseline=baseline)
    features = magecs.columns
    magecs = magecs.reset_index()
    # rename features in case_magecs to reflect the fact that they are derived for a specific model
    prefix = 'm' if model_name is None else model_name
    postfix = prefix
    for feat in features:
        if feat == 'orig_prob' or (feat[:8] == 'perturb_' and feat[-5:] == '_prob'):
            magecs.rename(columns={feat: feat + '_' + postfix}, inplace=True)
        else:
            magecs.rename(columns={feat: create_magec_col(prefix, feat)}, inplace=True)
    return magecs


def normalize_magecs(magecs,
                     features=None,
                     model_name=None):
    """
    Normalize MAgECs for every 'case' using an L2 norm.
    Use (typically) all MAgEC columns (or a subset of features). Former is advised.
    NOTE: The convention is that MAgECs are prefixed with model_name.
    """
    out = magecs.copy()

    if features is None:
        prefix = 'm_' if model_name is None else model_name + '_'
        cols = [c for c in magecs.columns if c.startswith(prefix)]
    else:
        cols = [create_magec_col(m_prefix(magecs, feat, model_name), feat) for feat in features]

    for (idx, row) in out.iterrows():
        norm = np.linalg.norm(row.loc[cols].values)
        out.loc[idx, cols] = out.loc[idx, cols] / norm
    return out


def plot_all_violin(magecs,
                    features=('Age', 'BloodPressure', 'BMI', 'Glucose', 'Insulin',
                              'SkinThickness', 'DiabetesPedigreeFunction'),
                    model_name=None):
    """
    Violin plots for MAgECs.
    """
    colors = list(mcolors.TABLEAU_COLORS)
    fig, ax = plt.subplots(nrows=1, ncols=len(features), figsize=(16, 10))
    ymin = 10
    ymax = -10
    for i, feat in enumerate(features):
        mfeat = m_prefix(magecs, feat, model_name) + "_" + feat
        ymin = min(np.min(magecs[mfeat]), ymin)
        ymax = max(np.max(magecs[mfeat]), ymax)
        sns.violinplot(y=magecs[mfeat], ax=ax[i], color=colors[i % len(colors)])
        ax[i].grid('on')
        ax[i].set_ylabel('')
        ax[i].set_title(feat)
    for i in range(len(features)):
        ax[i].set_ylim([1.5 * ymin, 1.5 * ymax])


def magec_cols(magec, features):
    all_cols = magec.columns
    orig_prob_col = [col for col in all_cols if col.startswith('orig_prob_')]
    jcols = ['case', 'timepoint']
    m_cols = [col for col in all_cols if '_'.join(col.split('_')[1:]) in features]
    prob_cols = [col for col in all_cols if col.startswith('perturb_') and
                 col[8:].split('_prob_')[0] in features]
    cols = jcols + m_cols + prob_cols + orig_prob_col
    return jcols, cols


def magec_models(*magecs, **kwargs):
    """
    Wrapper function for joining MAgECs from different models together and (optionally) w/ tabular data
    """
    Xdata = kwargs.get('Xdata', None)
    Ydata = kwargs.get('Ydata', None)
    features = kwargs.get('features', [])
    assert len(magecs) > 1
    jcols, cols = magec_cols(magecs[0], features)
    magec = magecs[0][cols]
    if Xdata is not None:
        magec = magec.merge(Xdata.reset_index(), left_on=jcols, right_on=jcols)
    if Ydata is not None:
        magec = magec.merge(Ydata.reset_index(), left_on=jcols, right_on=jcols)
    for mgc in magecs[1:]:
        _, cols = magec_cols(mgc, features)
        mgc = mgc[cols]
        magec = magec.merge(mgc, left_on=jcols, right_on=jcols)
    return magec


def magec_rank(magecs,
               models=('mlp', 'rf', 'lr'),
               rank=3,
               features=('BloodPressure', 'BMI', 'Glucose', 'Insulin', 'SkinThickness'),
               outcome='Outcome'):
    """
    Compute top-magecs (ranked) for each model for each 'case/timepoint' (individual row in tabular data).
    Input is a list of one or more conputed magecs given a model.
    Output is a Pandas dataframe with computed magecs, filtering out positive magecs.
    Positive magecs indicate counter-productive interventions.
    """
    ranks = {}

    # each row contains all MAgEC coefficients for a 'case/timepoint'
    for (idx, row) in magecs.iterrows():
        model_ranks = {}
        if outcome in row:
            key = (row['case'], row['timepoint'], row[outcome])
        else:
            key = (row['case'], row['timepoint'])
        for model in models:
            # initialize all models coefficients (empty list)
            model_ranks[model] = list()
        for col in features:
            # iterate of all features
            for model in models:
                # each model should contain a corresponding magec
                feat = create_magec_col(model, col)
                assert feat in row, "feature {} not in magecs".format(feat)
                magec = row[feat]
                # we are using a priority queue for the magec coefficients
                # heapq is a min-pq, we are reversing the sign so that we can use a max-pq
                if len(model_ranks[model]) < rank:
                    heapq.heappush(model_ranks[model], (-magec, col))
                else:
                    _ = heapq.heappushpop(model_ranks[model], (-magec, col))
                    # store magecs (top-N where N=rank) for each key ('case/timepoint')
        ranks[key] = model_ranks
        # create a Pandas dataframe with all magecs for a 'case/timepoint'
    out = list()
    out_col = None
    columns = []
    for k, v in ranks.items():
        if len(k) == 3:
            l = [k[0], k[1], k[2]]
            if out_col is None:
                out_col = outcome
                columns = ['case', 'timepoint', outcome]
        else:
            l = [k[0], k[1]]
            if not len(columns):
                columns = ['case', 'timepoint']
        for model in models:
            while v[model]:  # retrieve priority queue's magecs (max-pq with negated (positive) magecs)
                magec, feat = heapq.heappop(v[model])
                if magec < 0:  # negative magecs are originally positive magecs and are filtered out
                    l.append(None)
                    l.append("not_found")
                else:
                    l.append(-magec)  # retrieve original magec sign
                    l.append(feat)
        out.append(l)

    out = pd.DataFrame.from_records(out)
    # create dataframe's columns
    for model in models:
        if rank == 1:
            columns.append(model + '_magec')
            columns.append(model + '_feat')
        else:
            for r in range(rank, 0, -1):
                columns.append(model + '_magec_{}'.format(r))
                columns.append(model + '_feat_{}'.format(r))
    out.columns = columns
    out['case'] = out['case'].astype(magecs['case'].dtype)
    out['timepoint'] = out['timepoint'].astype(magecs['timepoint'].dtype)
    if out_col:
        out[out_col] = out[out_col].astype(magecs[out_col].dtype)

    pert_cols = ['perturb_' + col + '_prob' + '_' + model for col in features for model in models]
    orig_cols = ['orig_prob_' + model for model in models]
    all_cols = ['case', 'timepoint'] + pert_cols + orig_cols + features
    out = out.merge(magecs[all_cols],
                    left_on=['case', 'timepoint'],
                    right_on=['case', 'timepoint'])
    return out


def print_ranks_stats(ranks, models=('mlp', 'rf', 'lr')):
    columns = ranks.columns
    for model in models:
        cols = [col for col in columns if col.startswith(model + '_' + 'feat')]
        if len(cols):
            print("\t {} MAgEC Stats".format(model))
            for col in cols:
                print("**** " + col + " ****")
                print(ranks[col].value_counts())
                print("***********")


def magec_rbos(ranks, models=('mlp', 'rf', 'lr'), p=0.9):
    """
    Given a ranked list of magecs from one or more models compute pairwise RBOs.
    :param ranks:
    :param models:
    :param p: RBO's p'value
    :return:
    """

    models = sorted(models)

    cols = [c for c in ranks if '_feat_' in c]

    m_cols = dict()
    for model in models:
        t_sorted = sorted([('_'.join(c.split('_')[:-1]),
                            int(c.split('_')[-1])) for c in cols if c.startswith(model)],
                          key=lambda x: x[1])
        m_cols[model] = [t[0] + '_' + str(t[1]) for t in t_sorted]

    out = list()

    combos = [(m1, m2) for i, m1 in enumerate(models) for j, m2 in enumerate(models) if i > j]

    for (_, row) in ranks.iterrows():

        case = row.case
        timepoint = row.timepoint

        m_ranked = dict()
        for m, cols in m_cols.items():
            feats = list()
            for col in cols:
                feat = row[col]
                if feat != 'not_found':
                    feats.append(feat)
            m_ranked[m] = feats

        combo_sim = list()
        for c in combos:
            l1 = m_ranked[c[0]]
            l2 = m_ranked[c[1]]
            sim = rbo.RankingSimilarity(l1, l2).rbo(p=p)
            combo_sim.append(sim)

        case_out = [case, timepoint] + [feats for _, feats in m_ranked.items()] + combo_sim
        case_cols = ['case', 'timepoint'] + [m + '_ranked' for m, _ in m_ranked.items()] + [c[0] + '_' + c[1] for c in
                                                                                            combos]

        out.append(pd.Series(case_out, index=case_cols))

    return pd.DataFrame.from_records(out)


def magec_consensus(magec_ranks,
                    models=('mlp', 'rf', 'lr'),
                    use_weights=False,
                    weights={'rf': None, 'mlp': None, 'lr': None},
                    outcome='Outcome',
                    policy='sum'):
    """
    Given a ranked list of magecs from one or more models compute a single most-important magec.
    There are 2 types of "MAgEC" columns in magec_ranks:
    1. 'feat_' with the name of the magec feature
    2. 'magec_' with the value of the magec
    The prefix in the column name indicates the model used (e.g. 'mlp_').
    The column names end in _{rank_number} when rank > 1.
    INPUT magec_ranks EXAMPLE when rank > 1
        case                                          2
        timepoint                                     0
        Outcome                                       1
        mlp_magec_3                                 NaN
        mlp_feat_3                            not_found
        mlp_magec_2                                 NaN
        mlp_feat_2                            not_found
        mlp_magec_1                            -0.16453
        mlp_feat_1                        SkinThickness
        rf_magec_3                           -0.0511598
        rf_feat_3                         SkinThickness
        rf_magec_2                            -0.152834
        rf_feat_2                         BloodPressure
        rf_magec_1                            -0.282895
        rf_feat_1                               Glucose
        lr_magec_3                           -0.0158208
        lr_feat_3                         SkinThickness
        lr_magec_2                           -0.0356751
        lr_feat_2                               Insulin
        lr_magec_1                            -0.614731
        lr_feat_1                               Glucose
        perturb_BloodPressure_prob_mlp         0.658987
        perturb_BloodPressure_prob_rf          0.433044
        perturb_BloodPressure_prob_lr          0.784033
        perturb_BMI_prob_mlp                   0.745703
        perturb_BMI_prob_rf                    0.802258
        perturb_BMI_prob_lr                    0.822721
        perturb_Glucose_prob_mlp               0.604908
        perturb_Glucose_prob_rf                0.383129
        perturb_Glucose_prob_lr                0.493356
        perturb_Insulin_prob_mlp               0.618258
        perturb_Insulin_prob_rf                0.484568
        perturb_Insulin_prob_lr                0.721596
        perturb_SkinThickness_prob_mlp         0.517638
        perturb_SkinThickness_prob_rf          0.473091
        perturb_SkinThickness_prob_lr          0.728289
        orig_prob_mlp                          0.591666
        orig_prob_rf                           0.493407
        orig_prob_lr                           0.733549
        BloodPressure                           1.11285
        BMI                                   -0.697264
        Glucose                                0.992034
        Insulin                               -0.435478
        SkinThickness                          0.779127

    INPUT magec_ranks EXAMPLE when rank = 1
        case                                          2
        timepoint                                     0
        Outcome                                       1
        mlp_magec                              -0.16453
        mlp_feat                          SkinThickness
        rf_magec                              -0.282895
        rf_feat                                 Glucose
        lr_magec                              -0.614731
        lr_feat                                 Glucose
        perturb_BloodPressure_prob_mlp         0.658987
        perturb_BloodPressure_prob_rf          0.433044
        perturb_BloodPressure_prob_lr          0.784033
        perturb_BMI_prob_mlp                   0.745703
        perturb_BMI_prob_rf                    0.802258
        perturb_BMI_prob_lr                    0.822721
        perturb_Glucose_prob_mlp               0.604908
        perturb_Glucose_prob_rf                0.383129
        perturb_Glucose_prob_lr                0.493356
        perturb_Insulin_prob_mlp               0.618258
        perturb_Insulin_prob_rf                0.484568
        perturb_Insulin_prob_lr                0.721596
        perturb_SkinThickness_prob_mlp         0.517638
        perturb_SkinThickness_prob_rf          0.473091
        perturb_SkinThickness_prob_lr          0.728289
        orig_prob_mlp                          0.591666
        orig_prob_rf                           0.493407
        orig_prob_lr                           0.733549
        BloodPressure                           1.11285
        BMI                                   -0.697264
        Glucose                                0.992034
        Insulin                               -0.435478
        SkinThickness                          0.779127
    """

    cols = list(set(magec_ranks.columns) - {'case', 'timepoint', outcome})

    def name_matching(cols, models):
        # get all magec column names
        col_names = dict()
        for col in cols:
            prefix = col.split('_')[0]
            if prefix in models:
                if prefix in col_names:
                    col_names[prefix].append(col)
                else:
                    col_names[prefix] = [col]
        # magec/feat column names come in pairs
        magecs_feats = dict()
        for model, cols in col_names.items():
            feat2magic = dict()
            assert len(cols) % 2 == 0, "magec/feat cols should come in pairs"
            if len(cols) == 2:
                if 'feat' in cols[0] and 'magec' in cols[1]:
                    feat2magic[cols[0]] = cols[1]
                elif 'feat' in cols[1] and 'magec' in cols[0]:
                    feat2magic[cols[1]] = cols[0]
                else:
                    raise ValueError('magec/feat substring not present in column names')
            else:
                # reversed names sorted (e.g. 1_taef_plm)
                feats = sorted([col[::-1] for col in cols if 'feat' in col])
                # reversed names sorted (e.g. 1_cegam_plm)
                magecs = sorted([col[::-1] for col in cols if 'magec' in col])
                assert len(feats) == len(cols) / 2, "'feat' substring missing in column name"
                assert len(magecs) == len(cols) / 2, "'magec' substring missing in column name"
                for i, feat in enumerate(feats):
                    feat2magic[feat[::-1]] = magecs[i][::-1]
            # return dictionary with magec feature column names and magec value column name for every model
            magecs_feats[model] = feat2magic
        return magecs_feats

    magecs_feats = name_matching(cols, models)

    out = list()
    for (idx, row) in magec_ranks.iterrows():
        member = list()
        key = row['case'], row['timepoint']
        winner = magec_winner(magecs_feats, row, use_weights=use_weights, weights=weights, policy=policy)
        member.append(key[0])
        member.append(key[1])
        winner_feat = None if winner is None else winner[0]
        winner_score = None if winner is None else winner[1]
        winner_consensus = None if winner is None else winner[2]
        winner_models = None if winner is None else winner[3]
        winner_prob_ratio_all = None
        winner_prob_ratio = None
        if winner_feat is not None:
            all_model_ratios = list()
            model_ratios = list()
            for model in magecs_feats.keys():
                orig_prob = row['orig_prob_' + model]
                feat_prob = row['perturb_' + winner_feat + '_prob_' + model]
                ratio = 100*(orig_prob-feat_prob+1e-6) / (orig_prob+1e-6)
                all_model_ratios.append(ratio)
                if model in winner_models:
                    model_ratios.append(ratio)
            winner_prob_ratio_all = np.mean(all_model_ratios)
            winner_prob_ratio = np.mean(model_ratios)
        member.append(winner_feat)
        member.append(winner_score)
        member.append(winner_consensus)
        member.append(winner_models)
        member.append(winner_prob_ratio)
        member.append(winner_prob_ratio_all)
        out.append(member)
    out = pd.DataFrame.from_records(out)
    out.columns = ['case', 'timepoint',
                   'winner', 'score',
                   'consensus', 'models',
                   'avg_percent_consensus', 'avg_percent_all']
    return out


def magec_winner(magecs_feats,
                 row,
                 scoring=lambda w: abs(w),
                 use_weights=False,
                 weights={'rf': None, 'mlp': None, 'lr': None},
                 policy='sum'):
    """
    Compute MAgEC winner from a list of MAgECs from one or more models for a single 'case/timepoint'.
    magecs_feats is a dictionary with magec feature column names and magec value column names for every model,
     e.g
    {'rf': {'rf_feat_1': 'rf_magec_1', 'rf_feat_2': 'rf_magec_2'},
     'mlp': {'mlp_feat_1': 'mlp_magec_1', 'mlp_feat_2': 'mlp_magec_2'},
     'lr': {'lr_feat_1': 'lr_magec_1', 'lr_feat_2': 'lr_magec_2'}}
    """

    assert policy in ['sum', 'mean'], "Only 'sum' or 'mean' policy is supported"

    winner = None
    consensus = {}
    scores = {}
    if use_weights:
        assert sorted(weights.keys()) == sorted(magecs_feats.keys())
    for model, feat_dict in magecs_feats.items():
        for feat_col, score_col in feat_dict.items():
            feat = row[feat_col]
            if feat == 'not_found':
                continue
            score = scoring(row[score_col])
            if use_weights:
                if weights[model] is not None:
                    score *= weights[model]
            if feat in scores:
                scores[feat] += score
                consensus[feat].add(model)
            else:
                scores[feat] = score
                consensus[feat] = {model}
    # get consensus
    for feat, score in scores.items():
        if policy == 'mean':
            score /= len(consensus[feat])
        if winner is None or score > winner[1]:
            winner = (feat, score, len(consensus[feat]), sorted(list(consensus[feat])))

    return winner


def enhance_consensus(consensus, rbos, models=('mlp', 'rf', 'lr')):
    m = sorted(models)
    combos = [m1+'_'+m2 for i, m1 in enumerate(m) for j, m2 in enumerate(m) if i > j]
    jcols = ['case', 'timepoint']
    consensus = consensus.merge(rbos[jcols+combos], left_on=jcols, right_on=jcols)
    data = list()
    for (_, row) in consensus.iterrows():
        case = row.case
        timepoint = row.timepoint
        winner = row.winner
        score = row.score
        consensus = row.consensus
        avg_percent_consensus = row.avg_percent_consensus
        avg_percent_all = row.avg_percent_all
        mdls = sorted(row.models)
        rbos = [m1+'_'+m2 for i, m1 in enumerate(mdls) for j, m2 in enumerate(mdls) if i > j]
        rbo_min = np.min(row[rbos])
        rbo_max = np.max(row[rbos])
        tmp = pd.Series((case, timepoint, winner, score, consensus,
                         avg_percent_consensus, avg_percent_all, rbo_min, rbo_max),
                        index=['case', 'timepoint', 'winner', 'score', 'consensus',
                               'avg_percent_consensus', 'avg_percent_all', 'rbo_min', 'rbo_max'])
        data.append(tmp)
    return pd.DataFrame.from_records(data)


def magec_similarity(case_magecs,
                     x_validation_p,
                     features=('BloodPressure', 'BMI', 'Glucose', 'Insulin',
                               'SkinThickness', 'DiabetesPedigreeFunction'),
                     model_name=None):
    if model_name is None:
        model_name = "m"
    cols = [model_name + "_" + feat for feat in features] + list(features) + ['Outcome']
    df = case_magecs.merge(x_validation_p, left_on=['case', 'timepoint'], right_index=True)[cols]
    return df


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


def pdf(data):
    # Empirical average and variance
    avg = np.mean(data)
    var = np.var(data)
    # Gaussian PDF
    pdf_x = np.linspace(np.min(data), np.max(data), 100)
    pdf_y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (pdf_x - avg) ** 2 / var)
    return pdf_x, pdf_y, avg, var


def plot_train_valid(train, valid, feature):
    train = train[feature]
    valid = valid[feature]
    pdf_x_t, pdf_y_t, avg_t, var_t = pdf(train)
    pdf_x_v, pdf_y_v, avg_v, var_v = pdf(valid)
    # Figure
    plt.figure(figsize=(10, 8))
    plt.hist(train, 30, density=True, alpha=0.5)
    plt.hist(valid, 30, density=True, alpha=0.5)
    plt.plot(pdf_x_t, pdf_y_t, 'b--')
    plt.plot(pdf_x_v, pdf_y_v, 'g--')
    plt.legend(["train fit", "valid fit", "train", "valid"])
    plt.title("mean ({:.2g}, {:.2g}), std ({:.2g}, {:.2g})".format(avg_t, avg_v, np.sqrt(var_t), np.sqrt(var_v)))
    plt.show()


def plot_feature(df, feature, ax=None):
    data = df[feature]
    pdf_x, pdf_y, avg, var = pdf(data)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(data, 30, density=True, alpha=0.5)
    ax.plot(pdf_x, pdf_y, 'b--')
    ax.set_title("{} (mean: {:.2g}, std: {:.2g})".format(feature, avg, np.sqrt(var)))
    return ax


def model_performance(model, X, y, subtitle):
    # Kfold
    cv = KFold(n_splits=5, shuffle=False, random_state=42)
    y_real = []
    y_proba = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 1

    for train, test in cv.split(X, y):
        model.fit(X.iloc[train], y.iloc[train])
        pred_proba = model.predict_proba(X.iloc[test])
        precision, recall, _ = precision_recall_curve(y.iloc[test], pred_proba[:, 1])
        y_real.append(y.iloc[test])
        y_proba.append(pred_proba[:, 1])
        fpr, tpr, t = roc_curve(y[test], pred_proba[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Confusion matrix
    y_pred = cross_val_predict(model, X, y, cv=5)
    conf_matrix = confusion_matrix(y, y_pred)
    trace1 = go.Heatmap(z=conf_matrix, x=["0 (pred)", "1 (pred)"],
                        y=["0 (true)", "1 (true)"], xgap=2, ygap=2,
                        colorscale='Viridis', showscale=False)

    # Show metrics
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]
    fp = conf_matrix[0, 1]
    tn = conf_matrix[0, 0]
    Accuracy = ((tp + tn) / (tp + tn + fp + fn))
    Precision = (tp / (tp + fp))
    Recall = (tp / (tp + fn))
    F1_score = (2 * (((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))))

    show_metrics = pd.DataFrame(data=[[Accuracy, Precision, Recall, F1_score]])
    show_metrics = show_metrics.T

    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2 = go.Bar(x=(show_metrics[0].values),
                    y=['Accuracy', 'Precision', 'Recall', 'F1_score'], text=np.round_(show_metrics[0].values, 4),
                    textposition='auto', textfont=dict(color='black'),
                    orientation='h', opacity=1, marker=dict(
            color=colors,
            line=dict(color='#000000', width=1.5)))

    # Roc curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    trace3 = go.Scatter(x=mean_fpr, y=mean_tpr,
                        name="Roc : ",
                        line=dict(color=('rgb(22, 96, 167)'), width=2), fill='tozeroy')
    trace4 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color=('black'), width=1.5,
                                  dash='dot'))

    # Precision - recall curve
    y_real = y
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    trace5 = go.Scatter(x=recall, y=precision,
                        name="Precision" + str(precision),
                        line=dict(color=('lightcoral'), width=2), fill='tozeroy')

    mean_auc = round(mean_auc, 3)

    # Subplots
    fig = tls.make_subplots(rows=2, cols=2, print_grid=False,
                            specs=[[{}, {}],
                                   [{}, {}]],
                            subplot_titles=('Confusion Matrix',
                                            'Metrics',
                                            'ROC curve' + " " + '(' + str(mean_auc) + ')',
                                            'Precision - Recall curve',
                                            ))
    # Trace and layout
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace4, 2, 1)
    fig.append_trace(trace5, 2, 2)

    fig['layout'].update(showlegend=False, title='<b>Model performance report (5 folds)</b><br>' + subtitle,
                         autosize=False, height=830, width=830,
                         plot_bgcolor='black',
                         paper_bgcolor='black',
                         margin=dict(b=195), font=dict(color='white'))
    fig["layout"]["xaxis1"].update(color='white')
    fig["layout"]["yaxis1"].update(color='white')
    fig["layout"]["xaxis2"].update((dict(range=[0, 1], color='white')))
    fig["layout"]["yaxis2"].update(color='white')
    fig["layout"]["xaxis3"].update(dict(title="false positive rate"), color='white')
    fig["layout"]["yaxis3"].update(dict(title="true positive rate"), color='white')
    fig["layout"]["xaxis4"].update(dict(title="recall"), range=[0, 1.05], color='white')
    fig["layout"]["yaxis4"].update(dict(title="precision"), range=[0, 1.05], color='white')
    for i in fig['layout']['annotations']:
        i['font'] = titlefont = dict(color='white', size=14)
    py.iplot(fig)


def scores_table(model, X, y, subtitle):
    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    res = []
    for sc in scores:
        scores = cross_val_score(model, X, y, cv=5, scoring=sc)
        res.append(scores)
    df = pd.DataFrame(res).T
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    df = df.rename(columns={0: 'accuracy', 1: 'precision', 2: 'recall', 3: 'f1', 4: 'roc_auc'})

    trace = go.Table(
        header=dict(values=['<b>Fold', '<b>Accuracy', '<b>Precision', '<b>Recall', '<b>F1 score', '<b>Roc auc'],
                    line=dict(color='#7D7F80'),
                    fill=dict(color='#a1c3d1'),
                    align=['center'],
                    font=dict(size=15)),
        cells=dict(values=[('1', '2', '3', '4', '5', 'mean', 'std'),
                           np.round(df['accuracy'], 3),
                           np.round(df['precision'], 3),
                           np.round(df['recall'], 3),
                           np.round(df['f1'], 3),
                           np.round(df['roc_auc'], 3)],
                   line=dict(color='#7D7F80'),
                   fill=dict(color='#EDFAFF'),
                   align=['center'], font=dict(size=15)))

    layout = dict(width=800, height=400, title='<b>Cross Validation - 5 folds</b><br>' + subtitle, font=dict(size=15))
    fig = dict(data=[trace], layout=layout)

    py.iplot(fig, filename='styled_table')


def predict_classes(model, data):
    """
    Model output (predicted) classes.
    """
    if hasattr(model, 'predict_classes'):
        return model.predict_classes(data).ravel()
    else:
        return model.predict(data).ravel()


def model_metrics(yhat_probs, yhat_classes, y_test, verbose=False):
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    if verbose:
        print('Accuracy: %f' % accuracy)

    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    if verbose:
        print('Precision: %f' % precision)

    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    if verbose:
        print('Recall: %f' % recall)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    if verbose:
        print('F1 score: %f' % f1)

    # ROC AUC
    roc_auc = roc_auc_score(y_test, yhat_probs)
    if verbose:
        print('ROC AUC: %f' % roc_auc)

    # confusion matrix
    matrix = confusion_matrix(y_test, yhat_classes)
    if verbose:
        print(matrix)

    return accuracy, precision, recall, f1, roc_auc


def evaluate(model, x_test, y_test, verbose=False):
    # predict probabilities for test set
    yhat_probs = predict(model, x_test)

    # predict classes for test set
    yhat_classes = predict_classes(model, x_test)

    # reduce to 1d array
    if len(yhat_probs[0].shape):
        yhat_probs = yhat_probs[:, 0]
        yhat_classes = yhat_classes[:, 0]

    return model_metrics(yhat_probs, yhat_classes, y_test, verbose=verbose)


def bold_column(table):
    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    return


def red_cell(table, r, c):
    for (row, col), cell in table.get_celld().items():
        if (row == r) and (col == c):
            cell.set_text_props(fontproperties=FontProperties(color='red'))

    return


def case_stats(data, case, timepoint=None, models=('lr', 'rf', 'mlp')):

    tmp = [col for col in data.columns if col.startswith('perturb_') and col[-len(models[0]):] == models[0]]
    feats = [col.split('perturb_')[1].split('_prob_')[0] for col in tmp]
    out = list()
    for feat in feats:
        for model in models:
            l = [case, timepoint]
            magec = model + '_' + feat
            perturb = 'perturb_' + feat + '_prob_' + model
            orig = 'orig_prob_' + model
            if magec in data and perturb in data and orig in data:
                magec = data[magec].values[0]
                perturb = data[perturb].values[0]
                orig = data[orig].values[0]
                l.append(model)
                l.append(feat)
                l.append(magec)
                l.append(orig)
                l.append(perturb)
                l.append(100 * (orig - perturb) / orig)
                out.append(pd.Series(l, index=['case', 'timepoint', 'model', 'feature',
                                               'magec', 'risk', 'risk_new', 'risk_prc_reduct']))
    return pd.DataFrame.from_records(out)


def magec_threshold(data, features, threshold=0.5, ensemble='ensemble', models=('lr', 'rf', 'mlp')):
    """
    A case for which the ensemble prediction is above a threshold (0.5),
    but the model predictions for perturbed features are below the threshold.
    The 'magec_postive' number denotes the number of models with at least one such feature.
    :param data:
    :param features:
    :param threshold:
    :param ensemble:
    :param models: 
    :return:
    """
    col = 'orig_prob_' + ensemble
    assert col in data, "{} not in dataframe".format(col)
    out = 0
    if data[col] <= threshold:
        return out
    else:
        for model in models:
            for feat in features:
                fcol = 'perturb_' + feat + '_prob_' + model
                assert fcol in data, "{} not in dataframe".format(fcol)
                if data[fcol] > threshold:
                    out += 1
                    break
    return out


def panel_plot(train_cols, features, stsc, joined, case, timepoint=None,
               models=('lr', 'rf', 'mlp'), label='Outcome',
               limit=None, rotate=None, save=None, title=None, magec_ensemble=False):

    if timepoint is None:
        data = joined.loc[joined.case == case]
    else:
        data = joined.loc[(joined.case == case) & (joined.timepoint == timepoint)]

    case_df = case_stats(data, case, timepoint, models=models)

    if limit is not None:
        topK = case_df.groupby('feature')['risk_new'].mean().sort_values()[:limit].index.values
        case_df = case_df[np.isin(case_df['feature'], topK)]
        train_cols_idx = [train_cols.to_list().index(x) for x in topK]
        features = topK
    else:
        train_cols_idx = [i for i in range(len(train_cols))]

    fig = plt.figure(figsize=(14, 10))
    grid = plt.GridSpec(3, 5, wspace=0.2, hspace=0.1)

    main_fig = fig.add_subplot(grid[0, 0:2])
    ml_fig = fig.add_subplot(grid[1, 0:2])
    mg_fig = fig.add_subplot(grid[2, :])
    bar_fig = fig.add_subplot(grid[:2, 2:])

    base = case_df.groupby('feature')['risk'].mean()

    bar_fig = sns.barplot(x="feature", y="risk_new", data=case_df, ci=None, ax=bar_fig)
    bar_fig.plot(np.linspace(bar_fig.get_xlim()[0], bar_fig.get_xlim()[1], 10),
                 np.mean(base.values) * np.ones(10), '--')
    bar_fig.legend(['current risk', 'estimated risk'], loc='upper right')
    bar_fig.set_ylabel('estimated risk')
    bar_fig.set_ylim([0, min(round(1.2*bar_fig.get_ylim()[1], 1), 1)])
    if rotate is not None:
        bar_fig.set_xticklabels(bar_fig.get_xticklabels(), rotation=rotate)
    bar_fig.set_xlabel('')

    collabel0 = ["Case", str(case)]

    cell_feat = [feat for feat in train_cols[train_cols_idx]]
    cell_vals = [round(val, 3) for val in stsc.inverse_transform(data[train_cols])[0][train_cols_idx]]
    celldata0 = [[x[0], x[1]] for x in zip(cell_feat, cell_vals)] + [['True Outcome', data[label].values[0]]]

    collabel1 = ["Model", "Predicted Risk"]

    celldata1 = [[model.upper(), round(data['orig_prob_'+model].values[0], 3)] for model in models]

    collabel2 = ["Model"] + ["MAgEC " + feat for feat in features]
    celldata2 = list()

    if not magec_ensemble:
        models = [m for m in models if m != 'ensemble']

    for model in models:
        add_model = True
        line = list()
        for feat in features:
            f = model + '_' + feat
            if f not in data:
                add_model = False
                break
            else:
                line.append(round(data[f].values[0], 3))
        if add_model:
            celldata2.append([model.upper()] + line)

    main_fig.axis('tight')
    main_fig.axis('off')
    ml_fig.axis('tight')
    ml_fig.axis('off')
    mg_fig.axis('tight')
    mg_fig.axis('off')

    table0 = main_fig.table(cellText=celldata0, colLabels=collabel0, loc='center', cellLoc='center')
    table1 = ml_fig.table(cellText=celldata1, colLabels=collabel1, loc='center', cellLoc='center')
    table2 = mg_fig.table(cellText=celldata2, colLabels=collabel2, loc='center', cellLoc='center')

    table0.set_fontsize(12)
    table0.scale(1.5, 1.5)

    table1.set_fontsize(12)
    table1.scale(1.5, 1.5)

    table2.set_fontsize(12)
    table2.scale(1.5, 1.5)

    table0.auto_set_column_width(col=list(range(len(collabel0))))
    table1.auto_set_column_width(col=list(range(len(collabel1))))
    table2.auto_set_column_width(col=list(range(len(collabel2))))

    bold_column(table0)
    bold_column(table1)
    bold_column(table2)

    if title is not None:
        plt.title(title)

    if save is not None:
        plt.savefig(str(save)+'.png', bbox_inches='tight')

    return fig


def build_base_rbos(mlp, sigmoidRF, lr, x_validation_p, y_validation_p, features, weights, baseline=0.01):
    models = ('lr', 'rf', 'mlp')
    # MLP
    base_case_mlp = case_magecs(mlp, x_validation_p, model_name='mlp', baseline=baseline)
    base_magecs_mlp = normalize_magecs(base_case_mlp, features=None, model_name='mlp')
    # RF
    base_case_rf = case_magecs(sigmoidRF, x_validation_p, model_name='rf', baseline=baseline)
    base_magecs_rf = normalize_magecs(base_case_rf, features=None, model_name='rf')
    # LR
    base_case_lr = case_magecs(lr, x_validation_p, model_name='lr', baseline=baseline)
    base_magecs_lr = normalize_magecs(base_case_lr, features=None, model_name='lr')

    base_joined = magec_models(base_magecs_mlp,
                               base_magecs_rf,
                               base_magecs_lr,
                               Xdata=x_validation_p,
                               Ydata=y_validation_p,
                               features=features)

    base_ranks = magec_rank(base_joined, rank=len(features), features=features)

    base_consensus = magec_consensus(base_ranks, use_weights=True, weights=weights, models=models)
    base_rbos = magec_rbos(base_ranks, models=models)

    return base_rbos, base_consensus, base_ranks


def ranked_stats(ranks):
    columns = ranks.columns
    stats = {}
    for model in ['lr', 'rf', 'mlp']:
        cols = [col for col in columns if col.startswith(model + '_' + 'feat')]
        if len(cols):
            for col in cols:
                tmp = ranks[col].value_counts()
                for z in zip(tmp.index.values.tolist(), tmp.values.tolist()):
                    if z[0] in stats:
                        stats[z[0]].append((z[1], model))
                    else:
                        stats[z[0]] = [(z[1], model)]
    return stats


def con_stats(consensus, label='CON@1'):
    conf = consensus.winner.value_counts().index.values.tolist()
    conv = consensus.winner.value_counts().values.tolist()
    con = {z[0]: (z[1], label) for z in zip(conf, conv)}
    return con


def get_data_importance(pima, x_train, y_train):
  pm.plot_pima_features(pima)

  rf = RandomForestClassifier(n_estimators=1000)

  model = rf.fit(x_train, y_train)

  cols = x_train.columns
  importance = model.feature_importances_

  zipped = list(zip(cols, importance))
  sorted_importance = reversed(sorted(zipped, key=lambda x: x[1]))
  print('Sorted feature importance from Random Forest Classifier:')
  for feature in sorted_importance:
    print(f'{feature[1]} -- {feature[0]}')


def generate_mlp_rf_svm_magecs(x_validation, y_validation, models):
  # MLP
  case_mlp = case_magecs(models['mlp'], x_validation, model_name='mlp')
  magecs_mlp = normalize_magecs(case_mlp, features=None, model_name='mlp')
  magecs_mlp = magecs_mlp.merge(y_validation, left_on=['case', 'timepoint'], right_index=True)
  # RF
  case_rf = case_magecs(models['rf'], x_validation, model_name='rf')
  magecs_rf = normalize_magecs(case_rf, features=None, model_name='rf')
  magecs_rf = magecs_rf.merge(y_validation, left_on=['case', 'timepoint'], right_index=True)
  # SVM
  case_svm = case_magecs(models['svm'], x_validation, model_name='svm')
  magecs_svm = normalize_magecs(case_svm, features=None, model_name='svm')
  magecs_svm = magecs_svm.merge(y_validation, left_on=['case', 'timepoint'], right_index=True)

  return magecs_mlp, magecs_rf, magecs_svm


def magec_top_important_features_per_model(ranks, models, weights):
  print('----------------------------------------------------------------------')
  for model in models:
    print()
    print(f'MAgEC feature importance by {model} model:')
    grouped_diab = ranks[ranks['Outcome'] == 1][[f'{model}_feat_1', f'{model}_magec_1']].groupby(f'{model}_feat_1').mean()
    grouped_nodiab = ranks[ranks['Outcome'] == 0][[f'{model}_feat_1', f'{model}_magec_1']].groupby(f'{model}_feat_1').mean()
    print('For diabetic individuals')
    print(grouped_diab.sort_values(f'{model}_magec_1'))
    print()
    print('For non-diabetic individuals')
    print(grouped_nodiab.sort_values(f'{model}_magec_1'))
    print('----------------------------------------------------------------------')


def show_magec_feature_importance(features, consensus_auc, x_validation, y_validation):
  res = consensus_auc.merge(x_validation, on='case')
  res = res.merge(y_validation, on='case')

  print('Combined MAgEC feature importance for all models:')
  magec_features = features + ['winner', 'score']
  magec_grouped_diab = res[res['Outcome'] == 1][['winner', 'score']].groupby('winner').mean()
  magec_grouped_nodiab = res[res['Outcome'] == 0][['winner', 'score']].groupby('winner').mean()

  print('For diabetic individuals')
  print(magec_grouped_diab.sort_values('score', ascending=False))
  print()
  print('For non-diabetic individuals')
  print(magec_grouped_nodiab.sort_values('score', ascending=False))
  print('----------------------------------------------------------------------')


def plot_top_confident_cases(features, consensus_auc, x_validation, y_validation):
  res = consensus_auc.merge(x_validation, on='case')
  res = res.merge(y_validation, on='case')

  fig, ax = plt.subplots(1,2,figsize=(10,6))
  ((res.loc[res['Outcome']==0].sort_values('score',ascending=False).head(5) - res.mean())/res.std())[features].plot.bar(ax=ax[0],legend=False,title='confident and incorrect')
  ((res.loc[res['Outcome']==1].sort_values('score',ascending=False).head(5) - res.mean())/res.std())[features].plot.bar(ax=ax[1],title='confident and correct')
  plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
  plt.show()


def run_magec(diabetes_fpath, cols_to_corr_new_feature=None, models=('mlp', 'rf', 'svm')):
  using_new_vars = False
  if cols_to_corr_new_feature and cols_to_corr_new_feature[0] is not None:
    using_new_vars = True

  pima, x_train, x_validation, stsc, x_train_p, x_validation_p, y_train_p, y_validation_p = pm.pima_data(diabetes_fpath, cols_to_corr_new_feature=cols_to_corr_new_feature)
  get_data_importance(pima, x_train_p, y_train_p)

  models_dict = pm.pima_models(x_train_p, y_train_p)

  features = ['BloodPressure', 'BMI', 'Glucose', 'Insulin', 'SkinThickness']
  if using_new_vars is True:
    features.append('GoodRandVar')
    features.append('BadRandVar')

  magecs_mlp, magecs_rf, magecs_svm = generate_mlp_rf_svm_magecs(x_validation_p, y_validation_p, models_dict)
  joined = magec_models(magecs_mlp, magecs_rf, magecs_svm, Xdata=x_validation_p, Ydata=y_validation_p, features=features)

  weights=dict.fromkeys(models, None)
  ranks = magec_rank(joined, models=models, rank=len(features), features=features)
  magec_top_important_features_per_model(ranks, models, weights)

  consensus_auc = magec_consensus(ranks, use_weights=True, models=models, weights=weights)
  show_magec_feature_importance(features, consensus_auc, x_validation_p, y_validation_p)
  plot_top_confident_cases(features, consensus_auc, x_validation_p, y_validation_p)
