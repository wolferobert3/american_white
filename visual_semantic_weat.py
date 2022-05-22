import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats

#WEAT, SC-WEAT to deal with unequal population sizes
def SC_WEAT_unequal(w, A, B):
    w_normed = w / np.linalg.norm(w)
    A_normed = A / np.linalg.norm(A,axis=-1,keepdims=True)
    B_normed = B / np.linalg.norm(B,axis=-1,keepdims=True)

    A_associations = w_normed @ A_normed.T
    B_associations = w_normed @ B_normed.T

    A_mean = np.mean(A_associations)
    A_std = np.std(A_associations)
    A_count = A_associations.shape[0]

    B_mean = np.mean(B_associations)
    B_std = np.std(B_associations)
    B_count = B_associations.shape[0]

    test_statistic = A_mean - B_mean
    effect_size = test_statistic / np.sqrt(((A_count - 1) * A_std ** 2 + (B_count - 1) * B_std ** 2) / (A_count + B_count - 2))

    t_stat, p_value = ttest_ind_from_stats(A_mean,A_std,A_count,B_mean,B_std,B_count,equal_var=False,alternative='less')

    return effect_size, p_value, t_stat

def WEAT_unequal(X, Y, A, B):

    X_normed = X / np.linalg.norm(X,axis=-1,keepdims=True)
    Y_normed = Y / np.linalg.norm(Y,axis=-1,keepdims=True)
    A_normed = A / np.linalg.norm(A,axis=-1,keepdims=True)
    B_normed = B / np.linalg.norm(B,axis=-1,keepdims=True)

    X_A = np.mean((X_normed @ A_normed.T),axis=1)
    X_B = np.mean((X_normed @ B_normed.T),axis=1)
    X_Diff = X_A - X_B
    X_Mean = np.mean(X_Diff)
    X_Std = np.std(X_Diff)
    X_count = X_Diff.shape[0]

    Y_A = np.mean((Y_normed @ A_normed.T),axis=1)
    Y_B = np.mean((Y_normed @ B_normed.T),axis=1)
    Y_Diff = Y_A - Y_B
    Y_Mean = np.mean(Y_Diff)
    Y_Std = np.std(Y_Diff)
    Y_count = Y_Diff.shape[0]

    test_statistic = X_Mean - Y_Mean
    effect_size = test_statistic / np.sqrt(((X_count - 1) * X_Std ** 2 + (Y_count - 1) * Y_Std ** 2) / (X_count + Y_count - 2))

    t_stat, p_value = ttest_ind_from_stats(X_Mean,X_Std,X_count,Y_Mean,Y_Std,Y_count,equal_var=False,alternative='greater')

    return effect_size, p_value, t_stat

#Define model to be considered
#MODEL_ = 'blip_itr_base_coco'
#MODEL_ = 'slip'
MODEL_ = 'clip_base32'
V_WRITE = 'CLIP'

#Read in embedding, descriptive dataframes
population_df = pd.read_csv(f'D:\\cfd\\CFD Version 3.0\\Images\\population_df.csv',index_col=0)
image_df = pd.read_csv(f'D:\\cfd\\embedding_df_{MODEL_}.vec',sep=' ',index_col=0)
language_df = pd.read_csv(f'D:\\cfd\\lang_df_{MODEL_}.vec',sep=' ',index_col=0)

#Create matrices for self-identified race
white_df_ = population_df[population_df['race'].isin(['W'])]
white_df = image_df.loc[white_df_.index]
white = white_df.to_numpy()

black_df_ = population_df[population_df['race'].isin(['B'])]
black_df = image_df.loc[black_df_.index]
black = black_df.to_numpy()

asian_df_ = population_df[population_df['race'].isin(['A'])]
asian_df = image_df.loc[asian_df_.index]
asian = asian_df.to_numpy()

latino_df_ = population_df[population_df['race'].isin(['L'])]
latino_df = image_df.loc[latino_df_.index]
latinx = latino_df.to_numpy()

comparison_list_r = [asian,black,latinx]
desc = ['asian','black','latinx']

#We/They WEAT
in_stimuli = ['we','us','our','ours','ourselves','familiar','similar','here']
out_stimuli = ['they','them','their','theirs','themselves','other','others','there']

in_arr = language_df.loc[in_stimuli].to_numpy()
out_arr = language_df.loc[out_stimuli].to_numpy()

for i in range(len(comparison_list_r)):
    es, p, t = WEAT_unequal(white,comparison_list_r[i],in_arr,out_arr)
    print(f'{desc[i]} Effect Size: {es}, P-Value: {p}, T-Statistic: {t}')

#Survey SC-WEAT
es_master = []
survey_lang = [f'a photo of someone who is patriotic',f'a photo of someone who is an immigrant to America',f'a photo of someone who treats people of all races and backgrounds equally']

#Get effect sizes, p-values, and t-statistics for the defining American characteristics
for lang in survey_lang:

    target_vec = language_df.loc[lang].to_numpy()

    es_list = []

    for comp in comparison_list_r:
        es,p,t = SC_WEAT_unequal(target_vec,white,comp)
        es_list += [es,p,t]

    es_master.append(es_list)

#Make a dataframe, with columns for Effect Size (es), p-value (p), and t-statistic (t)
header_ = ['A_es','A_p','A_t','B_es','B_p','B_t','L_es','L_p','L_t']
es_df = pd.DataFrame(np.array(es_master),index=survey_lang,columns=header_)
print(es_df)
es_df.to_csv(f'D:\\cfd\\effect_sizes_language_{MODEL_}.csv')