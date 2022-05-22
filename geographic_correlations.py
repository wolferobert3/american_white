import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from clip_functions import scaled_dot_product, match_best_to_images, balance_on_characteristic, get_population_statistics_rg, states_dict
import plotly.express as px

#Define model and write model setting
MODEL_ = 'clip_base32'
V_WRITE = 'CLIP'

#Read in previously gathered embeddings
population_df = pd.read_csv(f'D:\\cfd\\CFD Version 3.0\\Images\\population_df.csv',index_col=0)
image_df = pd.read_csv(f'D:\\cfd\\embedding_df_{MODEL_}.vec',sep=' ',index_col=0)
language_df = pd.read_csv(f'D:\\cfd\\lang_df_{MODEL_}.vec',sep=' ',index_col=0)

#CLIP Returns Analysis
ceiling_ = 108
race_stat_df = pd.read_csv(f'D:\\race_stats_df_proportional.csv',index_col=0)
races = race_stat_df.columns.tolist()
states = race_stat_df.index.tolist()

iat_means = pd.read_csv(f'D:\\iat_means.csv',index_col=0)

all_dfs, iats, weats = [],[],[]

#Get IAT stats in correct order
iats = [iat_means.loc[state]['State'] for state in states]
iats_black = [iat_means.loc[state]['Black'] for state in states]
iats_white = [iat_means.loc[state]['White'] for state in states]

#Balance dataframe 1,000 times and get most similar images for each state
for _ in range(1000):

    pop_df, veridicality_images = balance_on_characteristic(population_df,image_df,'race')
    image_features = veridicality_images.to_numpy()
    image_list = veridicality_images.index.tolist()

    full_race_breakdown = []

    for state in states:

        if state == 'District of Columbia':
            text_features = language_df.loc[f'a photo of someone who lives in Washington, D.C.'].to_numpy()
        else:
            text_features = language_df.loc[f'a photo of someone who lives in the state of {state}'].to_numpy()

        similarities = scaled_dot_product(text_features,image_features)[:ceiling_]
        best_matches = match_best_to_images(similarities,image_list)
        race_stats, gender_stats = get_population_statistics_rg(best_matches,population_df)

        state_race_breakdown = []
        
        for race in ['W','B','A','L']:
            race_count = race_stats[race]
            state_race_breakdown.append(race_count)

        full_race_breakdown.append(state_race_breakdown)

    clip_race_df = pd.DataFrame(index=states,columns=['W','B','A','L'],data=np.array(full_race_breakdown))
    all_dfs.append(clip_race_df)

#Get mean returns by state over 1,000 runs
all_W = np.expand_dims(np.mean(np.array([df['W'].to_numpy().squeeze() for df in all_dfs]),axis=0),1)
all_B = np.expand_dims(np.mean(np.array([df['B'].to_numpy().squeeze() for df in all_dfs]),axis=0),1)
all_A = np.expand_dims(np.mean(np.array([df['A'].to_numpy().squeeze() for df in all_dfs]),axis=0),1)
all_L = np.expand_dims(np.mean(np.array([df['L'].to_numpy().squeeze() for df in all_dfs]),axis=0),1)

#Create final array & dataframe of returned images for the model; Hispanic label used for comparison with census data
final_arr = np.concatenate((all_W,all_B,all_A,all_L),axis=1)
clip_race_df = pd.DataFrame(final_arr,index=states,columns = ['White','Black','Asian','Hispanic'])

#Get CLIP images returned, census population for comparison
clip_race_df['state'] = [states_dict[state].upper() for state in clip_race_df.index.tolist()]
clip_race_df['Pct of CLIP Images Returned'] = clip_race_df['Black'].tolist()
clip_race_df['CLIP % Black Images'] = clip_race_df['Black'].tolist()

race_stat_df['state'] = [states_dict[state].upper() for state in race_stat_df.index.tolist()]
race_stat_df['Census Population Pct'] = [i*100 for i in race_stat_df['Black'].tolist()]

#Get correlation between IATs and census
print('Correlation of IAT, IAT Black, IAT White for Census')
print(pearsonr(iats,race_stat_df['Black'].tolist()))
print(pearsonr(iats_black,race_stat_df['Black'].tolist()))
print(pearsonr(iats_white,race_stat_df['Black'].tolist()))

#Get correlation between IATs and model image returns
print('Correlation of IAT, IAT Black, IAT White for Model Returns')
print(pearsonr(iats,clip_race_df['Black'].tolist()))
print(pearsonr(iats_black,clip_race_df['Black'].tolist()))
print(pearsonr(iats_white,clip_race_df['Black'].tolist()))

#Lists for census ground truth (gt) and model returns
all_race_gt,all_race_clip,gt_grid,clip_grid = [],[],[],[]

for idx, race in enumerate(races):
    ground_truth = race_stat_df[race].tolist()
    clip_return = clip_race_df[race].tolist()
    clip_return = [i/ceiling_ for i in clip_return]

    #Add to grid for regressing
    if idx == 0:
        all_race_gt = list(ground_truth)
        all_race_clip = list(clip_return)
        gt_grid = [ground_truth]
        clip_grid = [clip_return]
    else:
        all_race_gt.extend(list(ground_truth))
        all_race_clip.extend(list(clip_return))
        gt_grid.append(ground_truth)
        clip_grid.append(clip_return)

    #Get correlation of model return and census statistics
    print(race)
    print(pearsonr(ground_truth,clip_return))

#Get R2 coefficient for model returns vs. census statistics
gt_arr = np.array(gt_grid)
clip_arr = np.array(clip_grid)
lin_arr = LinearRegression().fit(clip_arr.T,gt_arr.T)
r2 = lin_arr.score(clip_arr.T,gt_arr.T)
print(f'Model R2 Coefficient: {r2}')

#Visualize in choropleths for paper
fig = px.choropleth(clip_race_df,  # Input Pandas DataFrame
                    color_continuous_scale="Viridis",
                    range_color=[0,65], #Upper bound is max CLIP return
                    locations="state",  # DataFrame column with locations
                    color="Pct of CLIP Images Returned",  # DataFrame column with color values
                    hover_name="state", # DataFrame column hover info
                    locationmode = 'USA-states') # Set to plot as US States
fig.update_layout(
    title_text = 'State Rankings', # Create a Title
    geo_scope='usa',  # Plot only the USA instead of globe
    coloraxis_colorbar_x=-0.15
)
fig.show()

fig = px.choropleth(race_stat_df,  # Input Pandas DataFrame
                    color_continuous_scale="Viridis",
                    range_color=[0,65], #Same as CLIP scale
                    locations="state",  # DataFrame column with locations
                    color="Census Population Pct",  # DataFrame column with color values
                    hover_name="state", # DataFrame column hover info
                    locationmode = 'USA-states') # Set to plot as US States
fig.update_layout(
    title_text = 'State Rankings', # Create a Title
    geo_scope='usa',  # Plot only the USA instead of globe
    coloraxis_colorbar_x=-0.15
)
fig.show()

fig = px.choropleth(iat_means,  # Input Pandas DataFrame
                    color_continuous_scale="Viridis",
                    range_color=[.3,.48], #Min-max scale
                    locations="state",  # DataFrame column with locations
                    color="White",  # DataFrame column with color values
                    hover_name="state", # DataFrame column hover info
                    locationmode = 'USA-states') # Set to plot as US States
fig.update_layout(
    title_text = 'Mean IAT (White) by State', # Create a Title
    geo_scope='usa',  # Plot only the USA instead of globe
    coloraxis_colorbar_x=-0.15
)
fig.show()