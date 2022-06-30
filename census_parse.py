import pandas as pd
import numpy as np

target_headers = ['NAME','ORIGIN','RACE','POPESTIMATE2019']
race_stats = pd.read_csv('D:\\sc-est2019-alldata5.csv')
demographics_df = race_stats[target_headers]

races_key = {1:'White',2:'Black',4:'Asian'}
origins_keys = {1: 'Not Hispanic', 2: 'Hispanic'}

column_races = ['White','Black','Asian','Hispanic']

states = list(set(race_stats['NAME'].tolist()))
races = list(races_key.keys())
origins = list(origins_keys.keys())
race_stats_state_dict = {state:[] for state in states}
race_stats_state_list = []

for state in states:

    state_df = demographics_df.loc[demographics_df['NAME']==state]
    race_counts = []

    for race in races:
        race_sub_df = state_df.loc[(state_df['RACE']==race) & (state_df['ORIGIN']==1)]
        count = sum(race_sub_df['POPESTIMATE2019'].tolist())
        race_counts.append(count)

    origin_df = state_df.loc[state_df['ORIGIN']==2]
    origin_count = sum(origin_df['POPESTIMATE2019'].tolist())
    race_counts.append(origin_count)

    total_race = sum(race_counts)
    race_proportions = [race_count/total_race for race_count in race_counts]

    race_stats_state_dict[state] = race_proportions
    race_stats_state_list.append(race_proportions)

race_stats_df = pd.DataFrame(index=states,columns=column_races,data=np.array(race_stats_state_list))
race_stats_df.to_csv('D:\\race_stats_df_proportional.csv')