import numpy as np
import pandas as pd
from clip_functions import states_dict

#Get mean IAT scores for each state from published IAT materials - statewide, White participants, and Black participants
states_ = list(states_dict.keys())
iats = pd.read_csv(f'D:\\raceiatdat.csv')

state_means, white_means, black_means = [],[],[]

for state in states_:

    state_name = states_dict[state]

    state_df = iats[iats['STATE'].isin([state_name.upper()])]
    state_mean = np.mean(state_df['Implicit'])
    state_means.append(state_mean)

    state_white = state_df[state_df['raceomb']==6.0]
    white_mean = np.mean(state_white['Implicit'])
    white_means.append(white_mean)

    state_black = state_df[state_df['raceomb']==5.0]
    black_mean = np.mean(state_black['Implicit'])
    black_means.append(black_mean)

mean_arr = np.array([state_means,white_means,black_means])
mean_df = pd.DataFrame(mean_arr.T,index=states_,columns=['State','White','Black'])
mean_df.to_csv(f'C:\\Users\\wolfe\\Documents\\Research\\iat_means.csv')