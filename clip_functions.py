import numpy as np
from collections import Counter
from operator import itemgetter
from numpy.linalg import norm
from numpy import dot

states_dict = {
    'Alaska':'ak',
    'Alabama':'al',
    'Arkansas':'ar',
    'Arizona':'az',
    'California':'ca',
    'Colorado':'co',
    'Connecticut':'ct',
    'Delaware':'de',
    'Florida':'fl',
    'Georgia':'ga',
    'Hawaii':'hi',
    'Iowa':'ia',
    'Idaho':'id',
    'Indiana':'in',
    'Illinois':'il',
    'Kansas':'ks',
    'Kentucky':'ky',
    'Louisiana':'la',
    'Maine':'me',
    'Massachusetts':'ma',
    'Maryland':'md',
    'Michigan':'mi',
    'Minnesota':'mn',
    'Missouri':'mo',
    'Mississippi':'ms',
    'Montana':'mt',
    'North Carolina':'nc',
    'Nebraska':'ne',
    'New Hampshire':'nh',
    'New Jersey':'nj',
    'New Mexico':'nm',
    'Nevada':'nv',
    'New York':'ny',
    'North Dakota':'nd',
    'Ohio':'oh',
    'Oklahoma':'ok',
    'Oregon':'or',
    'Pennsylvania':'pa',
    'Rhode Island':'ri',
    'South Carolina':'sc',
    'South Dakota':'sd',
    'Tennessee':'tn',
    'Texas':'tx',
    'Utah':'ut',
    'Virginia':'va',
    'Vermont':'vt',
    'Washington':'wa',
    'West Virginia':'wv',
    'Wisconsin':'wi',
    'Wyoming':'wy',
}

def scaled_dot_product(text_vector,image_array):

    image_array /= norm(image_array, axis=-1, keepdims=True)
    text_vector /= norm(text_vector, keepdims=True)

    similarities = image_array @ text_vector.T    
    sorted_sims = np.argsort(-similarities.squeeze())

    return sorted_sims

def match_best_to_images(best_matches,target_images):
    matched_images = [target_images[i] for i in best_matches]
    return matched_images

def get_population_statistics_rg(ordered_images,population_dataframe):
    sub_df = population_dataframe.loc[ordered_images]
    race_counter = Counter(sub_df['race'].tolist())
    gender_counter = Counter(sub_df['gender'].tolist())
    return race_counter, gender_counter

def balance_on_characteristic(population_df,image_df,characteristic):
    characteristic_list = population_df[characteristic].tolist()
    sub_counts = Counter(characteristic_list)
    min_characteristic, min_count = min(sub_counts.items(), key=itemgetter(1))
    for c in sub_counts.keys():
        if c == min_characteristic:
            continue
        population_df = population_df.drop(population_df.query(f'{characteristic} == \'{c}\'').sample(sub_counts[c]-min_count).index)
    image_df = image_df.loc[population_df.index]
    return population_df, image_df

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))