import pandas as pd
from collections import Counter

#Visual Question Answering
#Define list of groups by self-identified race
groups = ['A','B','L','W']

#Read in output from BLIP for each CFD photo to the question "Is this person an American?"
american_answers = pd.read_csv(f'D:\\blip_american_string.txt',sep=' :: ',header=None,index_col=0)
img_list = american_answers.index.tolist()

#Iterate over groups
for group in groups:
    targets = american_answers.loc[[i for i in img_list if i[4] == group]] #Get answers only for images corresponding to the self-identified race
    counts = Counter(targets[1].tolist()) #Count the answers to the question
    max_ = len(targets.index.tolist()) #Number of images to which BLIP responded
    print(group)
    #Iterate through each response (yes/no) and print the count
    for key, value in counts.items():
        print(f'{group} - {key}: {value}')

#Read in output from BLIP for each CFD photo to the question "What state does this person live in?"
state_answers = pd.read_csv(f'D:\\blip_state_string.txt',sep=' :: ',header=None,index_col=0)
img_list = state_answers.index.tolist()

#Iterate over groups
for group in groups:
    targets = state_answers.loc[[i for i in img_list if i[4] == group]] #Get answers only for images corresponding to the self-identified race
    counts = Counter(targets[1].tolist()) #Count the answers to the question
    max_ = len(targets.index.tolist()) #Number of images to which BLIP responded
    print(group)
    #Iterate through each state response and print the count
    for key, value in counts.items():
        print(f'{group} - {key}: {value}')

#Count occurrences of race label in BLIP captions
#Read in captions
captions = pd.read_csv(f'D:\\blip_caption_string_with_mr.txt',sep='\t',header=None)

#Race words to be detected; qualitative review confirms that these are the primary words used for description
descs = ['black','Black','African','Asian','asian','White','white','Caucasian','Latino','Latina','Hispanic','multiracial','latino','latina','hispanic','african-american','african american','caucasian','asian-american','asian american']

#Image Captioning
#Iterate over groups
count_dict = {}
for p in groups:

    #Take only captions that are associated with the group under review
    l = captions[0].tolist()
    peoples = [i for i in range(len(l)) if p in l[i]]
    caps = captions.loc[peoples]

    #Iterate by top p
    for i in range(0,10):

        sub_ = caps[caps[1].isin([i])]
        #Remove list output
        the_caps = [cap.replace('[\'','').replace('\']','').replace('[\"','').replace('\"]','').lower() for cap in sub_[2].tolist()]
        labeled_ = []

        #Iterate over captions
        for cap in the_caps:
        
            #Iterate over target descriptions (including race)
            for desc in descs:

                if f'{desc} person' in cap:
                    labeled_.append(cap)
                    break
                if f'{desc} man' in cap:
                    labeled_.append(cap)
                    break
                if f'{desc} woman' in cap:
                    labeled_.append(cap)
                    break

        count_dict[f'{p} {i}'] = len(labeled_)

#Total CFD counts
w_c = 183
l_c = 108
b_c = 197
a_c = 109

race_dict = {}

#Iterate over levels of top p
for i in range(0,10):

    a = count_dict[f'AF {i}'] + count_dict[f'AM {i}']
    b = count_dict[f'BF {i}'] + count_dict[f'BM {i}']
    l = count_dict[f'LF {i}'] + count_dict[f'LM {i}']
    w = count_dict[f'WF {i}'] + count_dict[f'WM {i}']

    race_dict[f'A {i}'] = a
    race_dict[f'B {i}'] = b
    race_dict[f'L {i}'] = l
    race_dict[f'W {i}'] = w

    #Percentage labeled based on race at each level of top p
    print(f'A {i}: {a/a_c}')
    print(f'B {i}: {b/b_c}')
    print(f'L {i}: {l/l_c}')
    print(f'W {i}: {w/w_c}')