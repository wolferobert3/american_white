import numpy as np
from PIL import Image
import pandas as pd
from os import listdir

#Function to return mean brightness
def get_brightness(imag):
    pixels_ = []
    for i in range(1,50):
        for j in range(1,50):
            pix = imag.getpixel((i,j))
            pixels_.append(pix)

    rs,gs,bs = [i[0] for i in pixels_],[i[1] for i in pixels_],[i[2] for i in pixels_]
    brightness = (np.mean(rs) + np.mean(gs) + np.mean(bs)) / 3
    return brightness

#Set source, destination directories
DESTDIR = f'D:\\synthetics'
SOURCEDIR = f'D:\\cfd_age_series\\wikiart-20220301T024059Z-001\\wikiart'
imgs = listdir(SOURCEDIR)

#Crop images to forehead and save crops
for img in imgs:
    race_dir = img[4] #Corresponds to race (W for White)
    num_dir = img[17:-5] #Name of image
    imag = Image.open(f'{SOURCEDIR}\\{img}').convert('RGB')
    cropped = imag.crop((250,175,300,225))
    cropped.save(f'{DESTDIR}\\{race_dir}\\{num_dir}\\{img[:-4]}_cropped.png')

#Get mean brightness for each group
TARGET_ = f'D:\\synthetics\\W'
race_ = 'white'
mean_list = []

#Get brightness for each generated image in a series
for i in range(0,21):
    means,im_names = [],[]
    imgs = listdir(f'{TARGET_}\\{i}')
    for img in imgs:
        imag = Image.open(f'{TARGET_}\\{i}\\{img}')
        means.append(get_brightness(imag))
        im_names.append(img)
    df = pd.DataFrame(means,index=im_names,columns=['brightness'])
    df.to_csv(f'{TARGET_}\{race_}_{i}.csv')
    mean_list.append(np.mean(df['brightness'].to_numpy()))

#Print for LaTex
plot_list = ' '.join([f'({i*10}, {mean_list[i]})' for i in range(len(mean_list))])
print(plot_list)