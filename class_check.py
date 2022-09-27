import os
from PIL import Image
import numpy as np
import pandas as pd
import tqdm

# class 명 리스트, class 수를 담을 dictionary
class_name_list = ["background","skin","l_brow","r_brow","l_eye","r_eye","eye_g","l_ear","r_ear","ear_r","nose","mouth","u_lip","l_lip","neck","neck_l","cloth","hari","hat"]
class_name = {"background":0,"skin":0,"l_brow":0,"r_brow":0,"l_eye":0,"r_eye":0,"eye_g":0,"l_ear":0,"r_ear":0,"ear_r":0,"nose":0,"mouth":0,"u_lip":0,"l_lip":0,"neck":0,"neck_l":0,"cloth":0,"hari":0,"hat":0}


# class를 check 하고싶은 폴더명 바꿔 주시면 됩니다.
# root_dir = './NIA_8_full/test/labels'
root_dir = './check_test'

# root_dir 하위 폴더의 모든 image list 를 저장하는 list
img_ls = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('png'): 
            img_path = os.path.join(root, file)
            img_ls.append(img_path)

dataframe_name = {"name":img_ls,"background":0,"skin":0,"l_brow":0,"r_brow":0,"l_eye":0,"r_eye":0,"eye_g":0,"l_ear":0,"r_ear":0,"ear_r":0,"nose":0,"mouth":0,"u_lip":0,"l_lip":0,"neck":0,"neck_l":0,"cloth":0,"hari":0,"hat":0}
class_df = pd.DataFrame(dataframe_name)

f = open("./check_NIA.txt", 'w')

for i in tqdm.tqdm(range(len(img_ls))):
    with open(img_ls[i], 'r', encoding='utf-8') as file:
        gray = Image.open(img_ls[i][:-4] + '.png')
        classes = np.unique(gray)

        for cl in classes :
            if cl in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] :
                class_name[str(class_name_list[cl])] += 1
                class_df[str(class_name_list[cl])][i] = 1

    class_df.loc[len(img_ls)]=class_name
    class_df['name'][len(img_ls)]='Total number'
    
print(class_df)
class_df.to_csv('./NIA_parsing_class_check.csv')