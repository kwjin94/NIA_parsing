import os
import json 
from PIL import Image
import numpy as np
import cv2


root_dir = './NIA_8/'

img_ls = []
json_ls = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if not file.endswith('grayscale.png')and file.endswith('png'): 
            img_path = os.path.join(root, file)
            img_ls.append(img_path)
        if file.endswith('.json'):
            json_path = os.path.join(root, file)
            json_ls.append(json_path)
print(img_ls)
f = open("./check_NIA8.txt", 'w')
count = 0
point1 =0
point2= 0
point3 =0
bbox1 = 0
error = []

print('asdf')

print('len',len(json_ls))

data_loc = 1
cls = {"0 class":0,"1 class":0,"2 class":0,"3 class":0,"4 class":0,"5 class":0,"6 class":0,"7 class":0,"8 class":0,"9 class":0,"10 class":0,"11 class":0,"12 class":0,"13 class":0,"14 class":0,"15 class":0,"16 class":0,"17 class":0,"18 class":0}
for i in range(len(json_ls)):
    print(i)
    with open(json_ls[i], 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(data[data_loc])
        print('len_data',len(data))
        if len(data) != 2 :
            f.write('bbox 개수 확인 필요 : '+ json_ls[i] + '\n')
        else :
            bbox1 += 1

        if len(data[data_loc]['points']) > 106 :
            f.write(" points > 106 데이터 : " + str(json_ls[i]) +'\n')
            point1 += 1
        
        elif len(data[data_loc]['points']) == 106 :
            #f.write(" points = 106 데이터 : " + str(json_ls[i]) +'\n')
            point2 += 1

        elif len(data[data_loc]['points']) < 106 :
            f.write(" points < 106 데이터 : " + str(json_ls[i]) +'\n')
            point3 += 1



        if (data[data_loc]['box']['w'] >= 300) and (data[data_loc]['box']['h'] >= 300) :
            count += 1
        else : 
            error.append("box size < 300 데이터 : " + str(json_ls[i]) +'\n')
            

    
        gray = Image.open(img_ls[i][:-4] + 'grayscale.png')
        classes = np.unique(gray)
        for i in classes :
            if i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] :
                cls[str(i)+" class"] += 1
            else :
                continue
                cls['wrong class'] += 1
                #f.write("오라벨링 데이터 " + str(json_ls[i][:-4] + 'grayscale.png\n'))
                #f.write('wrong class' + str(i) +'\n')
f.write("전체 데이터 수량: "+ str(len(json_ls)) +'\n')
f.write('\n\n')
f.write('points > 106 데이터 수량 : ' +str(point1)+ "(" + str(point1/len(json_ls) * 100) + ")" + '\n')
f.write('points = 106 데이터 수량 : ' +str(point2)+ "(" + str(point2/len(json_ls) * 100) + ")" + '\n')
f.write('points < 106 데이터 수량 : ' +str(point3)+ "(" + str(point3/len(json_ls) * 100) + ")" + '\n')
f.write('\n\n')
for i in error :
    f.write(i)
f.write('\n')
f.write('bbox = 1 데이터 : ' +str(bbox1) +"(" + str(bbox1/len(json_ls) * 100) + ")" + '\n')
f.write('bbox =! 1 데이터 : ' +str(len(json_ls) -bbox1) +"(" + str((len(json_ls)-bbox1)/len(json_ls) * 100) + ")" + '\n\n' )
f.write('300x300 이상 데이터 수량: '+ str(count) +"(" + str(count/len(json_ls) * 100) + ")" + '\n\n')


for i in range(len(cls)) :
    if i == 19 :
        continue
        i = 'wrong'
    f.write(str(i) +" class 퍼센트 : " + str(cls[str(i) +" class"] / len(json_ls)) +'\n')
f.write('\n')
f.write("해당 클래스가 들어간 이미지 수량 :" +str(cls) +'\n\n\n\n')

f.close()
