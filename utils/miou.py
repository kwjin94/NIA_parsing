import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
from utils.transforms import transform_parsing
from tqdm import tqdm
from sklearn.metrics import f1_score
import pandas as pd
from pandas import DataFrame

LABELS = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
# LABELS = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette
def make_df_save_csv(confution_matrix, labels,name):
    CM = pd.DataFrame(confution_matrix)
    CM.set_axis(labels, axis='index', inplace=True)
    CM.set_axis(labels, axis='columns', inplace=True)
    # print(CM)

    CM.to_csv('./confusion_matrix/'+name+'.csv')

    return CM
def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

def fast_histogram(a, b, na, nb):

    '''
    fast histogram calculation
    ---
    * a, b: non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]
    * a = gt, b = pred, na = len(gt_label_names), nb = len(pred_label_names)
    * 여기서는 np.bincount 사용해서 계산하는데 scikit-learn에서 제공하는 confusion_matrix 함수 사용하면 똑같음.
    * BUT 함수 사용시 시간이 더 오래걸려서 안쓰는 듯.
    '''

    assert a.shape == b.shape # 비교 shape 다르면 안됨.
    # np.all 조건은 하나라도 안맞으면 False이다.
    # a 또는 b 픽셀중 음수 없어야하고, a,b 픽셀 중 na,nb(len(class)=19) 보다 큰 값이 없어야함.
    assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))

    # nb*a.reshape([-1]).astype(int) 이렇게하면 값에 19곱해주는데 괜찮나.
    # minlength = 19*19 인데 이렇게 할 필요있나.
    # --> 이게 confusino matrix이다. 대각 성분이 각 라벨의 정답 수 이다.
    ''' 
    reshape(na,nb)로 라벨 수 만큼의 정방행렬을 만들고, 
    '''
    hist = np.bincount(
        nb * a.reshape([-1]).astype(int) + b.reshape([-1]).astype(int),
        minlength=na * nb).reshape(na, nb)
    assert np.sum(hist) == a.size
    # hist는 대각성분이 각 class 별 정답 갯수이다.
    return hist


def _read_names(file_name):
    label_names = []
    for name in open(file_name, 'r'):
        # .strip() 앞뒤 공백제거
        name = name.strip()
        if len(name) > 0:
            label_names.append(name)
    return label_names


def _merge(*list_pairs):
    a = []
    b = []
    for al, bl in list_pairs:
        a += al
        b += bl
    return a, b


def compute_mean_ioU(preds, scales, centers, num_classes, datadir, input_size=[473, 473], dataset='test', reverse=False, label=None,list_image=None):
    # print(datadir, dataset)
    # print('after compute', list_image[:20])

    file_list_name = os.path.join(datadir, dataset + '_id.txt')
    file_list_name_train = os.path.join(datadir, 'train_id.txt')
    file_list_name_celebA = os.path.join(datadir, 'test_celebA_id.txt')
    if dataset == 'train':
    # if list_image is not None:
        file_list_name = file_list_name_train
    elif dataset == 'test_celebA':
        file_list_name = file_list_name_celebA
    else:
        file_list_name = file_list_name
    # print(file_list_name[:20])   
    
    val_id = [line.split()[0] for line in open(file_list_name).readlines()]
    # print('a',val_id[:10])
    # train_id = [line.split()[0] for line in open(file_list_name_train).readlines()]
    # print('original_gt_list',val_id[:20])
    confusion_matrix = np.zeros((num_classes, num_classes))

    ## label name 
    # label_names_file = os.path.join(datadir,  dataset + '_id.txt')
    label_names_file = os.path.join(datadir, 'test_labels.txt')
    # class 명 ex) background, skin, ... 등을 불러와서 리스트로 저장.
    gt_label_names = pred_label_names = _read_names(label_names_file)

    # assert gt_label_names[0] == pred_label_names[0] == 'bg'

    hists = []
    # print('dataset',dataset)

    if dataset == 'train' or dataset == 'test_celebA':
    # if list_image is not None:
        # print('list_train_True')
        val_id = list_image
        # print('kw_train_list', list_image[:20], '\n', 'len', len(list_image))
    # print('b',val_id[:10])
    max=0
    for i, im_name in enumerate(val_id):
        # if i<5:
            # print('im_name',im_name)
        if dataset == 'test_celebA':
            gt_path = os.path.join(datadir, dataset + '/labels/' + im_name + '.png')
            # if i==0:
                # print('gt_path', gt_path)
        else:
            gt_path = os.path.join(datadir, dataset + '/labels/' + im_name + '.grayscale.png')

        # proj = np.load(os.path.join(datadir, 'project', im_name + '.npy'))
        
        '''
        cv2.IMREAD_GRAYSCALE 로 불러왔기 때문에 dimension=1이다.
        따라서 gt.shape = (512,512)
        gt는 512x512 = 216,144개의 픽셀의 값으로 구성되어있다.
        즉 gt 라벨 이미지 픽셀이 가리키는 값 216,144개로 구성되어있다.
        '''
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if i < 5:
            cv2.imwrite('./NIA_image_test/gt_image/gt_'+ str(im_name) + dataset +'.png', gt)
        h, w = gt.shape

        # print('pred',preds,'\n','shape',len(preds[0]))

        pred_out = preds[i]
        # print(type(pred_out))

        # if i < 5:
        #     print(gt_path)

        s = scales[i]
        c = centers[i]
        # print(s,c)
        # 이미 crop된거라 굳이 안해도되지않나
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        # output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        # output_im.show()
        if reverse: # False
            pred = cv2.warpAffine(pred, proj, (gt.shape[1], gt.shape[0]), borderValue=0, flags = cv2.INTER_NEAREST)

        # gt to numpy
        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)
        ignore_index = gt != 255
        if np.max(gt)>max:
            max=np.max(gt)
        # num_255+= 
        # ignore_index를 통해서 gt값이 255인 경우를 제외하도록 한다.
        # 이때 512x512에서 flatten된 형태인 (262144,) 형태로 바뀐다. pred도 동일.
        gt = gt[ignore_index]
        # 동일한 조건으로 동일한 크기가 된다.
        pred = pred[ignore_index]
        if len(gt)!=262144:
            # print('len changed')
            # print(len(gt))
            break
        hist = fast_histogram(gt, pred,
                              len(gt_label_names), len(pred_label_names))
        # print('fast_histogram',hist)

        # 대각성분이 정답인 confusion matrix를 image 별로
        # hists에 append 한다.
        hists.append(hist)

        # confusion_matrix는 대각 성분은 정답을 맞게 예측한 값 즉 true positive
        # 이미지 한장당 나오는 confusion matrix를 다 더한다.
        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)
        # print('confusion_matrix', confusion_matrix)
    # hists에는 hist가 append되어있고, 이걸 stack으로 쌓은 후 sum을 해주면 전체
    # 성분의 합이 나온다.   
    hist_sum = np.sum(np.stack(hists, axis=0), axis=0)
    # print(hist_sum)
    # print(hists)
    # print('len hist',len(hists))
    eval_names = dict()
    for label_name in gt_label_names:
        gt_ind = gt_label_names.index(label_name)
        pred_ind = pred_label_names.index(label_name)
        eval_names[label_name] = ([gt_ind], [pred_ind])
    if 'l_eye' in eval_names and 'r_eye' in eval_names:
        eval_names['eyes'] = _merge(eval_names['l_eye'], eval_names['r_eye'])
    if 'l_brow' in eval_names and 'r_brow' in eval_names:
        eval_names['brows'] = _merge(eval_names['l_brow'], eval_names['r_brow'])
    if 'u_lip' in eval_names and 'mouth' in eval_names and 'l_lip' in eval_names:
        eval_names['mouths'] = _merge(
            eval_names['u_lip'], eval_names['mouth'], eval_names['l_lip'])
    if 'eyes' in eval_names and 'brows' in eval_names and 'nose' in eval_names and 'mouth' in eval_names:
        eval_names['overall'] = _merge(
            eval_names['eyes'], eval_names['brows'], eval_names['nose'], eval_names['mouth'])
    # print('eval_names',eval_names)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    ## .sum(0) : 열의 합
    ## .sum(1) : 행의 합
    # print(pos)
    tp = np.diag(confusion_matrix)
    CM = make_df_save_csv(confusion_matrix, gt_label_names, dataset)
    
    # pixel_accuracy는 전체픽셀 중 진짜로 예측된 값의 비율
    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    # mean_accuracy는 클래스별 accuracy의 평균
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    # print('Pixel accuracy: %f \n' % pixel_accuracy)
    # print('Mean accuracy: %f \n' % mean_accuracy)
    # print('Mean IoU: %f \n' % mean_IoU)
    mIoU_value = []
    f1_value = []
    ## mean f1과 best 비교 위해서
    mf1_value = []
    mkwf1_value=[]
    kwf1_value=[]
    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        mIoU_value.append((label, iou))
    idx_=0
    # print('each acc',(tp / np.maximum(1.0, pos)))
    for eval_name, (gt_inds, pred_inds) in (eval_names.items()):
        idx_+=1
        #A 는 실제 gt_inds class의 합,B는 gt_inds로 예측된 값들의 합
        #A는 recall에 분모로, B는 precision의 분모로 사용
        A = hist_sum[gt_inds, :].sum()
        B = hist_sum[:, pred_inds].sum()
        recall = hist_sum[gt_inds,gt_inds]/hist_sum[gt_inds, :].sum()
        precision = hist_sum[gt_inds,gt_inds]/hist_sum[:, pred_inds].sum()
        intersected = hist_sum[gt_inds, :][:, pred_inds].sum()
        f1 = 2 * int(intersected) / (A + B)
        # f1_kw = 2*(A*B)/(A+B)
        f1_kw = 2*(1/((1/recall)+(1/precision)))
        # print(f'f1_{eval_name}={f1}')
        
        if eval_name in gt_label_names[:]:
            mf1_value.append(f1)
            mkwf1_value.append(f1_kw)
        #if len(LABELS) >= idx_:
        #    f1_value.append((LABELS[idx_], f1))
            f1_value.append((eval_name, f1))
            kwf1_value.append((eval_name,f1_kw))
    mIoU_value.append(('Pixel accuracy', pixel_accuracy))
    mIoU_value.append(('Mean accuracy', mean_accuracy))
    mIoU_value.append(('Mean IoU', mean_IoU))
    mIoU_value = OrderedDict(mIoU_value)
    
    
    f1_value.append(('Mean_F1', np.array(mf1_value).mean()))
    f1_value = OrderedDict(f1_value)

    kwf1_value.append(('kw_Mean_F1', np.array(mkwf1_value).mean()))
    # kwf1_value = OrderedDict(kwf1_value)

    # kwf1_value와 같음
    return mIoU_value, f1_value, kwf1_value

def write_results(preds, scales, centers, datadir, dataset, result_dir, input_size=[473, 473]):
    palette = get_palette(20)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    json_file = os.path.join(datadir, 'annotations', dataset + '.json')
    with open(json_file) as data_file:
        data_list = json.load(data_file)
        data_list = data_list['root']
    for item, pred_out, s, c in zip(data_list, preds, scales, centers):
        im_name = item['im_name']
        w = item['img_width']
        h = item['img_height']
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        #pred = pred_out
        save_path = os.path.join(result_dir, im_name[:-4]+'.png')

        output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        output_im.putpalette(palette)
        output_im.save(save_path)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV NetworkEv")
    parser.add_argument("--pred-path", type=str, default='',
                        help="Path to predicted segmentation.")
    parser.add_argument("--gt-path", type=str, default='',
                        help="Path to the groundtruth dir.")

    return parser.parse_args()