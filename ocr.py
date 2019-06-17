# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:49:13 2019

@author: Administrator
"""

#!/usr/bin/env python
# coding=utf8

import numpy as np
import cv2
from numpy.linalg import norm
import os
from opencv_ocr import opencv_ocr

CHAR_SIZE = 20

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):

        self.model = cv2.ml.SVM_create()
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setC(C)
        self.model.setGamma(gamma)

    def train(self, samples, responses):
#        self.model = cv2.ml.SVM_create()
        self.model.train(samples, cv2.ml.ROW_SAMPLE,responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def __scall_and_border_image(src_image, target_size):
    _h, _w = src_image.shape[:2]
    _ratio_target = float(target_size[0]) / float(target_size[1])
    _ratio_image = float(_w) / float(_h)
    _max = max(_ratio_image, _ratio_target)
    if _max == _ratio_image:
        __w = target_size[0]
        __h = int(target_size[0] / _ratio_image)
    else:
        __w = int(target_size[1] * _ratio_image)
        __h = target_size[1]
    _src_image = cv2.resize(src_image, (__w, __h), interpolation=cv2.INTER_LINEAR)

    _left = (target_size[0] - __w) // 2
    _right = target_size[0] - __w - _left
    _top = (target_size[1] - __h) // 2
    _bottom = target_size[1] - __h - _top
    _src_image = cv2.copyMakeBorder(_src_image, _top, _bottom, _left, _right, cv2.BORDER_CONSTANT, value = 0)

    return _src_image, (_left, _top, float(__w)/float(_w), float(__h)/float(_h))

def load_chars(work_path):
    # images_path = os.listdir(work_path)
    digits, labels = [], []
    # for index, _path in enumerate(images_path):
    for fpathe,dirs,fs in os.walk(work_path):
        for f in fs:
            a_path = os.path.join(fpathe,f)
            if a_path[a_path.rfind('.'):] != '.jpg':
                continue
            ch_img = cv2.imread(a_path)
            ch_img, _ = __scall_and_border_image(ch_img, (CHAR_SIZE, CHAR_SIZE))
            ch_img = cv2.cvtColor(ch_img, cv2.COLOR_BGR2GRAY)
            digits.append(ch_img)
            labels.append(ord(a_path.split('.')[-2].split('_')[-1]))
    return np.array(digits), np.array(labels)

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*CHAR_SIZE*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (CHAR_SIZE, CHAR_SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

def hog_test(digit):
    samples = []
    if digit is not None:
        gx = cv2.Sobel(digit, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(digit, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)
    
    
def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print ('error: %.2f %%' % (err*100))

if __name__ == '__main__':
    # 读取训练用的字符图片，这里以名字末尾字符作为标签，进行有监督学习
    digits, labels = load_chars('%s/char_good/'%os.getcwd())
    print ('preprocessing...digits=%d,labels=%d'%(len(digits),len(labels)))
    # shuffle digits 打乱所有字符
    rand = np.random.RandomState(len(digits))
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    digits2 = map(deskew, digits)
    samples = preprocess_hog(digits2)

    if 0:# 取前90%的训练，后10%的测试
        train_n = int(0.9*len(samples))
        digits_train, digits_test = np.split(digits2, [train_n])
        samples_train, samples_test = np.split(samples, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])
    else:# 全部训练+全部测试
        digits_train, digits_test = digits2, digits2
        samples_train, samples_test = samples, samples
        labels_train, labels_test = labels, labels


    print ('training SVM...')
    model = SVM(C=12.00, gamma=5.383)
    model.train(samples_train, labels_train)
    evaluate_model(model, digits_test, samples_test, labels_test)
#    print ('saving SVM as "digits_svm.dat"...')
#    model.save('digits_svm.dat')
    image_path0="E:/Download/OpenCV_OCR-master/train/12.jpg"#必须用'/'
    ch_img =cv2.imread(image_path0)
    if ch_img.shape[0] > ch_img.shape[1]:
        ch_img = np.rot90(ch_img, 1)
    _opencv_ocr = opencv_ocr()       
    img_binary = _opencv_ocr._character_location(ch_img,Precise=False)
    # 分割每一个字符
    img_binary_copy = img_binary.copy()
    contours, hierarchy = cv2.findContours(
        img_binary_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 得到每一个字符的坐标
    _box_shape = []
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        _box_shape.append([x, x+w, y, y+h])
    _box_shape = sorted(_box_shape, key=lambda _box:_box[0])

    # 所有可能字符的list
    _image_list = [img_binary[_box[2]:_box[3],_box[0]:_box[1]] for _box in _box_shape]
    # Dynamic size
    _image_h_list = [_box[3]-_box[2] for _box in _box_shape]
    _median_h = int(np.median(_image_h_list))
    if _median_h < 6:
        _median_h = 6
    # 过滤掉一些不是字符的小图，并且将一些依旧未分割的连接字符分割
#        _opencv_ocr._correct_char_image(_image_list, (_median_h-5, _median_h+5))
    # 将正确的字符图片列表丢到SVM模型里进行识别
    string = _opencv_ocr._svm_classify_string(model, _image_list)
    print ('classify %s is :'%image_path0, string)
