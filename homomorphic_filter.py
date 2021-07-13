#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: homomorphic_filter.py
@time: 2021/7/8 下午3:14
@desc: 
'''
import sys, os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
from utils.imageprocess import Remap, ReadPara
from utils.file import MkdirSimple, Walk

CONFIG_FILE = 'config.yaml'

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="image file or dir")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir")

    args = parser.parse_args()
    return args

# 越来越暗
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H

        .
    """

    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, image, filter_params, filter='butterworth', H = None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            image: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(image.shape) != 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        image_log = np.log1p(np.array(image, dtype="float"))
        image_fft = np.fft.fft2(image_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = image_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = image_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) != 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        image_fft_filt = self.__apply_filter(I = image_fft, H = H)
        image_filt = np.fft.ifft2(image_fft_filt)
        image = np.exp(np.real(image_filt)) - 1
        return np.uint8(image)

def WriteImage(image, file, output_dir, root_len):
    if output_dir is not None:
        sub_path = file[root_len+1:]
        output_file = os.path.join(output_dir, sub_path)
        MkdirSimple(output_file)
        cv2.imwrite(output_file, image)

# 对比度较大，勉强可用
def homomorphic_filter(src,d0 = 10,r1 = 0.5,rh=2,c=4,h=2.0,l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows,cols = gray.shape

    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M,N = np.meshgrid(np.arange(-cols // 2,cols // 2),np.arange(-rows//2,rows//2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst,0,255))
    return dst


def main():
    args = GetArgs()

    homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)

    if os.path.isfile(args.input):
        root = len(os.path.dirname(args.input))
    else:
        root = len(os.path.dirname(args.input))
        files = Walk(args.input, ["jpg", "png"])
        for f in tqdm(files):
            image = cv2.imread(f)
            image = image[::, ::, 0]
            # img_filtered = homo_filter.filter(image=image, filter_params=[30, 2])
            img_filtered = homomorphic_filter(src=image, d0=5)
            img_show = np.hstack([image, img_filtered])
            cv2.imshow("result", img_show)
            cv2.waitKey(0)



if __name__ == '__main__':
    main()