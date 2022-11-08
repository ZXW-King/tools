#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: module.py
@time: 2022/10/11 下午3:54
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
CONFIG_FILE = ['config.yaml', 'MODULE.yaml', 'MoudleParam.yaml']


def GetConfigFile(path):
    for file_name in CONFIG_FILE:
        file = os.path.join(path, file_name)
        if os.path.exists(file):
            break
    return file