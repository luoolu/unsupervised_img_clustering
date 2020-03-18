# -*- coding: utf-8 -*-
# @Time : 2020/3/17 下午4:27
# @Author : LuoLu
# @FileName: cluster_result_feed2folder.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import csv
import glob
import os
from shutil import copyfile

result_path = '/home/luolu/PycharmProjects/unsupervised_img_clustering/result_cluster/'
src_img_path = '/home/luolu/PycharmProjects/unsupervised_img_clustering/result'

with open('Sample_submission.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    header = next(csv_reader)
    print(header)
    print(header[0])
    print(header[1])
    print(len(header))
    # create folder
    for item in range(len(header)):
        try:
            # Create target Directory
            os.mkdir(result_path + str(header[item]))
            print("Directory ", str(header[item]), " Created ")
        except FileExistsError:
            print("Directory ", str(header[item]), " already exists")
    # rest = [row for row in csv_reader]
    # print(rest)g
    # print(glob.glob(src_img_path))

    for row in csv_reader:
        #     print(row[0].replace("'", ""))
        # print(row[0])
        for i in range(len(header)):
            for fname in os.listdir(src_img_path):
                if fname.__eq__(row[i].replace("'", "") + '.png'):
                    # print(fname)
                    #                     copy it to result    copyfile(src, dst)
                    #                     print(src_img_path + '/' + fname)
                    #                     print(result_path + str(header[i]) + '/' + fname)
                    copyfile(src_img_path + '/' + fname, result_path + str(header[i]) + '/' + fname)
