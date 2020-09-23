# -*- coding:utf-8 -*-

"""

    Author:zhuwenwen
    Email：stephenbarrnet@gmail.com
    Date: 2020-09-23

"""

import os
import cv2
import numpy as np
from PIL import Image
from shutil import copy


input_dir = "F:\\zhuwenwen\\Test_cases\\input"
output_dir = 'F:\\zhuwenwen\\Test_cases\\output\\'


# -----------------------图片反色-----------------------------#
def ImgColorConvert(filename):
    count = 0
    im = Image.open(filename)

    img = cv.imread(filename)

    cropImg_lt = img[0:10, 0:10]
    cropImg_lb = img[im.size[1] - 10:im.size[1], 0:10]
    cropImg_rt = img[0:10, im.size[0] - 10:im.size[0]]
    cropImg_rb = img[im.size[1] - 10:im.size[1], im.size[0] - 10:im.size[0]]

    if os.path.exists(tmp_dir) == False:
        os.makedirs(tmp_dir)

    path_temp_1 = tmp_dir + "lt_" + str(count) + ".jpg"
    path_temp_2 = tmp_dir + "lb_" + str(count) + ".jpg"
    path_temp_3 = tmp_dir + "rt_" + str(count) + ".jpg"
    path_temp_4 = tmp_dir + "rb_" + str(count) + ".jpg"

    cv.imwrite(path_temp_1, cropImg_lt)
    cv.imwrite(path_temp_2, cropImg_lb)
    cv.imwrite(path_temp_3, cropImg_rt)
    cv.imwrite(path_temp_4, cropImg_rb)

    count += 1

    list_1 = pixel_rgb_mean(path_temp_1)
    list_2 = pixel_rgb_mean(path_temp_2)
    list_3 = pixel_rgb_mean(path_temp_3)
    list_4 = pixel_rgb_mean(path_temp_4)

    for i in range(3):
        ave_value_1 = np.mean([list_1[i], list_2[i], list_3[i], list_4[i]])
        ave_value_2 = np.mean([list_1[i], list_2[i], list_3[i], list_4[i]])
        ave_value_3 = np.mean([list_1[i], list_2[i], list_3[i], list_4[i]])

    ave_value = np.mean([ave_value_1, ave_value_2, ave_value_3])

    if ave_value > 128:
        image = Image.open(filename)
        # 色域反转并保存
        inverted_image = PIL.ImageOps.invert(image)
        inverted_image.save(filename)

    image_path = filename
    return Image.open(image_path), ave_value


# 计算RGB图片像素均值
def pixel_rgb_mean(img_name_rgb):

    img_size = 224
    sum_r = 0
    sum_g = 0
    sum_b = 0
    count = 0

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (img_size, img_size))
    sum_r = sum_r+img[:, :, 0].mean()
    sum_g = sum_g+img[:, :, 1].mean()
    sum_b = sum_b+img[:, :, 2].mean()
    count = count+1

    sum_r = sum_r/count
    sum_g = sum_g/count
    sum_b = sum_b/count
    img_mean = [sum_r, sum_g, sum_b]
    print(img_mean)
    return img_mean


# 计算灰度图像素均值
def pixel_gray_mean(img_name_gray):
    img = cv2.imread(img_name, 0)
    height, width = img.shape
    size = img.size

    average = 0
    for i in range(height):
        for j in range(width):
            average += img[i][j] / size


# -----------------------汉字转拼音-----------------------------#
def hanzi2pinyin(inputpath, outputPath):
    count = 0
    # 该文件夹下所有的文件（包括文件夹）
    filelist = os.listdir(inputpath)
    # 遍历所有文件
    for files in filelist:
        # 原来的文件路径
        Olddir = os.path.join(inputpath, files)
        # 如果是文件夹则跳过
        if os.path.isdir(Olddir):
            continue
        # 获取文件名
        filename = os.path.splitext(files)[0]
        # 把文件名里的汉字转换成其首字母
        filename1 = lazy_pinyin(filename)
        filenameToStr_1 = ''.join(filename1)

        # 文件扩展名
        filetype = os.path.splitext(files)[1]

        Newdir = os.path.join(outputPath, filenameToStr_1 + filetype)

        for i in os.listdir(outputPath):
            j = outputPath + '\\' + str(i)
            if j == Newdir:
                Newdir = outputPath + '\\' + filenameToStr_1 + str(count) + '.DCM'
                count += 1

        os.rename(Olddir, Newdir)


# -----------------------将dcm文件转换为jpg文件-----------------------------#
jug = []

def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    # frame_num, width, height = img_array.shape
    return img_array


def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    return str(information['PatientName'])


def limited_equalize(img, limit=4.0):
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img_limited_equalized = np.array(img)
    return img_limited_equalized


def dicom_jpeg(path):
    dicompath = path
    a = loadFile(dicompath)
    img = a[0]
    idname = loadFileInformation(dicompath)

    if idname not in jug:
        jug.append(idname)
    else:
        idname = str(idname) + '\t' + str(random.randint(0, 99))
        jug.append(str(idname))
    big = 0
    for p in img:
        if int(max(p)) > big:
            big = max(p)
    print(big)
    img = np.array(img, dtype=np.float32)
    img = img * 255
    img = img / int(big)

    im = np.uint8(img)
    im = limitedEqualize(im)
    jpegpath = 'D:\\zhuwenwen\\Datasets\\JSTdetasets\\JST2\\output\\' + str(idname) + '.jpg'
    cv2.imwrite(jpegpath, im)


# 程序入口
for filename in os.listdir(inputpath):
    savejpg_name = img_dir + filename
    dicom_jpeg(savejpg_name)
    os.remove(savejpg_name)


# -----------------------对图片进行旋转-----------------------------#
def rotate(data, ori='left'):
    if ori == 'left':
        data = list(map(list, zip(*data)))[::-1]
    else:
        data = list(map(list, zip(*data[::-1])))

    data = np.array(data)
    return data


files = os.listdir(inputpath)
for file in files:
    fileIn = input_path + file
    fileOut = outputpath + file
    img = cv2.imread(fileIn)
    #img_left = rotate(img)
    #cv2.imwrite(fileOut, img_left)
    img_right = rotate(img, ori='right')
    cv2.imwrite(fileOut, img_right)


# -------------------按照特征点裁剪指掌骨图片------------------------#
'''
    pass
'''
# 根据单个点的坐标截图
def shot(img, filename, loc_y, loc_x, w, y):
    global count

    img_tmp = img[loc_x - w:loc_x + w, loc_y - y:loc_y + y]

    output_name = output_dir + file_name + "\\" + filename + '_' + str(count) + ".jpg"
    count += 1
    cv2.imwrite(output_name, img_tmp)


# 在图片上面画点
def circle(file_img, file_name, loc_x, loc_y):
    img = cv2.imread(file_img)
    #cv2.circle(img, (loc_x, loc_y), 10, (0, 0, 1), -1)

    output_name = output_dir + file_name + "\\" + file_name + ".jpg"
    cv2.imwrite(output_name, img)

    img = cv2.imread(output_name)

    sp_x = img.shape[0]
    sp_y = img.shape[1]

    global count
    count_list = [3, 6, 8]
    if count not in count_list:
        w = sp_x // 20
        y = sp_y // 20
    else:  # 远节指骨关节
        w = sp_x // 15
        y = sp_y // 15

    shot(img, file_name, loc_x, loc_y, w, y)  # x,y


# 根据四个点的坐标截图
def shot_bak(img, file_name, extre, wide, exter_x, exter_y, loc_1, loc_2, loc_3, loc_4):
    global count

    #img_tmp = img[loc_2-exter_y:loc_1+exter_y, loc_3-exter_x:loc_4+exter_x]
    wide_tmp = int(wide // 2)
    img_tmp = img[loc_1 - int(wide):loc_1 + wide_tmp, loc_3 - exter_x:loc_4 + exter_x]

    output_name = output_dir + file_name + "\\" + file_name + "_" + str(count) + ".jpg"
    count += 1
    cv2.imwrite(output_name, img_tmp)


# 在图片上面画点
def circle_bak(file_img, file_name, wide, loc_1, loc_2, loc_3, loc_4, loc_5, loc_6):
    img = cv2.imread(file_img)

    output_name = output_dir + file_name + "\\" + file_name + ".jpg"
    cv2.imwrite(output_name, img)

    img = cv2.imread(output_name)

    extre = img.shape[1] // 20
    exter_x = img.shape[0] // 35
    exter_y = img.shape[1] // 50

    shot_bak(img, file_name, extre, wide, exter_x, exter_y, loc_1, loc_3, loc_4, loc_6)


# 根据单个点的坐标截图
def shot_bak_1(img, filename, loc_y, loc_x, w, y, w_bak, y_bak):
    global count

    #img_tmp_1 = img[loc_x - w:loc_x + w - w_bak, loc_y - y:loc_y + y]
    #img_tmp_2 = img[loc_x - w + y_bak:loc_x + w, loc_y - y:loc_y + y]

    img_tmp_1 = img[50:60, 50:60]
    img_tmp_2 = img[50:60, 50:60]

    output_name_1 = output_dir + file_name + "\\" + filename + '_' + str(count) + "_1" + ".jpg"
    output_name_2 = output_dir + file_name + "\\" + filename + '_' + str(count) + "_2" + ".jpg"
    count += 1
    cv2.imwrite(output_name_1, img_tmp_1)
    cv2.imwrite(output_name_2, img_tmp_2)


# 在图片上面画点
def circle_bak_1(file_img, file_name, loc_x, loc_y):
    img = cv2.imread(file_img)
    #cv2.circle(img, (loc_x, loc_y), 10, (0, 0, 1), -1)

    output_name = output_dir + file_name + "\\" + file_name + ".jpg"
    cv2.imwrite(output_name, img)

    img = cv2.imread(output_name)

    sp_x = img.shape[0]
    sp_y = img.shape[1]

    global count
    count_list = [3, 6, 8]
    if count not in count_list:
        w = sp_x // 20
        y = sp_y // 20
        w_bak = 0
        y_bak = 0
    else:  # 远节指骨关节
        w = sp_x // 15
        y = sp_y // 15
        w_bak = sp_x // 30
        y_bak = sp_x // 25

    shot_bak_1(img, file_name, loc_x, loc_y, w, y, w_bak, y_bak)  # x,y

# 根据四个点的坐标截图
def shot_bak_2(img, file_name,wide, exter_x, exter_y, loc_1, loc_2, loc_3, loc_4):
    global count

    #img_tmp = img[loc_2 - exter_y:loc_1 + exter_y, loc_3 - exter_x:loc_4 + exter_x]
    wide_tmp = int(wide // 2)
    img_tmp = img[loc_1 - int(wide - 30):loc_1 + wide_tmp, loc_3 - exter_x:loc_4 + exter_x]

    output_name = output_dir + file_name + "\\" + file_name + "_" + str(count) + ".jpg"
    count += 1
    cv2.imwrite(output_name, img_tmp)


# 在图片上面画点
def circle_bak_2(file_img, file_name, wide, loc_1, loc_2, loc_3, loc_4, loc_5, loc_6):
    img = cv2.imread(file_img)

    output_name = output_dir + file_name + "\\" + file_name + ".jpg"
    cv2.imwrite(output_name, img)

    img = cv2.imread(output_name)
    exter_x = img.shape[0] // 35
    exter_y = img.shape[1] // 50

    shot_bak_2(img, file_name, wide, exter_x, exter_y, loc_1, loc_3, loc_4, loc_6)


def read_txt(file_txt, file_img, file_name):
    read_list = [35, 34, 33, 27, 26, 25, 19, 18, 17, 2, 3]
    f = open(file_txt)

    data = f.readlines()

    for i in read_list:
        print(i)
        temp = data[i].split(',')
        temp[1] = temp[1][:-1]
        temp[0] = float(temp[0])
        temp[1] = float(temp[1])

        #
        if i == 2:
            temp_2 = data[2].split(',')
            temp_1 = data[1].split(',')
            temp_0 = data[0].split(',')

            #
            temp_2[1] = temp_2[1][:-1]
            temp_2[0] = float(temp_2[0])
            temp_2[1] = float(temp_2[1])

            temp_1[1] = temp_1[1][:-1]
            temp_1[0] = float(temp_1[0])
            temp_1[1] = float(temp_1[1])

            temp_0[1] = temp_0[1][:-1]
            temp_0[0] = float(temp_0[0])
            temp_0[1] = float(temp_0[1])

            wide = temp_2[1] - temp_0[1]
            circle_bak(file_img, file_name, wide, int(temp_0[0]), int(temp_0[1]), int(temp_1[0]), int(temp_1[1]), int(temp_2[0]), int(temp_2[1]))

        elif i == 3:
            temp_5 = data[5].split(',')
            temp_4 = data[4].split(',')
            temp_3 = data[3].split(',')

            temp_5[1] = temp_5[1][:-1]
            temp_5[0] = float(temp_5[0])
            temp_5[1] = float(temp_5[1])

            temp_4[1] = temp_4[1][:-1]
            temp_4[0] = float(temp_4[0])
            temp_4[1] = float(temp_4[1])

            temp_3[1] = temp_3[1][:-1]
            temp_3[0] = float(temp_3[0])
            temp_3[1] = float(temp_3[1])

            wide = temp_4[1] - temp_3[1]
            circle_bak_2(file_img, file_name, wide, int(temp_3[0]), int(temp_4[0]), int(temp_5[0]), int(temp_3[1]), int(temp_4[1]), int(temp_5[1]))
        elif i == 25:
            circle_bak_1(file_img, file_name, int(temp[1]), int(temp[0]))
        elif i == 33:
            circle_bak_1(file_img, file_name, int(temp[1]), int(temp[0]))
        else:
            circle(file_img, file_name, int(temp[1]), int(temp[0]))

        # 返回点的坐标
        # return int(temp[0]), int(temp[1])


count = 1
files = os.listdir(input_dir)
for file in files:
    print("filename = ", file)
    if '.jpg' in file:
        count = 1
        file_name = file.split('.')[0]
        file_txt = input_dir + "\\" + file.split('.')[0] + '.txt'
        file_img = input_dir + "\\" + file.split('.')[0] + '.jpg'
        dir = output_dir + file_name
        if not os.path.exists(dir):
            os.mkdir(dir)
        read_txt(file_txt, file_img, file_name)


# -----------------移动文件----------------- #
def movefile(inputpath, inputpath_2, outputpath):
    files_ori = os.listdir(inputpath)
    files_des = os.listdir(inputpath_2)

    for file in files_ori:
        file_temp = file
        file = file.split('.')[0] + ".png"
        if file in files_des:
            copy(input1_dir + "\\" + file_temp, outputpath)


# ----------------------------将png转成jpg--------------------------- #
def PNG_JPG(inputpath):
    img = cv2.imread(inputpath, 0)
    w, h = img.shape[::-1]
    infile = inputpath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=70)
            os.remove(inputpath)
        else:
            img.convert('RGB').save(outfile, quality=70)
            os.remove(inputpath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)


