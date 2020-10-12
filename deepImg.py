# -*- coding:utf-8 -*-

"""

    Author:zhuwenwen
    Email：stephenbarrnet@gmail.com
    Date: 2020-09-23

"""

import os
import cv2
import PIL
import numpy as np
from PIL import Image
from shutil import copy
from math import *
import imutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches

input_dir = "F:\\003__zhuwenwen\\Test_cases\\input"
output_dir = 'F:\\003__zhuwenwen\\Test_cases\\output\\'

'''
    图片反色
'''


def ImgColorConvert(filename):
    count = 0
    im = Image.open(filename)

    img = cv2.imread(filename)

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

    cv2.imwrite(path_temp_1, cropImg_lt)
    cv2.imwrite(path_temp_2, cropImg_lb)
    cv2.imwrite(path_temp_3, cropImg_rt)
    cv2.imwrite(path_temp_4, cropImg_rb)

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

    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    sum_r = sum_r + img[:, :, 0].mean()
    sum_g = sum_g + img[:, :, 1].mean()
    sum_b = sum_b + img[:, :, 2].mean()
    count = count + 1

    sum_r = sum_r / count
    sum_g = sum_g / count
    sum_b = sum_b / count
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


'''
    汉字转拼音
'''


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


'''
    将dcm文件转换为jpg文件
'''
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
'''
for filename in os.listdir(inputpath):
    savejpg_name = img_dir + filename
    dicom_jpeg(savejpg_name)
    os.remove(savejpg_name)
'''


def rotate(data, ori='left'):
    '''
        说明 ：对图片进行左右90旋转
        data ： 图片矩阵
        ori='left' ： 默认向左旋转
    '''
    if ori == 'left':
        data = list(map(list, zip(*data)))[::-1]
    else:
        data = list(map(list, zip(*data[::-1])))

    data = np.array(data)
    return data


'''
files = os.listdir(input_dir)
for file in files:
    fileIn = input_dir + file
    fileOut = input_dir + file
    img = cv2.imread(fileIn)
    # img_left = rotate(img)
    # cv2.imwrite(fileOut, img_left)
    img_right = rotate(img, ori='right')
    cv2.imwrite(fileOut, img_right)
'''

'''
    按照特征点裁剪指掌骨图片
'''


# 根据单个点的坐标截图
def shot(img, filename, loc_y, loc_x, w, y):
    global count

    img_tmp = img[loc_x - w:loc_x + w, loc_y - y:loc_y + y]

    output_name = output_dir + file_name + "\\" + filename + '_' + str(count) + ".jpg"
    count += 1
    cv2.imwrite(output_name, img_tmp)


# 根据单个点的坐标截图
def shot_rotate(img, loc_y, loc_x, w, y):
    global count

    img_tmp = img[loc_x - w:loc_x + w, loc_y - y:loc_y + y]
    output_name = dir_outname
    count += 1
    cv2.imwrite(output_name, img_tmp)


# 在图片上面画点
def circle(file_img, file_name, loc_x, loc_y):
    img = cv2.imread(file_img)
    # cv2.circle(img, (loc_x, loc_y), 10, (0, 0, 1), -1)

    # output_name = output_dir + file_name + "\\" + file_name + ".jpg"
    output_name = "F:\\003  zhuwenwen\\testoutput\\test.jpg"
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


# 在图片上面画点
def circle_rotate(file_name, rotate_image, loc_x, loc_y):
    # cv2.circle(img, (loc_x, loc_y), 10, (0, 0, 1), -1)
    # output_name = output_dir + file_name + "\\" + file_name + ".jpg"

    sp_x = rotate_image.shape[1]
    sp_y = rotate_image.shape[0]

    global count
    count_list = [3, 6, 8]
    '''
    if count not in count_list:
        w = sp_x // 20
        y = sp_y // 20
    else:  # 远节指骨关节
    '''
    w = sp_x // 23
    y = sp_y // 20

    shot_rotate(rotate_image, loc_y, loc_x, w, y)  # x,y


# 根据四个点的坐标截图
def shot_bak(img, file_name, extre, wide, exter_x, exter_y, loc_1, loc_2, loc_3, loc_4):
    global count

    # img_tmp = img[loc_2-exter_y:loc_1+exter_y, loc_3-exter_x:loc_4+exter_x]
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

    # img_tmp_1 = img[loc_x - w:loc_x + w - w_bak, loc_y - y:loc_y + y]
    # img_tmp_2 = img[loc_x - w + y_bak:loc_x + w, loc_y - y:loc_y + y]

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
    # cv2.circle(img, (loc_x, loc_y), 10, (0, 0, 1), -1)

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
def shot_bak_2(img, file_name, wide, exter_x, exter_y, loc_1, loc_2, loc_3, loc_4):
    global count

    # img_tmp = img[loc_2 - exter_y:loc_1 + exter_y, loc_3 - exter_x:loc_4 + exter_x]
    wide_tmp = int(wide // 2)
    img_tmp = img[loc_1 - int(wide - 30):loc_1 + wide_tmp, loc_3 - exter_x:loc_4 + exter_x]

    # output_name = output_dir + file_name + "\\" + file_name + "_" + str(count) + ".jpg"
    output_name = "F:\\003  zhuwenwen\\testoutput\\test.jpg"
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
    '''
        说明 ：读取特征点坐标，跟模具坐标裁剪部分区域图片
        file_txt ： 特征点坐标txt文件
        file_img ： 输入图片路径
        file_name ： 图片所在文件夹名称
    '''
    read_list = [35, 34, 33, 27, 26, 25, 19, 18, 17, 2, 3]
    f = open(file_txt)
    data = f.readlines()

    for i in read_list:
        print(i)
        temp = data[i].split(',')
        temp[1] = temp[1][:-1]
        temp[0] = float(temp[0])
        temp[1] = float(temp[1])

        if i == 2:
            temp_2 = data[2].split(',')
            temp_1 = data[1].split(',')
            temp_0 = data[0].split(',')

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
            circle_bak(file_img, file_name, wide, int(temp_0[0]), int(temp_0[1]), int(temp_1[0]), int(temp_1[1]),
                       int(temp_2[0]), int(temp_2[1]))

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
            circle_bak_2(file_img, file_name, wide, int(temp_3[0]), int(temp_4[0]), int(temp_5[0]), int(temp_3[1]),
                         int(temp_4[1]), int(temp_5[1]))
        elif i == 25:
            circle_bak_1(file_img, file_name, int(temp[1]), int(temp[0]))
        elif i == 33:
            circle_bak_1(file_img, file_name, int(temp[1]), int(temp[0]))
        else:
            circle(file_img, file_name, int(temp[1]), int(temp[0]))


def movefile(inputpath, inputpath_2, outputpath):
    '''
        说明 ：将inputpath与inputpath_2中共有的图片保存到outputpath中
        inputpath ： 待移动的图片文件夹
        inputpath_2 ： 待比对的文件夹
        outputpath ： 保存图片文件夹
    '''
    files_ori = os.listdir(inputpath)
    files_des = os.listdir(inputpath_2)

    for file in files_ori:
        file_temp = file
        file = file.split('.')[0] + ".png"
        if file in files_des:
            copy(input1_dir + "\\" + file_temp, outputpath)


def PNG_JPG(inputpath):
    '''
        说明 ：将png转成jpg
        inputpath ： 输入路径
    '''
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


def value_cal(txt):
    '''
        说明 ：根据坐标计算角度
        txt : 坐标文本
    '''
    flag = 0
    f = open(txt)
    data = f.readlines()
    temp_1 = data[35].split(',')
    temp_2 = data[36].split(',')
    if abs(float(temp_1[1]) - float(temp_2[1])) != 0:
        flag = 1
        alpha = round(abs(float(temp_1[0]) - float(temp_2[0])) / abs(float(temp_1[1]) - float(temp_2[1])), 2)
        # 反正切
        inv = np.arctan(alpha)
        # 角度制单位
        return np.degrees(inv), flag
        # return inv, flag
    else:
        return 0, 0


def rotate_1(inputpath, outputpath, value):
    '''
        说明 ：将图片旋转任意角度值并保存旋转后的图片
        picin : 输入图片路径
        picout ： 输出图片路径
        value ： 角度值
    '''
    im1 = PIL.Image.open(inputpath)  # 打开图片路径1
    # 旋转value角度
    im2 = im1.rotate(270 + eval(str(value)))
    # im2.show()
    im2.save(outputpath)  # 保存路径


def rotate(picin, picout, value):
    '''
        说明 ：将图片旋转任意角度值
        picin : 输入图片路径
        picout ： 输出图片路径
        value ： 角度值
    '''
    # img = cv2.imread("plane.jpg")
    img = Image.open(picin)
    img = np.array(img)
    height, width = img.shape[:2]

    degree = 270 + value
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    cv2.imwrite(picout, imgRotation)


def rotatepoint(x, y, pointx, pointy, angle):
    '''
        说明 ：将坐标点以另一个点为圆心旋转
        (x,y) ：为要转的点
        (pointx,pointy) ：为中心点
        angle ： 顺时针旋转角度（弧度制）
    '''
    srx = (x - pointx) * cos(angle) + (y - pointy) * sin(angle) + pointx
    sry = (y - pointy) * cos(angle) - (x - pointx) * sin(angle) + pointy
    return srx, sry


'''
# 根据坐标旋转图片
count = 1
inputpath = "F:\\003  zhuwenwen\\testinput\\"
outputpath = "F:\\003  zhuwenwen\\testoutput\\"
files = os.listdir(inputpath)
for file in files:
    file_name = inputpath + file
    file_pic = inputpath + file + "\\" + file + ".jpg"  # 第5远端指骨
    file_txt = inputpath + file + "\\" + file + ".txt"
    value, flag = value_cal(file_txt)
    if flag == 1:
        outputpath_pic = inputpath + file + "\\" + file + ".jpg"
        rotate_1(file_pic, outputpath_pic, value)
        #read_txt_bak(file_txt, outputpath_pic, file_name, value)
'''

'''
# 根据坐标裁剪图片
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
'''

'''
input_path = "E:\\BBBBOne\\LandMark\\Landmark-Detection-from-Hand-X-ray-Images-Using-Binarization-Perturbation-master\\WebProject\\images\\JST_result"
output_path = "F:\\003__zhuwenwen\\Test_cases\\output_bak"

files_1 = os.listdir(input_path)
files_2 = os.listdir(output_path)
for file in files_1:
    if ".txt" in file:
        file_src = input_path + "\\" + file
        file_dst = output_path + "\\" + file.split('.')[0]
        if os.path.exists(file_dst):
            copy(file_src, file_dst)
'''


def read_txt_rotate(contours, file_img, rotate_image, dir_name):
    read_list = [35, 34, 33, 27, 26, 25, 19, 18, 17, 2, 3]

    for i in read_list:
        print(i)
        temp[0] = float(contours[i][0])
        temp[1] = float(contours[i][1])
        print("temp[0] = ", temp[0])
        print("temp[1] = ", temp[1])
        if i == 35:
            circle_rotate(dir_name, rotate_image, int(temp[1]), int(temp[0]))
        else:
            pass


def show_image_with_keypoints(image, pts):
    '''
        说明 ： 在图片上面画点
        image ： 图片矩阵
        pts ： 点的坐标（list）
    '''
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for p in pts:
        circle = patches.Circle((int(float(p[0])), int(float(p[1]))), 1, linewidth=5, edgecolor='r', fill=False)
        ax.add_patch(circle)
    plt.show()


def rotate_image(img, contours, angle):
    '''
        说明 ： 按角度旋转图片与图片上面的点
        img ： 图片矩阵
        contours ： 点的坐标（list）
        angle ： 角度
    '''
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    rotate_image = imutils.rotate_bound(img, angle)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    rot_pts = []
    for i, shape in enumerate(contours):
        x, y = shape
        p = np.array([x, y, 1]).reshape((3, 1))
        p = np.matmul(M, p)
        rot_pts.append([p[0, 0], p[1, 0]])
        # shape[0] = p[0, 0]
        # shape[1] = p[1, 0]
        # contours[i][i] = shape
    return rotate_image, rot_pts


dir_inpath = "F:\\003  zhuwenwen\\testinput\\"
dir_outpath = "F:\\003  zhuwenwen\\testoutput\\"
count = 1
contours = []
read_list = []
for i in range(37):
    read_list.append(i)
files = os.listdir(dir_inpath)
for file in files:
    file_name = dir_inpath + file + "\\" + file + ".jpg"
    file_txt = dir_inpath + file + "\\" + file + ".txt"
    dir_name = dir_inpath + file
    dir_outname = dir_outpath + file + "_1.jpg"


    del contours[:]
    f = open(file_txt)
    data = f.readlines()
    img = cv2.imread(file_name)
    height = img.shape[0]
    for i in read_list:
        print(i)
        temp = data[i].split(',')
        temp[1] = temp[1][:-1]
        temp[0] = float(temp[0])
        temp[1] = float(temp[1])
        contours.append([temp[1], temp[0]])
    print("contours = ", contours)

    value, flag = value_cal(file_txt)
    angle = 90 - value
    # show_image_with_keypoints(img, contours)

    rotate_images, rot_pts = rotate_image(img, contours, angle)
    cv2.imwrite(dir_outname, rotate_images)
    # show_image_with_keypoints(rotate_image, rot_pts)
    read_txt_rotate(rot_pts, dir_outname, rotate_images, dir_name)