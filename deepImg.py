import os
import cv2
from numpy import *
from PIL import Image

input = "F:\\zhuwenwen\\Test_cases\\input\\"
input_dir = "F:\\zhuwenwen\\Test_cases\\input\\zww.jpg"
output_dir = "F:\\zhuwenwen\\Test_cases\\output\\"
img_dir='F:\\zhuwenwen\\deepImg\\deepImg\\image\\output\\'

#图片处理
class ImgDeal:

    #图片反色
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

    #计算RGB图片像素均值
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

    #计算灰度图像素均值
    def pixel_gray_mean(img_name_gray):
        img = cv2.imread(img_name, 0)
        height, width = img.shape
        size = img.size

        average = 0
        for i in range(height):
            for j in range(width):
                average += img[i][j] / size

class fileDeal:

    def hanzi2pinyin(self):
        count = 0
        path = u"D:\\zhuwenwen\\Datasets\\JSTdetasets\\JST2\\dcm"
        new_path = u"D:\\zhuwenwen\\Datasets\\JSTdetasets\\JST2\\output"
        filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）

        for files in filelist:  # 遍历所有文件
            Olddir = os.path.join(path, files)  # 原来的文件路径
            if os.path.isdir(Olddir):  # 如果是文件夹则跳过
                continue
            filename = os.path.splitext(files)[0]  # 获取文件名
            # 把文件名里的汉字转换成其首字母
            filename1 = lazy_pinyin(filename)
            filenameToStr_1 = ''.join(filename1)

            # 文件扩展名
            filetype = os.path.splitext(files)[1]

            Newdir = os.path.join(new_path, filenameToStr_1 + filetype)

            for i in os.listdir(new_path):
                j = new_path + '\\' + str(i)
                if j == Newdir:
                    Newdir = new_path + '\\' + filenameToStr_1 + str(count) + '.DCM'
                    count += 1

            os.rename(Olddir, Newdir)

class dicom2jpg:
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

    def limitedEqualize(img, limit=4.0):
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

    img_dir = "D:\\zhuwenwen\\Datasets\\JSTdetasets\\JST2\\dcm\\"
    for filename in os.listdir(img_dir):
        savejpg_name = img_dir + filename
        dicom_jpeg(savejpg_name)
        os.remove(savejpg_name)


#根据点的坐标截图
def Shot(img, loc_y, loc_x, w, y):



    img_tmp = img[loc_x - w:loc_x + w, loc_y - y:loc_y + y]

    output_name = output_dir + "1.jpg"
    cv2.imwrite(output_name, img_tmp)


def circle(loc_x, loc_y):
    img = cv2.imread(input_dir)
    cv2.circle(img, (loc_x, loc_y), 10, (0, 0, 213), -1)

    output_name = output_dir + "0.jpg"
    cv2.imwrite(output_name, img)

    img = cv2.imread(output_name)

    sp_x = img.shape[0]
    sp_y = img.shape[1]

    w = sp_x // 24
    y = sp_y // 24

    Shot(img, loc_x, loc_y, w, y)  #x,y

def readTxt(path):

    f = open(path + "zww.txt")

    data = f.readlines()

    #print(data[35], data[34], data[33], data[27], data[26], data[25], data[19], data[18])

    temp = data[35].split(',')
    temp[1] = temp[1][:-1]
    temp[0] = float(temp[0])
    temp[1] = float(temp[1])
    return int(temp[0]), int(temp[1])

x, y = readTxt(input)

circle(y, x)  #x,y
