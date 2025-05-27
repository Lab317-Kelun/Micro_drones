# -*- coding: UTF-8 -*-
'''
@author: mengting gu
@contact: 1065504814@qq.com
@time: 2021/2/19 16:57
@file: eval.py
@desc: 
'''
# -*-coding:utf-8-*-
import argparse
import logging
import time

from robot_cv.msg import robot_data
###
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import yaml
from easydict import EasyDict

from models import get_model # type: ignore

from utils import (
    Logger,
    count_parameters,
    data_augmentation,
    get_data_loader,
)
###

import numpy as np

import cv2

import rospy
from std_msgs.msg import Int8
import torch.nn.functional as F
from PIL import Image

import cv2
import numpy as np
import math

pi = 3.14
i=1
###
parser = argparse.ArgumentParser(description="PyTorch CIFAR Dataset Training")
parser.add_argument("--work-path", default= "/home/nvidia/robot_ws/src/robot_cv/script", type=str)
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")

args = parser.parse_args()
logger = Logger(
    log_file_name=args.work_path + "/log.txt",
    log_level=logging.DEBUG,
    logger_name="CIFAR",
).get_log()
config = None
###

def eval(test_loader, net, device):

    net.eval()

    correct = 0
    total = 0

    logger.info(" === Validate ===")

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze()
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # logger.info(
            #     "   == test acc: {:6.3f}% | true label : {}, predict as : {}".format(
            #          100.0 * correct / total, targets, predicted
            #     )
            # )
        logger.info(
            "   == test acc: {:6.3f}% , model best prec : {:6.3f}%".format(
                100.0 * correct / total, best_prec
            )
        )

def get_key_by_value(value,default=None):
  
    print(f"value:{value.item()}")
    value = value.item()
    print(value)
    print(class_dict)
    for key,val in class_dict.items():
        print(f"key:{key},val:{val}")
        if val == value:
            print(f"pipei")
            return key
            
        return default
def deal_image(img):
    #cv2.imshow("threshold_hou", img)
    #cv2.waitKey(0)
    
    cv2.namedWindow("threshold",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("threshold",640,480)
    #cv2.imshow('img1',img)
    #cv2.waitKey(0)
    #img = cv2.pyrDown(img, cv2.IMREAD_UNCHANGED)
    # threshold 函数对图像进行二化值处理，由于处理后图像对原图像有所变化，因此img.copy()生成新的图像，cv2.THRESH_BINARY是二化值
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 140, 255, cv2.THRESH_BINARY) #
    
    #thresh=cv2.adaptiveThreshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,2)
    cv2.imshow("threshold_erzhi", thresh)
    cv2.waitKey(0)
    #开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10, 10))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("threshold", closed)
    cv2.waitKey(0)

    # 层次类型，参数cv2.RETR_EXTERNAL是获取最外层轮廓，cv2.RETR_TREE是获取轮廓的整体结构
    #contours是图像的轮廓、hier是层次类型
    contours, hier = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("lunkuo")
    print(contours)
    for c in contours:
        # 轮廓绘制方法一
        # boundingRect函数计算边框值，x，y是坐标值，w，h是矩形的宽和高
        #x, y, w, h = cv2.boundingRect(c)
        # 在img图像画出矩形，(x, y), (x + w, y + h)是矩形坐标，(0, 255, 0)设置通道颜色，2是设置线条粗度
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 轮廓绘制方法二
        # 查找最小区域
        #rect = cv2.minAreaRect(c)
        # 计算最小面积矩形的坐标
        #box = cv2.boxPoints(rect)
    # 将坐标规范化为整数
        #box = np.int0(box)
    # 绘制矩形
        #cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

    # 轮廓绘制方法三
    # 圆心坐标和半径的计算
        (x, y), radius = cv2.minEnclosingCircle(c)
    # 规范化为整数
        center = (int(x), int(y))
        radius = int(radius)
        #if pi*radius*radius > 300:    #300000
        m=pi*radius*radius
        print(m)
        #if m>15000 and m<30000:
        #if m>3000 and m<10000:
        if True:
            print("manzu_circle")
            print(m)
    # 勾画圆形区域
            #cv2.imshow("threshold", img)
            #cv2.waitKey()            
            img = cv2.circle(img, center, radius, (0, 255, 0), 2)
            #cv2.imshow("threshold_circle", img)
            #cv2.waitKey(0)
            a = math.sqrt((radius*radius)/2)
            print(f"center:{center}")
            
            print(m)
            #cv2.rectangle(img, (int(x-a), int(y-a)), (int(x+a), int(y+a)), (0, 0, 255), 2)
            
            #cv2.imshow("threshold", img)
            #cv2.waitKey()
            
            height=width=int(a+a)
            #cropped_img = img[start_point[1]:int(start_point[1]+height), start_point[0]:int(start_point[0]+width)]
            cropped_img = img[int(y-a):int(y+a), int(x-a):int(x+a)]
            #cv2.imshow("threshold1", cropped_img)
            #cv2.waitKey()
            ###
            #cv2.namedWindow("contours",cv2.WINDOW_NORMAL)

            #cv2.resizeWindow("contours",640,480)
            #cv2.imshow("contours", cropped_img)
            #cv2.waitKey()
            #cv2.destroyAllWindows()S
            print("have circle")
            #cv2.imshow("threshold111", img)
            #cv2.waitKey(0)
            #return cropped_img
            
        else:
             print(f"no circle")
    cv2.imshow("threshold222", img)
    cv2.waitKey()
    #return img

# # 轮廓绘制方法四
# 围绕图形勾画蓝色线条
    #cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
# 显示图像
    #cv2.namedWindow("contours",cv2.WINDOW_NORMAL)

    #cv2.resizeWindow("contours",640,480)
    #cv2.imshow("contours", img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    #if i==0 :
        #return cropped_img
    #else:
       # return img


        
def one_image_demo():
    global net
    
    img = np.zeros((32,32),np.uint8)
    ###
    net.eval()                                   #important
    # img = image_processing.read_image(image_path, resize_height=config.input_size, resize_width=config.input_size)

    ###
    cap=cv2.VideoCapture(0)
            
    while (cap.isOpened()):

        ret,frame=cap.read()
        if not ret:  
            print("无法接收帧（流可能已结束）")  
            break
        #cv2.imshow('frame',frame)
        #cv2.waitKey(0)
        #time.sleep(1)
        if cv2.waitKey(5) >= 0:
            break
        img = frame
        #cv2.imshow("threshold", img)
        #cv2.waitKey()
        #cv2.imshow('frame1',img)
        #cv2.waitKey(0)           #
        if img.any():
            break
    cap.release()  
    cv2.destroyAllWindows()
    


    
    ###
    #img = cv2.imread(img)
    #img = '/home/nvidia/robot_ws/src/robot_cv/script/0_1.jpg'
    #img = cv2.imread(img,1)
    #cv2.imshow('img1',img)
    #cv2.waitKey(0)
    #cv2.imshow("threshold1", img)
    #cv2.waitKey()
    img = deal_image(img)
    print("sucess_return")
    #cv2.imshow("threshold", img)
    #cv2.waitKey()
    cv2.resize(img,(32,32))
    
    pil_image = Image.fromarray(img)
    transform = transforms.Compose([transforms.ToTensor()])
    data_loader_val = transform(pil_image).unsqueeze(0)
    print(data_loader_val.shape)
    
    data_loader_val=F.interpolate(data_loader_val,size=(32,32),mode='area')
    print(data_loader_val.shape)
    print(f"device:{device}")
    data_loader_val = data_loader_val.to(device, non_blocking=True)
    print(f"net:{net}")
    outputs = net(data_loader_val)
    _, predicted = outputs.max(1)
    # print(outputs)
    
    print("img : {}, predict as : {}".format(img, predicted[0]))
    print(predicted[0])
    
    #class_dict={'apple':0,'aquarium_fish':1,'baby':2}
    class_dict = {'apple':0,'aquarium_fish':1,'baby':2,'bear':3,'beaver':4,'bed':5,'bee':6,'beetle':7,'bicycle':8,'bottle':9,'bowl':10,'boy':11,'bridge':12,'bus':13,'butterfly':14,'camel':15,'can':16,'castle':17,'caterpillar':18,'cattle':19,'chair':20,'chimpanzee':21,'clock':22,'cloud':23,'cockroach':24,'couch':25,'crab':26,'crocodile':27,'cup':28,'dinosaur':29,'dolphin':30,'elephant':31,'flatfish':32,'forest':33,'fox':34,'girl':35,'hamster':36,'house':37,'kangaroo':38,'keyboard':39,'lamp':40,'lawn_mower':41,'leopard':42,'lion':43,'lizard':44,'lobster':45,'man':46,'maple_tree':47,'motorcycle':48,'mountain':49,'mouse':50,'mushroom':51,'oak_tree':52,'orange':53,'orchid':54,'otter':55,'palm_tree':56,'pear':57,'pickup_truck':58,'pine_tree':59,'plain':60,'plate':61,'poppy':62,'porcupine':63,'possum':64,'rabbit':65,'raccoon':66,'ray':67,'road':68,'rocket':69,'rose':70,'sea':71,'seal':72,'shark':73,'shrew':74,'skunk':75,'skyscraper':76,'snail':77,'snake':78,'spider':79,'squirrel':80,'streetcar':81,'sunflower':82,'sweet_pepper':83,'table':84,'tank':85,'telephone':86,'television':87,'tiger':88,'tractor':89,'train':90,'trout':91,'tulip':92,'turtle':93,'wardrobe':94,'whale':95,'willow_tree':96,'wolf':97,'woman':98,'worm':99}
    value=predicted[0].item()
    print(value)
    print(type(value))
    for key,val in class_dict.items():
        print(f"key:{key},val:{val}")
        print(type(val))
        if abs(val-value) < 0.1:
            result=key
            print(f"类的名称是：{result}")
            break                                       ###miao a!!!
            
        else:
            print("err")

    #result = get_key_by_value(predicted[0], default='Not Found')
    
    #############对比识别到的类名和二维码的类名
    ###
    predict_class=result
    print(" 打开文件")
    file = open('/home/nvidia/robot_ws/src/robot_cv/script/test.txt', mode='r')
    real_class = file.read()
    print(real_class)
    array_class = []
    array_class = real_class.split(",")
    print(array_class)
    #diyigezhi
    print(array_class[0])
    
    for i in range(0,3):
        if array_class[i]==predict_class:
            print(" 投放物块")
            #发送投放物块的标志1
            #pub.publish(1)
            robot_Data.is_drop=1
            robot_Data.is_leave=1
            pub.publish(robot_Data)
            robot_Data.is_drop=0
            robot_Data.is_leave=0
            pub.publish(robot_Data)
        else:
            robot_Data.is_drop=0
            robot_Data.is_leave=1
            pub.publish(robot_Data)
            robot_Data.is_leave=0
            pub.publish(robot_Data)
            print("not drop")
            
    file.close()
###
    #img = cv2.imread(image_path)
    # 确保成功读取图像文件
    # 将OpenCV图像转换为RGB格式
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_rgb = cv2.resize(img, (240, 240))
    
    # 将OpenCV图像转换为PIL Image
    #pil_image = Image.fromarray(img_rgb)
    #transform = transforms.Compose([transforms.ToTensor()])
    #data_loader_val = transform(pil_image).unsqueeze(0)
###
def call_back(msg):
    
   
    data = msg.data  #接收到的msg对象通常是一个ROS消息类型的实例，而不是一个简单的数值
    print(msg)
    print(data)

    if data == 1 : 
        
        print("成功接收到1")
        one_image_demo()

def init_net():
    # pass
    global frame,device,net
    ###
    global args, config, best_prec 

    # read config from yaml file
    with open(args.work_path + "/config.yaml") as f:
        config = yaml.load(f,Loader=yaml.SafeLoader)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)

    # define netowrk
    net = get_model(config)
    #ckpt_file_name = args.work_path + "/" + config.ckpt_name + ".pth.tar"
    ckpt_file_name = '/home/nvidia/robot_ws/src/robot_cv/script/densenet100bc_best_best.pth.tar'
    checkpoint = torch.load(ckpt_file_name)
    net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint["state_dict"].items()}, strict=True)
    best_prec = checkpoint["best_prec"]
    logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = "cuda" if config.use_gpu else "cpu"
    # data parallel for multiple-GPU
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.to(device)

    # load training data, do data augmentation and get data loader
    #transform_train = transforms.Compose(data_augmentation(config))
    #transform_test = transforms.Compose(data_augmentation(config, is_train=False))
    #train_loader, test_loader = get_data_loader(transform_train, transform_test, config)
    #eval(, net, device)
    ###

    
    #one_image_demo demo
    # image_path = "D:/temp/dataset/dataset/train/character_0/1.jpg"
    # one_image_demo(img, net, device)
    print("进入mian函数")
    #rospy.init_node('apple')
    #sub=rospy.Subscriber('classify_image',Int8,call_back(device))
    #pub=rospy.Publisher('c_object',Int8,queue_size=10)
    #rate = rospy.Rate(1)
    #print("成功创建订阅者")
    #while True:
      #  {
            
        #}

    

if __name__ == "__main__":
   
    init_net()
    rospy.init_node('apple')
    robot_Data=robot_data()
    pub=rospy.Publisher('c_object',robot_data,queue_size=10)
    sub=rospy.Subscriber('classify_image',Int8,call_back)


    #rate = rospy.Rate(1)
    print("成功创建订阅者")
    while True:
        rate = rospy.Rate(1)
      
    

        


    
