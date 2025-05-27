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

from PIL import Image
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import yaml
from easydict import EasyDict
from torchvision.io import read_image

from models import get_model
from utils import (
    Logger,
    count_parameters,
    data_augmentation,
    get_data_loader,
)

import numpy as np



parser = argparse.ArgumentParser(description="PyTorch CIFAR Dataset Training")
parser.add_argument("--work-path", default= "./experiments/cifar100/densenet100bc", type=str)
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")

args = parser.parse_args()
logger = Logger(
    log_file_name=args.work_path + "/log.txt",
    log_level=logging.DEBUG,
    logger_name="CIFAR",
).get_log()
config = None


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
def get_key_by_value(dictionary, value, default=None):
    for key, val in dictionary.items():
        if val == value:
            return key
    return default
def one_image_demo(image_path, net, device):
    net.eval()
    img = cv2.imread(image_path)
    pil_image = Image.fromarray(img)
    transform = transforms.Compose([transforms.ToTensor()])
    data_loader_val = transform(pil_image).unsqueeze(0)
    print(data_loader_val.shape)

    data_loader_val = data_loader_val.to(device, non_blocking=True)
    outputs = net(data_loader_val)
    _, predicted = outputs.max(1)
    # print(outputs)

    print("img : {}, predict as : {}".format(image_path, predicted[0]))
    class_dict = {'apple':0,'aquarium_fish':1,'baby':2,'bear':3,'beaver':4,'bed':5,'bee':6,'beetle':7,'bicycle':8,'bottle':9,'bowl':10,'boy':11,'bridge':12,'bus':13,'butterfly':14,
                  'camel':15,'can':16,'castle':17,'caterpillar':18,'cattle':19,'chair':20,'chimpanzee':21,'clock':22,'cloud':23,'cockroach':24,'couch':25,'crab':26,'crocodile':27,'cup':28,
                  'dinosaur':29,'dolphin':30,'elephant':31,'flatfish':32,'forest':33,'fox':34,'girl':35,'hamster':36,'house':37,'kangaroo':38,'keyboard':39,'lamp':40,'lawn_mower':41,'leopard':42,'lion':43,'lizard':44,
                  'lobster':45,'man':46,'maple_tree':47,'motorcycle':48,'mountain':49,'mouse':50,'mushroom':51,'oak_tree':52,'orange':53,'orchid':54,'otter':55,'palm_tree':56,'pear':57,'pickup_truck':58,'pine_tree':59,'plain':60,
                  'plate':61,'poppy':62,'porcupine':63,'possum':64,'rabbit':65,'raccoon':66,'ray':67,'road':68,'rocket':69,'rose':70,'sea':71,'seal':72,'shark':73,'shrew':74,'skunk':75,'skyscraper':76,'snail':77,'snake':78,'spider':79,'squirrel':80,
                  'streetcar':81,'sunflower':82,'sweet_pepper':83,'table':84,'tank':85,'telephone':86,'television':87,'tiger':88,'tractor':89,'train':90,'trout':91,'tulip':92,
                  'turtle':93,'wardrobe':94,'whale':95,'willow_tree':96,'wolf':97,'woman':98,'worm':99}


    result = get_key_by_value(class_dict,predicted[0], default='Not Found')
    print(f"类的名称是：{result}")
def main():
    global args, config, best_prec

    # read config from yaml file
    with open(args.work_path + "/config.yaml") as f:
        config = yaml.load(f,Loader=yaml.SafeLoader)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)

    # define netowrk
    net = get_model(config)
    ckpt_file_name = args.work_path + "/" + config.ckpt_name +"_best.pth.tar"
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
    transform_train = transforms.Compose(data_augmentation(config))
    transform_test = transforms.Compose(data_augmentation(config, is_train=False))
    train_loader, test_loader = get_data_loader(transform_train, transform_test, config)
    eval(test_loader, net, device)


    #one_image_demo demo
    #image_path = "C:/Users/Kelun/Desktop/5.png"
    image_path = "C:/Users/Kelun/Desktop/dataset/val/character__64/23.jpg"
    # image_path = "C:/Users/Kelun/Desktop/24年比赛道具/cifar100/test/&crocodile&_reptiles_61.png"
    one_image_demo(image_path, net, device)



if __name__ == "__main__":
    main()