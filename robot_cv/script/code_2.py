from array import array
import cv2
from pyzbar.pyzbar import decode
from std_msgs.msg import Int8
import rospy
data = ' '
####收到话题内容为1，将fram传入image识别
# 读取图像
def call_back(msg):
    data=msg.data
    if data == 1:
        while(1):
    
            a=code()
            if a==1:
                print("识别二维码sucess")
                break
        #if a==0:
            #print("fail1")
            #a=code();
            #if a==0:
               ## print("fail2")
               # a=code()
               # if a==0:
                  #  print("fail3")
                   # a=code()
            else:
                print("识别二维码fail,retrying")

        

def code():
    data_str = 0
    pub=rospy.Publisher('code_sucess',Int8,queue_size=10)
    cap=cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret,frame= cap.read()
        if not ret:
            break
        #cv2.imshow('frame',frame)
        if cv2.waitKey(5)>=0:
            break
        image=frame
        #cv2.imshow('frame1',image)
        #cv2.waitKey(0)
        if image.any():
            break
        #转换为灰度图像
    # image = '/home/chenchen/code-python/work_code_and_redapple/image.png'
    # image=cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用 Pyzbar 进行二维码解码
    decoded_objects = decode(gray_image)
    #print("for")
    # 打印识别的结果
    for obj in decoded_objects:
        data_str=obj.data.decode('utf-8')
        array=[]
        array=data_str.split(",")
        #print('Type:', obj.type)
        print('Data:', array)  # 解码二维码数据
    
    if data_str == 0 :
        return 0
    file = open('test.txt', mode='w+', encoding='utf-8')
    #  write 写入
    file.write(data_str)
    # 关he闭文件，不关闭文件可能会出问题
    file.close()
    #读取文件内容
    file = open('test.txt', mode='r')
    real_class=file.read()
    print(real_class)
    array_class=[]
    array_class=real_class.split(",")
    #print(array_class)
    #print(array_class[2])
    if array_class[2]=='left':
    	
        data=1
    else:
        data=2
    pub.publish(data)
    print(f"pub_sucess:{data}")
    file.close()
    return 1

    
    
def main():
    rospy.init_node('code')

    rospy.Subscriber('re_code',Int8,call_back)
    print("创建接收者成功")

    while True:
        pass
    

    # image = cv2.imread('D:/temp/dataset/CIFAR-ZOO-master/unique/code2.png')



if __name__=="__main__":

    main()
