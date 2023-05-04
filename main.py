from PyQt5 import QtWidgets
from help import Ui_Form
from PyQt5.QtWidgets import QFileDialog
import cv2
import time
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QGraphicsPixmapItem
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

confThreshold = 0.8
nmsThreshold = 0.4
COLORS = [(0, 255, 255),(0,0,255), (255, 255, 0),(0, 255, 0), (255, 0, 0)]#颜色
color=(0,255,0)
color1=(0,0,255)
c=(0, 0, 0)
class_names = []#初始化一个列表以存储类名

# weights="./moxing/3l/yolov4-tiny-3l_best.weights"
# cfg="./moxing/3l/yolov4-tiny-3l.cfg"
# m="./moxing/3l/coco.names"
weights="./moxing/yolov4-custom_best.weights"
cfg="./moxing/yolov4-custom.cfg"
m="./moxing/coco.names"
# 网络设置
net = cv2.dnn_DetectionModel(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

net.setInputSize(896, 896)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

with open(m, "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
g=0
text1=""

class mywindow(QtWidgets.QWidget, Ui_Form):
    
    def  __init__ (self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.read_file)
        self.pushButton_2.clicked.connect(self.read_voc)
        self.pushButton_3.clicked.connect(self.stop_voc)
        self.cap = []
        self.timer_camera = QtCore.QTimer()
    def read_file(self): #选取文件
        def E2C(label):
            predefined_En=["plate","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","Ao","CA","Er","Ga","Gn","Gg","Gu","Gi","Hi","Hu","Je","Ji","Jg","Jn","Jig","Jin","Li","Lu","Mg","Mi","Ng","Qg","Qo","Sh","Su","Wa","Xi","Xg","Xu","Yu","Yui","Ye","Yn","Zh","Za"
        ]
            predefined_CN=["plate","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","澳","川","鄂","甘","赣","港","贵","桂","黑","沪","吉","冀","津","晋","京","警","辽","鲁","蒙","闽","宁","青","琼","陕","苏","皖","湘","新","学","渝","豫","粤","云","浙","藏"]
            #找到英文label名称在list中的位置
            loc = predefined_En.index(label)
            #显示对应位置的中文名称
            label_CN=predefined_CN[loc]
            return label_CN
        def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=35):
            if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 创建一个可以在给定图像上绘图的对象
            draw = ImageDraw.Draw(img)
            # 字体的格式
            fontStyle = ImageFont.truetype(
                "simsun.ttc", textSize, encoding="utf-8")
            # 绘制文本
            draw.text((left, top), text, textColor, font=fontStyle)
            # 转换回OpenCV格式
            result=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            return result
        filename, filetype =QFileDialog.getOpenFileName(self, "选取文件", "./", "Images (*.png *.jpeg *.jpg)")
        image=cv2.imread(filename)

        image = image[:, :, ::-1].copy()
        a=[]
        b=[]
        left1=0
        top1=0
        classes, confidences, boxes = net.detect(image, confThreshold, nmsThreshold)
        for (classid, score, box) in zip(classes, confidences, boxes):
            left, top, width, height = box                
            if classid ==0:
                #在imgA上画出矩形
                #box1=left, top, width, height
                left1=left
                top1=top
                cv2.rectangle(image, box, color1, 3)
                
            else :
                label = class_names[classid[0]]#标签置信度
                confidence=score*100
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                label_CN = E2C(label)
                a.append(left)
                b.append(label_CN)
                c=zip(a,b)
                d=sorted(c,key=lambda x:x[0])
                e=zip(*d)
                a,b=[list(x) for x in e]
                h="".join(b)
        text = "{}".format(h)
        self.textBrowser.setText(text)
        
        #在imgA上显示中文标签+置信度
        image = cv2ImgAddText(image, text, left1,top1-20)
        height = image.shape[0]
        width = image.shape[1]
        frame = QImage(image, width, height, width*3,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
    
    def read_voc(self): #选取文件
        videoName, _ = QFileDialog.getOpenFileName(self, "Open", "", "*.mp4;;*.avi;;All Files(*)")
        self.cap = cv2.VideoCapture(videoName)
        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.voc)
    
    def voc(self):
        global g
        global text1
        def E2C(label):
            predefined_En=["plate","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","Ao","CA","Er","Ga","Gn","Gg","Gu","Gi","Hi","Hu","Je","Ji","Jg","Jn","Jig","Jin","Li","Lu","Mg","Mi","Ng","Qg","Qo","Sh","Su","Wa","Xi","Xg","Xu","Yu","Yui","Ye","Yn","Zh","Za"
        ]
            predefined_CN=["plate","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","澳","川","鄂","甘","赣","港","贵","桂","黑","沪","吉","冀","津","晋","京","警","辽","鲁","蒙","闽","宁","青","琼","陕","苏","皖","湘","新","学","渝","豫","粤","云","浙","藏"]
            #找到英文label名称在list中的位置
            loc = predefined_En.index(label)
            #显示对应位置的中文名称
            label_CN=predefined_CN[loc]
            return label_CN
        def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=55):
            if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 创建一个可以在给定图像上绘图的对象
            draw = ImageDraw.Draw(img)
            # 字体的格式
            fontStyle = ImageFont.truetype(
                "simsun.ttc", textSize, encoding="utf-8")
            # 绘制文本
            draw.text((left, top), text, textColor, font=fontStyle)
            # 转换回OpenCV格式
            result=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            return result
        if (self.cap.isOpened()):
            ret, self.frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                start = time.time()
                classes, confidences, boxes = net.detect(frame, confThreshold, nmsThreshold)
                end = time.time()
                start_drawing = time.time()
                count=0
                a=[]
                b=[]
                f=0
                h=""
                
                for (classid, score, box) in zip(classes, confidences, boxes):
                    #if classid==0 or classid==1 or classid==2 or classid==3 or classid==4 or classid==5 or classid==6:
                    left, top, width, height = box
                    if classid ==0:
                        #在imgA上画出矩形
                        #box1=left, top, width, height
                        left1=left
                        top1=top
                        cv2.rectangle(frame, box, color1, 3)
                        f=1
                    else:
                        label = class_names[classid[0]]#标签置信度
                        confidence=score*100
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        
                        label_CN = E2C(label)
                        a.append(left)
                        b.append(label_CN)
                        c=zip(a,b)
                        d=sorted(c,key=lambda x:x[0])
                        e=zip(*d)
                        a,b=[list(x) for x in e]
                        h="".join(b)
                
                #输出中文标签和置信度
                # text = "{}:{}".format(b, confidence)
                if len(h)==7 and h[0] in ["澳","川","鄂","甘","赣","港","贵","桂","黑","沪","吉","冀","津","晋","京","警","辽","鲁","蒙","闽","宁","青","琼","陕","苏","皖","湘","新","学","渝","豫","粤","云","浙","藏"]:
                    g=1
                    text = "{}".format(h)
                    text1=text
                    print(text1)
                
            #在imgA上显示中文标签+置信度
                #if f==1 and g==1:
                #    frame = cv2ImgAddText(frame, text1, left1,top1-20)
                
                if g==1:
                    frame = cv2ImgAddText(frame, text1, 100,100)
                    self.textBrowser.setText(text1)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frame.data, width, height, bytesPerLine,QImage.Format_RGB888)
                pix = QPixmap.fromImage(q_image)
                self.item = QGraphicsPixmapItem(pix)
                self.scene = QGraphicsScene()  # 创建场景
                self.scene.addItem(self.item)
                self.graphicsView.setScene(self.scene)
            else:
                self.cap.release()
                self.timer_camera.stop()  # 停止计时器

    def stop_voc(self):
         if self.cap != []:
            self.cap.release()
            self.timer_camera.stop()  # 停止计时器
            
#################################################################

if __name__=="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    ui = mywindow()    
    ui.show()
    sys.exit(app.exec_())
