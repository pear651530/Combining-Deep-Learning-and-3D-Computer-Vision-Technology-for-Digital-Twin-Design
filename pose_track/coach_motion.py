from __future__ import print_function, absolute_import, division
import sys
# import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
from src import viz
# from mpl_toolkits.mplot3d import Axes3D
from PyQt5 import QtCore, QtWidgets, uic, QtGui
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
import cv2
import os
import pickle
import time
import PoseTrackingModel as ptk
# import io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.autograd import Variable
import errno
import socket
import warnings
warnings.filterwarnings("ignore")
#-----------------------------------------------------------------------------------------------------

# global variable
port = 5067       # have to be same as unity

# init TCP connection with unity
# return the socket connected

def init_TCP():
    # '127.0.0.1' = 'localhost' = your computer internal data transmission IP
    address = ('127.0.0.1', port)
    # address = ('192.168.0.107', port)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#Ipv4,TCP協定
        s.connect(address)
        # print(socket.gethostbyname(socket.gethostname()) + "::" + str(port))
        print("Connected to address:", socket.gethostbyname(socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print("Error while connecting :: %s" % e)
        
        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()
def send_info_to_unity(s, args):
    msg = '%.4f ' * len(args) % args

    try:
        s.send(bytes(msg, "utf-8"))
    except tcp_socket.error as e:
        print("error while sending :: " + str(e))

        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()
#-----------------------------------------------------------------------------------------------------

from CTransform import *

# Loading the UI window
qtCreatorFile = "InteractionUI.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

fig = plt.figure()

class Ax3DPose17(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.
    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    self.I   = np.array([1,5,6,1,2,3,1,8,9,10, 9,12,13,9,15,16])-1
    self.J   = np.array([5,6,7,2,3,4,8,9,10,11,12,13,14,15,16,17])-1
    # Left / right indicator
    self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
    self.ax = ax

    vals = np.zeros((17, 3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]])
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]])
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]])
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.
    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 51, "channels should have 51 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (17, -1) )

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    r = 750
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('auto')
def saveData(datapose17, num):
    send_info_to_unity(tcp_socket,
                    (datapose17[0][0],datapose17[0][1],datapose17[0][2],datapose17[0][3],datapose17[0][4],datapose17[0][5],datapose17[0][6],datapose17[0][7],datapose17[0][8]
                     ,datapose17[0][9],datapose17[0][10],datapose17[0][11],datapose17[0][12],datapose17[0][13],datapose17[0][14],datapose17[0][15],datapose17[0][16]
                     ,datapose17[2][0],datapose17[2][1],datapose17[2][2],datapose17[2][3],datapose17[2][4],datapose17[2][5],datapose17[2][6],datapose17[2][7],datapose17[2][8]
                     ,datapose17[2][9],datapose17[2][10],datapose17[2][11],datapose17[2][12],datapose17[2][13],datapose17[2][14],datapose17[2][15],datapose17[2][16]
                     ,datapose17[1][0],datapose17[1][1],datapose17[1][2],datapose17[1][3],datapose17[1][4],datapose17[1][5],datapose17[1][6],datapose17[1][7],datapose17[1][8]
                     ,datapose17[1][9],datapose17[1][10],datapose17[1][11],datapose17[1][12],datapose17[1][13],datapose17[1][14],datapose17[1][15],datapose17[1][16]
                    )
                )

# from 2d pose to 3d pose by network
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size = 17 * 2
        # 3d joints
        self.output_size = 17 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(
            self.linear_stages)  ## 组装模型的容器，容器内的模型只是被存储在ModelList里并没有像nn.Sequential那样严格的模型与模型之间严格的上一层的输出等于下一层的输入

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        # post processing
        y = self.w2(y)

        return y


# from 2d to 3d
class From2dto3d():
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = LinearModel().to(self.device)  # 初始化模型
        self.model.apply(weight_init)  # 初始化模型参数
        path_ckpt = os.getcwd()
        self.checkpoint = torch.load(os.path.join(f'{path_ckpt}/src', 'ckpt_best.pth.tar'))
        self.model.load_state_dict(self.checkpoint['state_dict'])

    def run(self, inputs):
        self.model.eval()  # eval model
        inputs = np.array(inputs)
        inputs = inputs.reshape(1, -1)
        inputs = torch.tensor(inputs)
        inputs = inputs.to(torch.float32)
        inputs = Variable(inputs.cuda())  # convert to Variable type
        self.outputs = self.model(inputs)
        return self.outputs

class ActionRec(QtWidgets.QMainWindow, Ui_MainWindow):
    video_num = '1'
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        super().__init__()
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.init_UDP()
        self.capture = cv2.VideoCapture('motion_video'+self.video_num+'.mp4')  # through computer cam
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.currentFrame = np.array([])
        self.originFrame = np.array([])
        self.thresh = np.array([])
        self.poselist = []
        self.final_pose=[]
        self.start_state = False
        self.save_state = False
        self.flag=1
        self.cTime = 0
        self.pTime = 0
        self.numframes = 0  ## the total number of valid frames
        self.detector = ptk.PoseDetector()
        self.landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

        self.setWindowTitle("coach")
        self.video_lable = QtWidgets.QLabel(self.centralwidget)
        self._timer = QtCore.QTimer(self)  # open Qt timer
        self.setFixedSize(300, 350)
        self.video_lable.setGeometry(0, 0, 300, 350)
        #self.process_input_data()
        self._timer.timeout.connect(self.play)  # response function of Qt timer
        self._timer.start(27)  # the end time of Qt timer, it means get a frame to synthesis GEI every about 27 ms
        self.update()
        self._from2dto3d = From2dto3d()


    def init_UDP(self):
        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT = 5005

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        self.sock.setblocking(0)  # Set non-blocking mode
    def transtoPose17(self, pose3D):
        global xx_new, yy_new
        datapose17 = [[], [], []]
        center = (np.array(pose3D[23]) + np.array(pose3D[24])) * 0.5
        chest = (np.array(pose3D[11]) + np.array(pose3D[12])) * 0.5
        spine = (center + chest) * 0.5
        mouth = (np.array(pose3D[9]) + np.array(pose3D[10])) * 0.5
        neck = (chest + mouth) * 0.5
        head = pose3D[0]
        for i in range(3):
            if i == 0:
                datapose17[0] = [0, pose3D[24][i], pose3D[26][i], pose3D[28][i], pose3D[23][i], pose3D[25][i],
                                 pose3D[27][i], spine[i], chest[i], neck[i],
                                 head[i], pose3D[11][i], pose3D[13][i], pose3D[15][i], pose3D[12][i], pose3D[14][i],
                                 pose3D[16][i]]
                for n, v in enumerate(datapose17[0]):
                    datapose17[0][n] = (center[i] - v)

            if i == 1:
                datapose17[2] = [0, pose3D[24][i], pose3D[26][i], pose3D[28][i], pose3D[23][i], pose3D[25][i],
                                 pose3D[27][i], spine[i], chest[i], neck[i],
                                 head[i], pose3D[11][i], pose3D[13][i], pose3D[15][i], pose3D[12][i], pose3D[14][i],
                                 pose3D[16][i]]

                for n, v in enumerate(datapose17[2]): datapose17[2][n] = (center[i] - v)
                datapose17[0] , datapose17[2] = change_skele_length(datapose17[0], datapose17[2])
            if i == 2:
                twoDposedata = []
                for i in range(17):
                    twoDposedata.append(datapose17[0][i])
                    twoDposedata.append(datapose17[2][i])
                threeDposedata = self._from2dto3d.run(twoDposedata)
                outputs_plot = threeDposedata
                threeDposedata = threeDposedata.data.cpu().numpy().squeeze().tolist()

                outputs_plot = outputs_plot.data.cpu().numpy().squeeze().copy()

                datapose17[1] = threeDposedata[1::3]
                for n, v in enumerate(datapose17[1]):datapose17[1][n] = v * -1

        return datapose17
    def play(self):
        try:
            data, addr = self.sock.recvfrom(1024)
            new_video_num = data.decode()
            
            if new_video_num == self.video_num:
                return False
            # 關閉當前正在播放的影片
            if self.capture.isOpened():
                self.capture.release()
            # 等待一段時間，讓舊的影片捕獲能夠完全釋放
            time.sleep(0.5)  # 這裡的時間可以根據需要調整
            # 更新 video_num 並重新打開新的影片
            self.video_num = new_video_num  # 將 video_num 更新為新影片編號
            video_path = 'motion_video' + self.video_num + '.mp4'
            self.capture = cv2.VideoCapture(video_path)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("video_number: ", self.video_num)
            # 重新初始化 mediapipe 組件
            self.detector = ptk.PoseDetector()
        except BlockingIOError as e:
            if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                # Other error occurred, handle it here if needed
                #print('error1')
                pass
            else:
                # No data received, continue with other code
                #print('error2')
                pass
        ret, self.originFrame = self.capture.read()
        if (ret == True):
            img_pose = self.detector.findPose(self.originFrame)  # get pose from frame
            self.poselist, self._pose2D, self._pose3D, self._DataPose3D = self.detector.findPosition(
                img_pose)  # get all landmarks position
            if len(self.poselist):
                DataPose17array = np.array(self.transtoPose17(self._pose3D))
                saveData(DataPose17array, self.numframes)  ## save 3D pose data
                self.currentFrame = cv2.cvtColor(self.originFrame, cv2.COLOR_BGR2RGB)
                self.currentFrame = cv2.resize(self.currentFrame, (300, 350))
                #cv2.imshow('test',self.currentFrame)
                self.final_pose=DataPose17array
                height, width = self.currentFrame.shape[:2]
                # show fps
                self.cTime = time.time()
                fps = 1 / (self.cTime - self.pTime)
                self.pTime = self.cTime
                cv2.putText(self.currentFrame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)
                #print(width,height)
                #print('a')
                img = QtGui.QImage(self.currentFrame,
                                    width,
                                    height,
                                    QtGui.QImage.Format_RGB888)

                img = QtGui.QPixmap.fromImage(img)
                self.video_lable.setPixmap(img)

            else:
                DataPose17array = np.zeros((3, 17))
                saveData(DataPose17array, self.numframes)
                self.state_print.setText('Failed to get landmarks!')
                self.state_print.setAlignment(QtCore.Qt.AlignCenter)#sys.exit(app.exec_())
            #print(self.final_pose)
        else:
            '''
            if self.flag==1:
                # 儲存成純文字檔案 (text file)
                text_filename = "coordinates"+self.video_num+".txt"
                #print(self.final_pose)
                with open(text_filename, "w") as text_file:
                    for i in range(0,17):
                        x, y, z = self.final_pose[0][i],self.final_pose[2][i],self.final_pose[1][i]
                        text_file.write(f"{x} {y} {z}\n")
                self.flag=2
            '''
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
if __name__ == "__main__":
    tcp_socket = init_TCP()
    app = QtWidgets.QApplication(sys.argv)
    window = ActionRec()
    window.show()
    sys.exit(app.exec_())