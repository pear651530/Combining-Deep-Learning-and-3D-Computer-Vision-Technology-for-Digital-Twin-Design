from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFrame, QPushButton, QGridLayout, QHBoxLayout,QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import subprocess
import ctypes
import os
from PyQt5.QtGui import QResizeEvent
GWL_STYLE = -16
WS_VISIBLE = 0x10000000
import win32con
import threading
import sys
import subprocess
import win32gui
import time
import socket
from PyQt5.QtCore import Qt, QTimer
video_num='3'
motivation=""
right_label_2 = None
# 定義發送UDP的函數
def send_udp_data(video_num):
    UDP_IP = "127.0.0.1"  # 設置接收端的IP地址
    UDP_PORT = 5005       # 設置接收端的UDP端口

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(str(video_num).encode(), (UDP_IP, UDP_PORT))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(str(video_num).encode(), (UDP_IP, 5006))
# class 1: unity
class UnityContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.unity_hwnd = None
        self.top_left_widget = parent  # 使用 parent 作為 top_left_widget

    def run_unity_program(self):
        self.process = subprocess.Popen(['student.exe'])

    def find_unity_window(self):
        def enum_windows_callback(hwnd, hwnds):
            if "UnityWndClass" in win32gui.GetClassName(hwnd):
                hwnds.append(hwnd)

        hwnds = []
        win32gui.EnumWindows(enum_windows_callback, hwnds)
        if hwnds:
            self.unity_hwnd = hwnds[0]
            #print(hwnds)

    def embed_unity_window(self):
        if self.unity_hwnd:
            ctypes.windll.user32.SetParent(int(self.unity_hwnd), int(self.winId()))
            style = ctypes.windll.user32.GetWindowLongPtrA(int(self.unity_hwnd), ctypes.c_int(GWL_STYLE))
            style |= WS_VISIBLE
            ctypes.windll.user32.SetWindowLongPtrA(int(self.unity_hwnd), ctypes.c_int(GWL_STYLE), style)
            win32gui.SetForegroundWindow(self.unity_hwnd)
            
            win32gui.SetWindowPos(self.unity_hwnd, win32con.HWND_TOP, 0, 0, 300, 380, win32con.SWP_SHOWWINDOW)


    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self.unity_hwnd:
            win32gui.MoveWindow(self.unity_hwnd, 0, 0, 300, 380, True)

    def closeEvent(self, event):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        
# class 2: python
class PythonContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.python_hwnd = None
        self.bottom_left_widget = parent  # 使用 parent 作為 top_left_widget

    def run_python_program(self):
        # 執行另一個 Python 檔案
        result = subprocess.run(['python', 'student_motion.py'])
        
    def find_python_window(self):
        def enum_windows_callback(hwnd, hwnds):
            #class_name = win32gui.GetClassName(hwnd)
            window_text = win32gui.GetWindowText(hwnd)
            #print(f"Class name: {class_name}, Window text: {window_text}")
            if "MainWindow" in win32gui.GetWindowText(hwnd):
                hwnds.append(hwnd)

        hwnds = []
        win32gui.EnumWindows(enum_windows_callback, hwnds)
        #print(hwnds)
        if hwnds:
            self.python_hwnd = hwnds[0]

    def embed_python_window(self):
        if self.python_hwnd:
            ctypes.windll.user32.SetParent(int(self.python_hwnd), int(self.winId()))
            style = ctypes.windll.user32.GetWindowLongPtrA(int(self.python_hwnd), ctypes.c_int(GWL_STYLE))
            style |= WS_VISIBLE
            ctypes.windll.user32.SetWindowLongPtrA(int(self.python_hwnd), ctypes.c_int(GWL_STYLE), style)
            win32gui.SetForegroundWindow(self.python_hwnd)
            
            win32gui.SetWindowPos(self.python_hwnd, win32con.HWND_TOP, 0, 0, 300, 380, win32con.SWP_SHOWWINDOW)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self.python_hwnd:
            win32gui.MoveWindow(self.python_hwnd, 0, 0, 300, 380, True)

    def closeEvent(self, event):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

# class 3: unity
class UnityContainer1(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.unity_hwnd = None
        self.top_left_widget = parent  # 使用 parent 作為 top_left_widget

    def run_unity_program(self):
        self.process = subprocess.Popen(['coach.exe'])

    def find_unity_window(self):
        def enum_windows_callback(hwnd, hwnds):
            if "coach" in win32gui.GetWindowText(hwnd):
                hwnds.append(hwnd)

        hwnds = []
        win32gui.EnumWindows(enum_windows_callback, hwnds)
        if hwnds:
            self.unity_hwnd = hwnds[0]
            #print(hwnds)

    def embed_unity_window(self):
        if self.unity_hwnd:
            ctypes.windll.user32.SetParent(int(self.unity_hwnd), int(self.winId()))
            style = ctypes.windll.user32.GetWindowLongPtrA(int(self.unity_hwnd), ctypes.c_int(GWL_STYLE))
            style |= WS_VISIBLE
            ctypes.windll.user32.SetWindowLongPtrA(int(self.unity_hwnd), ctypes.c_int(GWL_STYLE), style)
            win32gui.SetForegroundWindow(self.unity_hwnd)
            
            win32gui.SetWindowPos(self.unity_hwnd, win32con.HWND_TOP, 0, 0, 300, 380, win32con.SWP_SHOWWINDOW)


    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self.unity_hwnd:
            win32gui.MoveWindow(self.unity_hwnd, 0, 0, 300, 380, True)

    def closeEvent(self, event):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

# class 4: python
class PythonContainer1(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.python_hwnd = None
        self.bottom_right_widget = parent  # 使用 parent 作為 top_left_widget

    def run_python_program(self):
        # 執行另一個 Python 檔案並將位元組數據傳遞給 stdin
        send_udp_data(video_num)
        result = subprocess.run(['python', 'coach_motion.py'])


    def find_python_window(self):
        def enum_windows_callback(hwnd, hwnds):
            #class_name = win32gui.GetClassName(hwnd)
            window_text = win32gui.GetWindowText(hwnd)
            #print(f"Class name: {class_name}, Window text: {window_text}")
            if "coach" in win32gui.GetWindowText(hwnd):
                hwnds.append(hwnd)

        hwnds = []
        win32gui.EnumWindows(enum_windows_callback, hwnds)
        #print(hwnds)
        if hwnds:
            self.python_hwnd = hwnds[0]

    def embed_python_window(self):
        if self.python_hwnd:
            ctypes.windll.user32.SetParent(int(self.python_hwnd), int(self.winId()))
            style = ctypes.windll.user32.GetWindowLongPtrA(int(self.python_hwnd), ctypes.c_int(GWL_STYLE))
            style |= WS_VISIBLE
            ctypes.windll.user32.SetWindowLongPtrA(int(self.python_hwnd), ctypes.c_int(GWL_STYLE), style)
            win32gui.SetForegroundWindow(self.python_hwnd)
            
            win32gui.SetWindowPos(self.python_hwnd, win32con.HWND_TOP, 0, 0, 300, 380, win32con.SWP_SHOWWINDOW)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self.python_hwnd:
            win32gui.MoveWindow(self.python_hwnd, 0, 0, 300, 380, True)

    def closeEvent(self, event):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        self.bottom_right_widget.removeWidget(self)
        self.bottom_right_widget.setCurrentIndex(0)
        event.accept()
def openUnity():
    try:
        # 获取当前脚本所在的文件夹路径
        current_directory = os.path.dirname(os.path.abspath(__file__))
        # 构建 Unity Hub.exe 的完整路径
        unity_hub_path = os.path.join(current_directory+'\\Unity Hub\\', "Unity Hub.exe")
        # 使用 subprocess 打开 Unity Hub.exe
        subprocess.Popen([unity_hub_path])
    except Exception as e:
        print(f"Error: {e}")
def openUsage():
    try:
        subprocess.Popen(['start', 'winword', '系統使用手冊.docx'], shell=True)
    except Exception as e:
        print(f"Error opening Word document: {e}")
mmm="腿部1"
def on_button1_clicked():
    global mmm
    video_num='1'
    mmm="腿部1"
    print(video_num)
    send_udp_data(video_num)
def on_button2_clicked():
    global mmm
    mmm="腿部2"
    video_num='2'
    print(video_num)
    send_udp_data(video_num)
def on_button3_clicked():
    global mmm
    mmm="腿部3"
    video_num='3'
    print(video_num)
    send_udp_data(video_num)
def on_button4_clicked():
    global mmm
    mmm="手部1"
    video_num='4'
    print(video_num)
    send_udp_data(video_num)
def on_button5_clicked():
    global mmm
    mmm="手部2"
    video_num='5'
    print(video_num)
    send_udp_data(video_num)

def update_motivation():
    global motivation, right_label_2,mmm
    print("update start...")
    try:
        text_filename = "motion_move.txt"
        with open(text_filename, "r") as text_file:
            motivation = ""
            motivation=motivation+mmm+'\n'
            for line in text_file:
                motivation += line
    except:
        motivation = "Wait a seconds..."
    right_label_2.setText(motivation)
    right_label_2.setWordWrap(True)

def main():
    global motivation, right_label_2
    text_filename = "motion_move.txt"
    with open(text_filename, "w") as text_file:
        text_file.write(f"Wait a seconds...\n")
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("健身學習系統")
    window.setFixedSize(1200, 900)  # 設定視窗大小
    # 建立左中右區域的佈局
    main_widget = QWidget()
    main_layout = QHBoxLayout(main_widget)

    left_widget = QWidget()
    middle_widget = QWidget()
    right_widget = QWidget()

    main_layout.addWidget(left_widget, 1)
    main_layout.addWidget(middle_widget, 5)
    main_layout.addWidget(right_widget, 2)
    #右邊按鈕切換影片
    right_layout = QVBoxLayout(right_widget)
    right_label_1 = QLabel("<如何改進動作>\n")
    # 設置字體和大小
    font = QFont("標楷體", 16)
    right_label_1.setFont(font)
    # 將 right_label_1 放在最上方
    right_layout.addWidget(right_label_1, 0, Qt.AlignTop| Qt.AlignHCenter)
    #print(motivation)
    right_label_2 = QLabel(motivation)
    right_label_2.setFont(font)

    right_label_2.setWordWrap(True)

    # 將 right_label_2 放在最上方
    right_layout.addWidget(right_label_2)
    right_layout.addStretch()


    #左邊按鈕切換影片
    left_layout = QVBoxLayout(left_widget)
    left_label = QLabel("選擇想要學習的動作")
    # 設置字體和大小
    font = QFont("標楷體", 16)
    left_label.setFont(font)
    # 將 left_label 放在最上方
    left_layout.addWidget(left_label, 0, Qt.AlignTop)

    # 添加按鈕
    button1_ = QPushButton("腿部1")
    button2_ = QPushButton("腿部2")
    button3_ = QPushButton("腿部3")
    button4_ = QPushButton("手部1")
    button5_ = QPushButton("手部2")
    button6_ = QPushButton("手部3")
    button7_ = QPushButton("全身1")
    button8_ = QPushButton("全身2")
    button9_ = QPushButton("全身3")
    button10_ = QPushButton("開啟Unity")
    button11_ = QPushButton("開啟使用手冊")
    # 設置按鈕字體大小為14
    font = QFont("標楷體", 16)
    button1_.setFont(font)
    button2_.setFont(font)
    button3_.setFont(font)
    button4_.setFont(font)
    button5_.setFont(font)
    button6_.setFont(font)
    button7_.setFont(font)
    button8_.setFont(font)
    button9_.setFont(font)
    button10_.setFont(font)
    button11_.setFont(font)
    button1_.setFixedSize(200, 50)
    button2_.setFixedSize(200, 50)
    button3_.setFixedSize(200, 50)
    button4_.setFixedSize(200, 50)
    button5_.setFixedSize(200, 50)
    button6_.setFixedSize(200, 50)
    button7_.setFixedSize(200, 50)
    button8_.setFixedSize(200, 50)
    button9_.setFixedSize(200, 50)
    button10_.setFixedSize(200, 50)
    button11_.setFixedSize(200, 50)

    left_layout.addWidget(button1_, 0, Qt.AlignTop)
    left_layout.addWidget(button2_, 0, Qt.AlignTop)
    left_layout.addWidget(button3_, 0, Qt.AlignTop)
    left_layout.addWidget(button4_, 0, Qt.AlignTop)
    left_layout.addWidget(button5_, 0, Qt.AlignTop)
    left_layout.addWidget(button6_, 0, Qt.AlignTop)
    left_layout.addWidget(button7_, 0, Qt.AlignTop)
    left_layout.addWidget(button8_, 0, Qt.AlignTop)
    left_layout.addWidget(button9_, 0, Qt.AlignTop)
    left_layout.addStretch()
    left_layout.addWidget(button10_, 0, Qt.AlignTop)
    left_layout.addWidget(button11_, 0, Qt.AlignTop)
    button1_.clicked.connect(on_button1_clicked)
    button2_.clicked.connect(on_button2_clicked)
    button3_.clicked.connect(on_button3_clicked)
    button4_.clicked.connect(on_button4_clicked)
    button5_.clicked.connect(on_button5_clicked)
    button10_.clicked.connect(openUnity)
    button11_.clicked.connect(openUsage)
    # 建立中間區域的四個小區塊
    middle_layout = QGridLayout(middle_widget)

    top_left_widget = QWidget()
    bottom_left_widget = QWidget()
    top_right_widget = QWidget()
    bottom_right_widget = QWidget()

    middle_layout.addWidget(top_left_widget, 0, 0)
    middle_layout.addWidget(bottom_left_widget, 1, 0)
    middle_layout.addWidget(top_right_widget, 0, 1)
    middle_layout.addWidget(bottom_right_widget, 1, 1)

    # 設定每個小區塊的背景顏色
    top_left_widget.setStyleSheet("background-color: lightblue;")
    bottom_left_widget.setStyleSheet("background-color: lightgreen;")
    top_right_widget.setStyleSheet("background-color: lightyellow;")
    bottom_right_widget.setStyleSheet("background-color: lightcoral;")

    # 建立按鈕並添加到各個小區塊的頂部
    top_left_layout = QVBoxLayout(top_left_widget)
    bottom_left_layout = QVBoxLayout(bottom_left_widget)
    top_right_layout = QVBoxLayout(top_right_widget)
    bottom_right_layout = QVBoxLayout(bottom_right_widget)

    button1 = QPushButton("開啟學員模型")
    button2 = QPushButton("開啟學員偵測")
    button3 = QPushButton("開啟教練模型")
    button4 = QPushButton("開啟教練偵測")
    font = QFont("標楷體", 16)
    button1.setFont(font)
    button2.setFont(font)
    button3.setFont(font)
    button4.setFont(font)


    # Unity button
    def embed_unity_window():
        unity_container.run_unity_program()
        time.sleep(2)
        unity_container.find_unity_window()
        unity_container.embed_unity_window()
    # Python button
    def embed_python_window():
        def run_python_program_thread():
            python_container.run_python_program()

        # Start 'run_python_program' as a separate thread
        python_thread = threading.Thread(target=run_python_program_thread)
        python_thread.start()
        time.sleep(12)
        python_container.find_python_window()
        python_container.embed_python_window()

    # Unity button
    def embed_unity_window1():
        unity_container1.run_unity_program()
        time.sleep(2)
        unity_container1.find_unity_window()
        unity_container1.embed_unity_window()

    # Python button
    def embed_python_window1():
        def run_python_program_thread():
            python_container1.run_python_program()

        # Start 'run_python_program' as a separate thread
        python_thread1 = threading.Thread(target=run_python_program_thread)
        python_thread1.start()
        time.sleep(12)
        python_container1.find_python_window()
        python_container1.embed_python_window()
    #--------------------------------------------
    # 調整佈局比例
    top_left_layout.addWidget(button1,1)
    bottom_left_layout.addWidget(button2,1)
    top_right_layout.addWidget(button3,1)
    bottom_right_layout.addWidget(button4,1)
    # 建立 Unity 容器
    unity_container = UnityContainer(top_left_widget)
    top_left_layout.addWidget(unity_container,9)
    button1.clicked.connect(embed_unity_window)
    # 建立 Python 容器
    python_container = PythonContainer(bottom_left_widget)
    bottom_left_layout.addWidget(python_container,9)
    button2.clicked.connect(embed_python_window)
    # 建立 Unity1 容器
    unity_container1 = UnityContainer1(top_right_widget)
    top_right_layout.addWidget(unity_container1,9)
    button3.clicked.connect(embed_unity_window1)
    # 建立 Python1 容器
    python_container1 = PythonContainer1(bottom_right_widget)
    bottom_right_layout.addWidget(python_container1,9)
    button4.clicked.connect(embed_python_window1)
    #--------------------------------------------
    timer = QTimer()
    timer.timeout.connect(update_motivation)
    timer.start(1000)
    # 設置主窗口的佈局
    window.setCentralWidget(main_widget)

    # 顯示視窗
    window.show()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()
