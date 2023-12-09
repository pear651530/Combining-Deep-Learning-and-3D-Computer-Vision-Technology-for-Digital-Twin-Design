from keras.models import load_model
from PIL import Image, ExifTags
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PySimpleGUI as psg
import io
import time
import os
import subprocess

img_size = (256, 256)


def get_mask(file_name):
    for i in range(0, 31):
        time.sleep(0.05)
        progress_text.update(f'{i}%')
        progress_bar.update(i)
            
    # 加载模型
    model = load_model('230617_unet_densenet.h5')
    
    progress_text.update('50%')
    progress_bar.update(50)
    
    file = file_name
    img = Image.open(file)
    try:
        for orientation in ExifTags.TAGS.keys() : 
            if ExifTags.TAGS[orientation]=='Orientation' : break 
        exif=dict(img._getexif().items())
        if   exif[orientation] == 3 : 
            img=img.rotate(180, expand = True)
        elif exif[orientation] == 6 : 
            img=img.rotate(270, expand = True)
        elif exif[orientation] == 8 : 
            img=img.rotate(90, expand = True)
    except:
        pass
    img.save(file) 
    
    path = file_name
    img0 = load_img(path, target_size=(512, 512), interpolation="bicubic")
    img1 = load_img(path, target_size=img_size, interpolation="bicubic")
    val_preds = model.predict(np.expand_dims(img1.copy(), axis= 0)) #(1,256,256,3)
    img_pred = val_preds.squeeze()
    
    # 顯示遮罩(mask)
    def display_mask():
        # 显示一个模型预测的快速工具
        """Quick utility to display a model's prediction."""
        mask = np.argmax(img_pred, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.where(mask == 1, 255, 0)
        mask = np.expand_dims(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST), axis= -1)
        cv2.imwrite("./UNet_output/ryota_mask.png", mask)

        mask_img = np.zeros((512, 512) + (3,), dtype="uint8")
        mask_img = np.where(mask == 255, img0, (0, 0, 0))
        mask_img = mask_img.astype(np.uint8)

        # 将颜色通道顺序从RGB转换为BGR
        bgr_image = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite("./UNet_output/ryota.png", bgr_image)

    # 預測
    display_mask()

    for i in range(71, 101):
        time.sleep(0.05)
        progress_text.update(f'{i}%')
        progress_bar.update(i)
    return  


##erosion
def del_white(file_name):
    # 讀取影像
    image = cv2.imread('./UNet_output/ryota_mask.png', 0)

    kernel_sizes = [(3, 3), (3, 5), (5, 5), (5, 7), (3, 7), (7, 7), (5, 9), (7, 9), (9, 9), (7, 11), (9, 11), (11, 11)]
    plt.figure(figsize=(20,10))
    show_arr=[]

    for i, chosen_kernel_size in enumerate(kernel_sizes):
        image_copy = image.copy()
        # 定义侵蚀的结构元素（kernel）
        kernel = np.ones(chosen_kernel_size, np.uint8)

        # 执行 erosion（侵蚀）操作
        erosion_image = cv2.erode(image_copy, kernel)

        path = file_name
        img0 = load_img(path, target_size=(512, 512), interpolation="bicubic")
        mask = erosion_image
        mask = np.expand_dims(mask, axis= -1)
        mask_img = np.zeros((512, 512) + (3,), dtype="uint8")
        mask_img = np.where(mask == 255, img0, (0, 0, 0))
        mask_img = mask_img.astype(np.uint8)
        # 将颜色通道顺序从RGB转换为BGR
        #bgr_image = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)

        #plt.subplot(2,3,i+1)
        #plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        #plt.title(chosen_kernel_size)
        #plt.axis('off')
        show_arr.append(mask_img)
        
    return show_arr


##dilation (如果白色人體部分有小黑塊時再使用
def del_black():
    # 讀取影像
    image = cv2.imread('./UNet_output/ryota_mask.png', 0)
    # 定義膨脹的結構元素（kernel）
    kernel = np.ones((5, 5), np.uint8)
    # 執行 dilation（膨脹）操作
    dilated_image = cv2.dilate(image, kernel)
    cv2.imwrite("./UNet_output/ryota_mask.png", dilated_image)


# pifu
def pifu():
    # 設置CUDA_VISIBLE_DEVICES環境變數
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    # 執行命令
    subprocess.call("python ./PIFu-master/apps/eval.py --name pifu_demo --batch_size 1 --mlp_dim 257 1024 512 256 128 1 --mlp_dim_color 513 1024 512 256 128 3 --num_stack 4 --num_hourglass 2 --resolution 256 --hg_down ave_pool --norm group --norm_color group --test_folder_path ./PIFu-master/sample_images --load_netG_checkpoint_path ./PIFu-master/checkpoints/net_G --load_netC_checkpoint_path ./PIFu-master/checkpoints/net_C", shell=True)

    return


# blender
def blender():
    # 切換目錄
    os.chdir("BlenderBone/Blender 3.5")
    # 執行命令
    subprocess.run(["blender.exe", "-P", "../BlenderScripty.py"])
    # 返回原本的目錄
    os.chdir("../..")

    return

def open_blender():
    pdf_path = './GetModel的骨架建立使用手冊.pdf'
    subprocess.Popen(['start', '', pdf_path], shell=True)

    return


#GUI
layout = [
    [psg.Text("請選擇一個文件：")],
    [psg.InputText(key='file_path'), psg.FileBrowse(key='browse_button')],
    [psg.Button('提交')],
    #[psg.Text('', key='combination')],
    [psg.Image(data='', key='image', size=(512,512), enable_events=True)],
    [psg.Text('拖曳下方滑塊，選擇想要的去被圖效果', key='note', visible=False)],
    [psg.Slider(range=(0, 11), default_value=0,
    expand_x=True, enable_events=True,
    orientation='horizontal', key='-SL-', visible=False)],
    [psg.Button('就要這張圖!', visible=False)],
    [psg.Button('建模', visible=False)],
    [psg.Button('去Blender加骨架', visible=False), psg.Button('開啟BlenderScripty使用手冊', visible=False)],
    [psg.Exit()]
]

showarr = []

window = psg.Window('去背!', layout, size=(700, 850), finalize=True)
combination_sizes = [(3, 3), (3, 5), (5, 5), (5, 7), (3, 7), (7, 7), (5, 9), (7, 9), (9, 9), (7, 11), (9, 11), (11, 11)]
while True:
    event, values = window.read()
    if event == psg.WIN_CLOSED or event == 'Exit':
      break
    if event == '提交':
        # 切換按鈕的啟用/禁用狀態
        #window['提交'].update(disabled=True)
        
        file_path = values['file_path']
        if file_path:
            psg.popup(f'你選擇了文件：{file_path}', title='文件路徑')
            
        # 打开进度窗口
        progress_layout = [
            [psg.Text('正在處理...')],
            [psg.ProgressBar(100, orientation='h', size=(20, 20), key='-PROGRESS-')],
            [psg.Text('0%', key='-PROGRESS_TEXT-')]
        ]
        progress_window = psg.Window('進度窗口', progress_layout, finalize=True, modal=True, no_titlebar=True)
        progress_bar = progress_window['-PROGRESS-']
        progress_text = progress_window['-PROGRESS_TEXT-']
    
        show_img = get_mask(file_path)
        progress_window.close()  # 关闭进度窗口
        del progress_window
        
        #查看效果
        jump_img_layout = [
            [psg.Text('現在的去背效果:')],
            [psg.Image(filename="./UNet_output/ryota.png", key='jump_img', size=(512,512))],
            [psg.Text('人像上有小黑塊嗎?:')],
            [psg.Button('什麼是小黑塊?')],
            [psg.Button('有'), psg.Button('沒有')]
        ]
        jump_img_window = psg.Window('目前結果檢查', jump_img_layout, finalize=True, modal=True)
        while True:
            jump_img_event, jump_img_values = jump_img_window.read()
            if jump_img_event == '什麼是小黑塊?':
                jump_img_black_layout = [
                    [psg.Text('小黑塊就是在彩色的人像上，有冒出黑色像素，如以下圖示:')],
                    [psg.Image(filename="./black_example.png", key='jump_img_black', size=(512,512))]
                ]
                jump_img_black_window = psg.Window('什麼是小黑塊?', jump_img_black_layout, finalize=True, modal=True)
                jump_img_black_event, jump_img_black_values = jump_img_black_window.read()
                if jump_img_black_event == psg.WIN_CLOSED:
                    jump_img_black_window.close()
                    del jump_img_black_layout
            if jump_img_event == '有':
                del_black()
                showarr = del_white(file_path)
                jump_img_window.close()
                del jump_img_window
                break

            if jump_img_event == '沒有' or jump_img_event == psg.WIN_CLOSED:
                showarr = del_white(file_path)
                jump_img_window.close()
                del jump_img_window
                break
        
        selected_index = 0
        img_data = showarr[selected_index]
        img_bytes = io.BytesIO()
        img = Image.fromarray(img_data)
        img.save(img_bytes, format='PNG')
        
        window['image'].update(data=img_bytes.getvalue())
        window['-SL-'].update(visible=True)
        window['就要這張圖!'].update(visible=True)
        window['note'].update(visible=True)
    if event == '-SL-':
        selected_index = int(values['-SL-'])
        img_data = showarr[selected_index]
        img_bytes = io.BytesIO()
        img = Image.fromarray(img_data)
        img.save(img_bytes, format='PNG')
        window['image'].update(data=img_bytes.getvalue())
        #window['combination'].update(combination_sizes[selected_index])
    if event == '就要這張圖!':
        selected_index = int(values['-SL-'])
        a = combination_sizes[selected_index][0]
        b = combination_sizes[selected_index][1]
        image = cv2.imread('./UNet_output/ryota_mask.png', 0)
        kernel = np.ones((a,b), np.uint8)
        erosion_image = cv2.erode(image, kernel)
        cv2.imwrite("./PIFu-master/sample_images/ryota_mask.png", erosion_image)
        #再次存檔輸出
        path = values['file_path']
        img0 = load_img(path, target_size=(512, 512), interpolation="bicubic")
        mask = cv2.imread('./PIFu-master/sample_images/ryota_mask.png', cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis= -1)
        mask_img = np.zeros((512, 512) + (3,), dtype="uint8")
        mask_img = np.where(mask == 255, img0, (0, 0, 0))
        mask_img = mask_img.astype(np.uint8)

        # 将颜色通道顺序从RGB转换为BGR
        bgr_image = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./PIFu-master/sample_images/ryota.png", bgr_image)
        
        psg.popup("儲存完成!", auto_close = True, auto_close_duration = 1)
        window['建模'].update(visible=True)
    if event == '建模':
        pifu()
        psg.popup("建模完成!\n請從./PIFu-master/results/pifu_demo打開obj檔查看結果", title="建模完成！", auto_close = True, auto_close_duration = 2)
        window['去Blender加骨架'].update(visible=True)
        window['開啟BlenderScripty使用手冊'].update(visible=True)
    if event == '去Blender加骨架':
        blender()
    if event == '開啟BlenderScripty使用手冊':
        open_blender()
        
window.close()