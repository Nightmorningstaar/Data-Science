#import os.file_path as osp

import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from tkinter import *
from tkinter import filedialog

def browseFiles():
	global file_path
	global last
	file_path = filedialog.askopenfilename()
	
	# Change label contents
    #print(file_path)
	label_file_explorer.configure(text="Click on the Run button then open the results folder!!")
	file_path_list = file_path.split(r"/")
	last = file_path_list[-1]

if __name__== "__main__":
    file_path = ""
    last = ""

    # Create the root window
    window = Tk()

    # Set window title
    window.title('File Explorer')

    # Set window size
    window.geometry("710x300")

    #Set window background color
    window.config(background = "white")

    # For constant window size
    window.resizable(0, 0)

    # Create a File Explorer label
    label_file_explorer = Label(window,
                                text = "File Explorer using Tkinter",
                                width = 100, height = 4,
                                fg = "blue")

        
    button_explore = Button(window,
                            text = "Browse Files",
                            command = browseFiles)


    label_file_explorer.grid(column = 1, row = 1)

    button_explore.grid(column = 1, row = 2)
    run_button = Button(window, text="Run", command=window.destroy)
    run_button.grid(column=1, row = 3)
    
    window.mainloop()

    model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
    # device = torch.device('cpu')

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    # read images
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #print(img)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(f'results/rlt_on{last}', output)
    