# import tkinter module
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
import detection

# import PIL module
from PIL import Image, ImageTk

# import json
import json


xpad = 5
ypad = 5

def analyze(filePath):
    # analyze the given image and report the inferences and confidence levels
    dict_result = detection.str_infer(filePath)
    str_result = json.dumps(dict_result)
    inference = detection.str_parse(str_result)
    confidence = detection.dict_parse(dict_result)

    # analyze the given image and report the inferences and confidence levels
    imgPath = detection.img_infer(filePath)

    # create and position inference label
    inference = ('Inference: ' + inference)
    l1 = Label(master, text=inference)
    l1.grid(row=0, column=0, padx=xpad, pady=ypad, sticky=NW)

    # create and position confidence label
    confidence = ('Confidence: ' + confidence)
    l2 = Label(master, text=confidence)
    l2.grid(row=0, column=1, padx=xpad, pady=ypad, sticky=N)


# creating main tkinter window/toplevel
master = Tk()
master.title("MRI Detector")

# select and image from the file explorer
filePath = askopenfilename()

# analyze the given image
analyze(filePath)

# set the image to the analyzed image
image = Image.open(imgPath)
photo = ImageTk.PhotoImage(image)
l3 = Label(image=photo)
l3.image = photo

# position the image
l3.grid(row=1, column=0, padx=xpad, pady=ypad, columnspan=3, sticky=EW)

# infinite loop which can be terminated by keyboard
# or mouse interrupt
mainloop()