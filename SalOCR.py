"""## Dependencies"""

import os
import cv2
import json
import math
import time
import easyocr
import statistics
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

# CLI Interface
# =====================================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imagefilepath", default="Data/Dataset/images/", help="directories where images are received as input, default is set as 'Data/Dataset/images/'.")
parser.add_argument("--maskfilepath", default="Predictions/Dataset/RGB_VST/", help="directories where images are received as input, default is set as 'Predictions/Dataset/RGB_VST/'.")
parser.add_argument("--outputfilepath", default="TextOutput/", help="directories where images are saved(or overwrites) as output, default is set as 'TextOutput/'.")
parser.add_argument("--pdf", action="store_true", help="Choose to generate a pdf displaying program decision making. Useful for troubleshooting.")
parser.add_argument("--salthresh", default=0.35, type=restricted_float, help="Saliency Threshold which determines OCR acceptance or rejection of text as salient. Default is 0.35, higher values correspond to stricter saliency criterions")
args = parser.parse_args()

"""## Paddle,Easy OCR Lone Inference Function"""
# Loading easyocr model into memory
EasyReader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

# Initializing Args
data_filepath = args.imagefilepath
mask_filepath = args.maskfilepath
OCR_output_path = args.outputfilepath
show_images = True
pdf_images = args.pdf
save_text = True
# Dynamic Variables
easy_sal_confi_thresh = args.salthresh


# Initializing filepaths 
image_filenames = os.listdir(data_filepath)
mask_filenames = []
for name in image_filenames:
    mask_filenames += [mask_filepath + name.replace(".jpg",".png")]
for i in range(len(image_filenames)):
    image_filenames[i] = data_filepath + image_filenames[i]

N = len(image_filenames)
total_text = {}



# Main
#========================
if N < 5 and show_images:
    show_images = False
if pdf_images == False:
    show_images == False

if show_images:
    fig, axes = plt.subplots(math.ceil(N/5), 5,figsize=(25, 4 * int(math.ceil(N/5))))
    ax = axes.ravel()

for image_i in tqdm(range(N)):
    # Initializing number of segments in the text from segmentation
    image_id = image_filenames[image_i][len(data_filepath):-4]
    counter = 0

    # Initialize images, masks
    im = cv2.imread(image_filenames[image_i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_height, im_width, rgb_constant = im.shape
    mask = cv2.imread(mask_filenames[image_i])
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Display Saliency Mask Boxes
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0,0,255), -1)

    # Perform Lone OCR Inference function
    image_text = {}
    easy_results = EasyReader.readtext(image_filenames[image_i])

    # Easy OCR
    easy_raw_text, easy_sal_text = "",""
    for token in easy_results:
        Bboxes,text,score = token
        easy_raw_text += text + " "
        p1,p2,p3,p4 = Bboxes            
        minpoint = [min(p1[0],p2[0],p3[0],p4[0]),min(p1[1],p2[1],p3[1],p4[1])]      # Determining anchorpoints
        maxpoint = [max(p1[0],p2[0],p3[0],p4[0]),max(p1[1],p2[1],p3[1],p4[1])]
        minpoint = [min(minpoint[0],im_width-1), min(minpoint[1],im_height-1)]
        maxpoint = [min(maxpoint[0],im_width-1), min(maxpoint[1],im_height-1)]
        minpoint = [max(minpoint[0],0), max(minpoint[1],0)]
        maxpoint = [max(maxpoint[0],0), max(maxpoint[1],0)]
        #print(minpoint, maxpoint)
        truth_ctr, false_ctr = 0,0
        for i in range(int(minpoint[1]),int(maxpoint[1])):
            for j in range(int(minpoint[0]),int(maxpoint[0])):
                if list(im[i][j]) == [0,0,255]:
                    truth_ctr += 1
                else:
                    false_ctr += 1
        # Salience confidence thresholding
        sal_confidence = truth_ctr / (truth_ctr + false_ctr + 1)
        #print("easy: ",sal_confidence)
        if(sal_confidence >= easy_sal_confi_thresh):
            cv2.rectangle(im, (int(p1[0]),int(p1[1])), (int(p3[0]),int(p3[1])), (57,255,20),3)
            easy_sal_text += text + " "
        else:
            cv2.rectangle(im, (int(p1[0]),int(p1[1])), (int(p3[0]),int(p3[1])), (255,49,49),3)
        # Saving Texts as dictionary, then json
    image_text["Raw"] = easy_raw_text
    image_text["Sal"] = easy_sal_text

    total_text[image_filenames[image_i][len(data_filepath):]] = image_text



    #print(" ") 
    # Outprocessing
    if show_images:
        #axes[math.floor(image_i/5),image_i%5].imshow(Image.fromarray(imOCR), cmap="gray", alpha=0.5)
        #axes[math.floor(image_i/5),image_i%5].imshow(thresh)
        axes[math.floor(image_i/5),image_i%5].imshow(im, cmap="gray", alpha=0.85)

if show_images:
    for a in ax:
        a.set_axis_off()
# Outprocessing (cont.)
if save_text:       # Encode dictionaries into JSON

    if not os.path.exists(OCR_output_path):
        os.makedirs(OCR_output_path)

    with open(OCR_output_path + "ExtractedText.json", "w") as fp:
        json.dump(total_text, fp) 
if pdf_images:
    fig.savefig('SaliencyVisualization.pdf', format='pdf', dpi=600)