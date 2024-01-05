# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

data = '/content/drive/MyDrive/RC/txt_mudit_b11_11597.jpg'

import pandas as pd
import cv2

!unzip RC.zip

import cv2
import os
import glob
img_dir = "/content/RC"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/pytesseract'

# Just needed in case you'd like to append it to an array
import os
data = []

for filename in os.listdir('/content/RC'):
        data.append(filename)

import glob
images = glob.glob('/content/RC' + '/*.jpg')

!pip install opencv-python
!pip install pytesseract
!sudo apt install tesseract-ocr

from google.colab.patches import cv2_imshow

# importing modules
import cv2
import pytesseract
from google.colab.patches import cv2_imshow
from pytesseract import Output

# reading image using opencv

image = cv2.imread(images[0])

#converting image into gray scale image

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#converting it to binary image by Thresholding

# this step is require if you have colored image because if you skip this part

# then tesseract won't able to detect text correctly and this will give incorrect result

threshold_img = cv2.threshold(gray_image, 90, 255, cv2.THRESH_TOZERO)[1]

# display image

cv2_imshow(threshold_img)

# Maintain output window until user presses a key

cv2.waitKey(0)

# Destroying present windows on screen

cv2.destroyAllWindows()

#configuring parameters for tesseract
from pytesseract import Output
custom_config = r'--oem 3 --psm 6'

# now feeding image to tesseract

details = pytesseract.image_to_data(threshold_img,output_type=Output.DICT, config=custom_config, lang= 'eng')
print(details.keys())

total_boxes = len(details['text'])

for sequence_number in range(total_boxes):
  if int(details['conf'][sequence_number]) >30:
    (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
    threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# display image

cv2_imshow(threshold_img)

# Maintain output window until user presses a key

cv2.waitKey(0)

# Destroying present windows on screen

cv2.destroyAllWindows()

details['text']

parse_text = []

word_list = []

last_word = ''

for word in details['text']:
  if word!='':
    word_list.append(word)
    last_word = word
  if (last_word!='' and word == '') or (word==details['text'][-1]):
    parse_text.append(word_list)
    word_list = []

details['text'][-1]

import csv

with open('result_text.xls', 'w', newline="") as file:
  csv.writer(file, delimiter=" ").writerows(parse_text)

# importing modules
import csv
import cv2
import pytesseract
from google.colab.patches import cv2_imshow
from pytesseract import Output

def text_extraction(pic):
# reading image using opencv
  image = cv2.imread(pic)

  #converting image into gray scale image

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  #converting it to binary image by Thresholding

  # this step is require if you have colored image because if you skip this part

  # then tesseract won't able to detect text correctly and this will give incorrect result

  threshold_img = cv2.threshold(gray_image, 90, 255, cv2.THRESH_TOZERO)[1]

  # Maintain output window until user presses a key

  cv2.waitKey(0)

  # Destroying present windows on screen

  cv2.destroyAllWindows()

  custom_config = r'--oem 3 --psm 6'

  # now feeding image to tesseract

  details = pytesseract.image_to_data(threshold_img,output_type=Output.DICT, config=custom_config, lang= 'eng')

  total_boxes = len(details['text'])

  for sequence_number in range(total_boxes):
    if int(details['conf'][sequence_number]) >30:
      (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
      threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

  # Maintain output window until user presses a key

  cv2.waitKey(0)

  # Destroying present windows on screen

  cv2.destroyAllWindows()

  parse_text = []

  word_list = []

  ch_no = []

  last_word = ''

  for word in details['text']:
    if word!='':
      word_list.append(word)
      last_word = word
    if (last_word!='' and word == '') or (word==details['text'][-1]):
      parse_text.append(word_list)
      word_list = []
    if (word!='' and len(word) == 17):
      ch_no.append(word)


  with open('result_text.txt', 'w', newline="") as file:
    csv.writer(file, delimiter=" ").writerows(parse_text)

text_extraction(data[0])

for imageName in os.listdir('/content/RC'):
  inputPath = os.path.join('/content/RC', imageName)
  image = cv2.imread(inputPath)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  threshold_img = cv2.threshold(gray_image, 90, 255, cv2.THRESH_TOZERO)[1]
  custom_config = r'--oem 3 --psm 6'
  details = pytesseract.image_to_data(threshold_img,output_type=Output.DICT, config=custom_config, lang= 'eng')
  total_boxes = len(details['text'])
  for sequence_number in range(total_boxes):
    if int(details['conf'][sequence_number]) >30:
      (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
      threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

  parse_text = []
  word_list = []
  ch_no = []

  last_word = ''

  for word in details['text']:
    if word!='':
      word_list.append(word)
      last_word = word
    if (last_word!='' and word == '') or (word==details['text'][-1]):
      parse_text.append(word_list)
      word_list = []
    if (word!='' and len(word) == 17):
      ch_no.append(word)

  with open('result_text.txt', 'w', newline="") as file:
    csv.writer(imageName+"\n")
    csv.writer(file, delimiter=" ").writerows(parse_text)

cv2_imshow(image1)
