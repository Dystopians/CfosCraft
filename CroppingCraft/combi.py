
output_path = "F:/00UNET/CroppingCraft/result_pred/"
output_path2 = "F:/00UNET/CroppingCraft/result_pred_conbined/"
from PIL import Image
window_size = 128
step_size = 120
crop_size=step_size
import numpy as np
import os
ids = next(os.walk(output_path))[2]
ids.sort()
# imgs = np.zeros((len(ids),step_size,step_size,3),dtype = np.uint8)
imgs = np.zeros((len(ids),step_size,step_size),dtype = np.uint8)
img_id = ['']*len(ids)
img_max_w = []
width= window_size
height=window_size

for n,id_ in enumerate(ids):
    path = output_path + id_
    # print(path)
    img = Image.open(path)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    cropped_image = img.crop((left, top, right, bottom))

    imgs[n] = cropped_image
    img_id[n]=id_.split('.')[0]
    img_max_w.append(int(img_id[n].split('_')[1]))

row = max(img_max_w)
column = (len(ids)/(row+1))-1
comsize = (int((column+1)*step_size),int((row+1)*step_size))
# target_image = Image.new("RGB", comsize)
target_image = Image.new("L", comsize)

for n,img in enumerate(imgs):
    i_row = int(img_id[n].split('_')[0])*step_size
    i_col = int(img_id[n].split('_')[1])*step_size
    posi = (i_row,i_col,i_row+step_size,i_col+step_size)
    img = Image.fromarray(img,'L')
    target_image.paste(img,posi)

target_image.show()
target_image.save(output_path2+"recombi.jpg")
print(max(img_max_w))
print(len(ids))


'''
output_path = "F:/00UNET/CroppingCraft/result-nowindows-pred/"
output_path2 = "F:/00UNET/CroppingCraft/result_pred_conbined/"
from PIL import Image
window_size = 128
step_size = 128
crop_size=step_size
import numpy as np
import os
ids = next(os.walk(output_path))[2]
ids.sort()
# imgs = np.zeros((len(ids),step_size,step_size,3),dtype = np.uint8)
imgs = np.zeros((len(ids),step_size,step_size),dtype = np.uint8)
img_id = ['']*len(ids)
img_max_w = []
width= window_size
height=window_size

for n,id_ in enumerate(ids):
    path = output_path + id_
    # print(path)
    img = Image.open(path)
    cropped_image = img

    imgs[n] = cropped_image
    img_id[n]=id_.split('.')[0]
    img_max_w.append(int(img_id[n].split('_')[1]))

row = max(img_max_w)
column = (len(ids)/(row+1))-1
comsize = (int((column+1)*step_size),int((row+1)*step_size))
# target_image = Image.new("RGB", comsize)
target_image = Image.new("L", comsize)

for n,img in enumerate(imgs):
    i_row = int(img_id[n].split('_')[0])*step_size
    i_col = int(img_id[n].split('_')[1])*step_size
    posi = (i_row,i_col,i_row+step_size,i_col+step_size)
    img = Image.fromarray(img,'L')
    target_image.paste(img,posi)

target_image.show()
target_image.save(output_path2+"recombi-nowindow.jpg")
print(max(img_max_w))
print(len(ids))

'''