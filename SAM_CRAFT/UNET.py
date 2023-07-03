import os
import numpy as np
from skimage.io import imread, imshow
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_PATH = '/content/drive/MyDrive/image/'
TRAIN_PATH_Y = '/content/drive/MyDrive/mask/'
TEST_PATH = '/content/drive/MyDrive/test/'
TEST_MASK_PATH = '/content/drive/MyDrive/test_mask/'

SAM_ROOT_PATH = 'F:/00UNET/SAM_CRAFT/'
SAM_TRAIN_PATH = SAM_ROOT_PATH+'SAM_TRAIN/'
SAM_TRAIN_MASK_PATH = SAM_ROOT_PATH+'SAM_MASK/'
SAM_TEST_PATH = SAM_ROOT_PATH+'SAM_TEST/'
SAM_VALIDATION_PATH = SAM_ROOT_PATH + 'SAM_VALIDATION/'
SAM_VALIDATION_MASK_PATH = SAM_ROOT_PATH + 'SAM_VALIDATION_MASK/'
SAM_NEWTEST_PATH = SAM_ROOT_PATH + 'SAM_NEW_TEST/'
SAM_NEWTEST_MASK_PATH = SAM_ROOT_PATH + 'SAM_NEW_TEST_MASK/'

train_ids = next(os.walk(SAM_TRAIN_PATH))[2]
test_ids = next(os.walk(SAM_VALIDATION_PATH))[2]
test_ids.sort()
train_ids.sort()
label_ids = next(os.walk(SAM_TRAIN_MASK_PATH))[2]
label_ids.sort()
test_mask_ids = next(os.walk(SAM_VALIDATION_MASK_PATH))[2]
test_mask_ids.sort()
SAM_test_ids = next(os.walk(SAM_TEST_PATH))[2]
SAM_test_ids.sort()
SAM_NEWtest_ids = next(os.walk(SAM_NEWTEST_PATH))[2]
SAM_test_ids.sort()
SAM_NEWtestMASK_ids = next(os.walk(SAM_NEWTEST_MASK_PATH))[2]
SAM_test_ids.sort()

X_train = np.zeros((len(train_ids),128,128),dtype = np.uint8)
for n,id_ in enumerate(train_ids):
    path = SAM_TRAIN_PATH + id_
    # print(path)
    img = imread(path)
    img=img[:,:,0]
    X_train[n] = img

Y_train = np.zeros((len(label_ids),128,128),dtype=bool)
for n,id_ in enumerate(label_ids):
    label_path = SAM_TRAIN_MASK_PATH + id_
    # print(label_path)
    mask_img = imread(label_path,1)
    ##通道为1
    Y_train[n] = mask_img

TEST = np.zeros((len(test_ids),128,128),dtype = np.uint8)
for n,id_ in enumerate(test_ids):
    test_path = SAM_VALIDATION_PATH + id_
    # print(test_path)
    test_img = imread(test_path)

    TEST[n] = np.mean(test_img, axis = 2)

TEST_MASK = np.zeros((len(test_mask_ids),128,128),dtype = np.uint8)
for n,id_ in enumerate(test_mask_ids):
    test_mask_path = SAM_VALIDATION_MASK_PATH + id_
    # print(test_path)
    test_mask_img = imread(test_mask_path)
    test_mask_img=test_mask_img[:,:,0]
    TEST_MASK[n] = test_mask_img

# Construction

inputs = tf.keras.layers.Input((128,128,1))
s = tf.keras.layers.Lambda(lambda x:x/255)(inputs)

c1 = tf.keras.layers.Conv2D(16,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(s)
c1 = tf.keras.layers.Conv2D(16,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(p1)
c2 = tf.keras.layers.Conv2D(32,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(p2)
c3 = tf.keras.layers.Conv2D(64,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(p3)
c4 = tf.keras.layers.Conv2D(128,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(p4)
c5 = tf.keras.layers.Conv2D(256,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(c5)


u6 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding ='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
u6 = tf.keras.layers.Conv2D(128,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(u6)
u6 = tf.keras.layers.Conv2D(128,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(u6)

u7 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding ='same')(u6)
u7 = tf.keras.layers.concatenate([u7,c3])
u7 = tf.keras.layers.Conv2D(64,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(u7)
u7 = tf.keras.layers.Conv2D(64,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(u7)

u8 = tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding ='same')(u7)
u8 = tf.keras.layers.concatenate([u8,c2])
u8 = tf.keras.layers.Conv2D(32,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(u8)
u8 = tf.keras.layers.Conv2D(32,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(u8)

u9 = tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding ='same')(u8)
u9 = tf.keras.layers.concatenate([u9,c1])
u9 = tf.keras.layers.Conv2D(16,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(u9)
u9 = tf.keras.layers.Conv2D(16,(3,3),activation ='relu',kernel_initializer ='he_normal',padding = 'same')(u9)

output = tf.keras.layers.Conv2D(1,(1,1),activation = 'sigmoid')(u9)

model = tf.keras.Model(inputs = [inputs],outputs = [output])
model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = 'accuracy')
'''
model1 = tf.keras.Model(inputs = [inputs],outputs = [output])
model1.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = 'accuracy')
prediction_train = model1.predict(TEST,verbose = 1)
prediction_train_t = (prediction_train<0.5).astype(np.uint8)
miou = compute_miou(prediction_train_t, TEST_MASK)
conF1 = compute_conF1(prediction_train_t, TEST_MASK)
print("mIOU:", miou)
print("F1:", conF1)
'''


model.summary()

results = model.fit(X_train,Y_train, batch_size = 8, epochs = 15)

model.save('SAM_CRAFT/RESULT/Unet_Model_2023-6-24_PURE_SAM2.h5')
# model = tf.keras.models.load_model('SAM_CRAFT/RESULT/Unet_model-2023-05-13-35样本.h5')

#
prediction_train = model.predict(TEST,verbose = 1)
prediction_train_t = (prediction_train<0.5).astype(np.uint8)

mask_threshold = 12
TEST_MASK = (TEST_MASK<mask_threshold).astype(np.uint8)

def compute_iou(pred_mask, true_mask):
    pred_mask = pred_mask[:,:,0]
    intersection = pred_mask & true_mask
    union = pred_mask | true_mask
    if np.sum(union)!=0 :
      iou = np.sum(intersection) / np.sum(union)
    else:
      iou=1
    return iou

def compute_miou(pred_masks, true_masks):
    num_classes = pred_masks.shape[0]
    ious = np.zeros(num_classes)
    for i in range(num_classes):
        ious[i] = compute_iou(pred_masks[i], true_masks[i])
    miou = np.mean(ious)
    return miou

def compute_F1(pred_mask, true_mask):
    pred_mask = pred_mask[:,:,0]
    TP = np.sum(pred_mask & true_mask)
    FN = np.sum(~pred_mask & true_mask)
    FP = np.sum(pred_mask & ~true_mask)
    if (TP+FN)==0:
        recall = 1
    else:
        recall = TP/(TP+FN)
    if (TP+FP)==0:
        pc = 1
    else:
        pc = TP/(TP+FP)
    if (recall+pc)==0:
        F1 = 0
    else:
        F1 = 2* recall*pc/(recall+pc)
    return F1

def compute_conF1(pred_masks, true_masks):
    num_classes = pred_masks.shape[0]
    u = np.zeros(num_classes)
    for i in range(num_classes):
        u[i] = compute_F1(pred_masks[i], true_masks[i])
    cF1 = np.mean(u)
    return cF1

def compute_ac(pred_mask, true_mask):
    pred_mask = pred_mask[:,:,0]
    TP = np.sum(pred_mask & true_mask)
    TN = np.sum(~pred_mask & ~true_mask)
    FN = np.sum(~pred_mask & true_mask)
    FP = np.sum(pred_mask & ~true_mask)
    ac = (TP+TN)/(TP+TN+FN+FP)
    return ac

def count_non_one(arr):
    non_one_count = np.count_nonzero(arr != 0)
    return non_one_count

def compute_cac(pred_masks, true_masks):

    num_classes = pred_masks.shape[0]
    u = np.zeros(num_classes)
    for i in range(num_classes):
        num = compute_ac(pred_masks[i], true_masks[i])
        if num==1:
            u[i]=0
        else:
            u[i] = num
    cac = np.sum(u)/count_non_one(u)
    return cac

miou = compute_miou(prediction_train_t, TEST_MASK)
conF1 = compute_conF1(prediction_train_t, TEST_MASK)
# ac = compute_cac(prediction_train_t, TEST_MASK)
print("mIOU:", miou)
# print("ac", ac)
print("F1:", conF1)
#
'''
modelraw = tf.keras.Model(inputs = [inputs],outputs = [output])
modelraw.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = 'accuracy')
raw_prediction = modelraw.predict(TEST,verbose = 1)
raw_prediction_t = (raw_prediction<0.5).astype(np.uint8)
miou = compute_miou(raw_prediction_t, TEST_MASK)
conF1 = compute_conF1(raw_prediction_t, TEST_MASK)
print("mIOU:", miou)
print("F1:", conF1)
'''


SAM_TEST_ = np.zeros((len(SAM_test_ids),128,128),dtype = np.uint8)
for n,id_ in enumerate(SAM_test_ids):
    test_path = SAM_TEST_PATH + id_
    # print(test_path)
    test_img = imread(test_path)
    SAM_TEST_[n] = np.mean(test_img, axis = 2)

prediction_test = model.predict(SAM_TEST_,verbose = 1)

save_NEWTRAIN_file = 'F:/00UNET/SAM_CRAFT/SAM_NEW_TEST/'
save_NEWTRAIN_MASK_file = 'F:/00UNET/SAM_CRAFT/SAM_NEW_TEST_MASK/'

prediction_test_t = (prediction_test<0.5).astype(np.uint8)
prediction_test_t = np.squeeze(prediction_test_t)

for n,i in enumerate(SAM_TEST_):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(SAM_TEST_[n])
    ax1.set_title('TEST')
    ax2.imshow(prediction_test_t[n])
    ax2.set_title('PRE')
    plt.tight_layout()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

    user_input = input("同意保存这两张图片吗？(y/n): ")
    if user_input.lower() == 'y':
        img=Image.fromarray(SAM_TEST_[n])
        img.save(save_NEWTRAIN_file+str(n)+'.png')
        maskimg=Image.fromarray((1-prediction_test_t[n])*255)
        maskimg.save(save_NEWTRAIN_MASK_file+str(n)+'-mask.png')
        plt.close()
        continue
    elif user_input.lower() == 'q':
        plt.close()
        break
    else:
        print("图片未保存。")
    plt.close()

SAM_NEWtest_ids = next(os.walk(SAM_NEWTEST_PATH))[2]
SAM_NEWtest_ids.sort()
SAM_NEWtestMASK_ids = next(os.walk(SAM_NEWTEST_MASK_PATH))[2]
SAM_NEWtestMASK_ids.sort()

X_train1 = np.zeros((len(SAM_NEWtest_ids),128,128),dtype = np.uint8)
for n,id_ in enumerate(SAM_NEWtest_ids):
    path = SAM_NEWTEST_PATH + id_
    # print(path)
    img = imread(path)
    X_train1[n] = img

Y_train1 = np.zeros((len(SAM_NEWtestMASK_ids),128,128),dtype=bool)
for n,id_ in enumerate(SAM_NEWtestMASK_ids):
    label_path = SAM_NEWTEST_MASK_PATH + id_
    # print(label_path)
    mask_img = imread(label_path,1)
    ##通道为1
    Y_train1[n] = mask_img
# validation_split=0.2

results = model.fit(X_train1,Y_train1, batch_size = 4, epochs = 15)
test2 = model.predict(TEST,verbose = 1)
test_t2 = (test2<0.6).astype(np.uint8)
test_t2 = np.squeeze(test_t2)
miou = compute_miou(test_t2, TEST_MASK)
print("mIOU:", miou)

model.save('SAM_CRAFT/RESULT/Unet_6-24_PURE_SAM3--'+str(miou.round(3))+'.h5')

'''

PRED_INPUT = "F:/00UNET/CroppingCraft/result/"
PRED_OUTPUT = "F:/00UNET/CroppingCraft/result_pred/"
PRED_INPUT_ids = next(os.walk(PRED_INPUT))[2]
PRED_INPUT_ids.sort()

PRED_RAW = np.zeros((len(PRED_INPUT_ids),128,128),dtype = np.uint8)
for n,id_ in enumerate(PRED_INPUT_ids):
    path = PRED_INPUT + id_
    # print(path)
    img = imread(path)

    PRED_RAW[n] = img

FINAL_PRED = model.predict(PRED_RAW,verbose = 1)
FINAL_PRED = (FINAL_PRED<0.5).astype(np.uint8)
FINAL_PRED = np.squeeze(FINAL_PRED )

for n,i in enumerate(FINAL_PRED):
    img=Image.fromarray(FINAL_PRED[n]*255)
    img.save(PRED_OUTPUT+PRED_INPUT_ids[n],quality=100)
    
'''

'''

PRED_INPUT = "F:/00UNET/CroppingCraft/result-nowindows/"
PRED_OUTPUT = "F:/00UNET/CroppingCraft/result-nowindows-pred/"
PRED_INPUT_ids = next(os.walk(PRED_INPUT))[2]
PRED_INPUT_ids.sort()

PRED_RAW = np.zeros((len(PRED_INPUT_ids),128,128),dtype = np.uint8)
for n,id_ in enumerate(PRED_INPUT_ids):
    path = PRED_INPUT + id_
    # print(path)
    img = imread(path)
    PRED_RAW[n] = img

FINAL_PRED = model.predict(PRED_RAW,verbose = 1)
FINAL_PRED = (FINAL_PRED<0.5).astype(np.uint8)
FINAL_PRED = np.squeeze(FINAL_PRED )

for n,i in enumerate(FINAL_PRED):
    img=Image.fromarray(FINAL_PRED[n]*255)
    img.save(PRED_OUTPUT+PRED_INPUT_ids[n],quality=100)

'''

