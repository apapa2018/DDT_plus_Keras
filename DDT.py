import argparse
import numpy as np

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os
from config1 import batch,  classes,DDTtrain,DDTimage,DDT_savedir,savepath
from keras.layers import  MaxPooling2D, Dense,GlobalMaxPooling2D
from keras.applications.xception import Xception,preprocess_input
from keras.models import Model

from skimage import measure
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"]="1"
size = 224

class DDTplus():
   def __init__(self):

      self.input_shape = (size, size, 3)
      #self.weight = savepath


   def CreateModel(self):
      '''
      构建imagenet训练参数的Xception网络
      '''
      base_model = Xception(weights='imagenet', include_top=False)
      x = base_model.output
      x = MaxPooling2D(name = "MaxPooling2D")(x)
      x = GlobalMaxPooling2D(name="GlobalMaxPooling2D")(x)
      # Fully Connected Layer

      output = Dense(classes, activation='softmax')(x)

      model = Model(base_model.input, output)

      return model



   def fit(self,ddttrain):
      '''
       提取通过网络提取训练集的PCA和归一化的特征
      '''

      descriptors0 = np.zeros((1, 2048))
      descriptors1 = np.zeros((1, 2048))

      model = self.CreateModel()
      # model.load_weights(self.weight)
    #提取两层特征
      model0 = Model(inputs=model.input, outputs=model.get_layer(name='block14_sepconv2').output)
      model1 = Model(inputs=model.input, outputs=model.get_layer(name="MaxPooling2D").output)

      #载入数据集
      datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
      generator = datagen.flow_from_directory(
         ddttrain,
         batch_size=batch,
         class_mode=None,
         shuffle=False)
      #target_size = (size, size),
      for i in range(len(generator)):
         image0 = model0.predict(generator[i])
         image1 = model1.predict(generator[i])

         output0 = image0.reshape((image0.shape[0]*image0.shape[1]*image0.shape[2],image0.shape[3]))
         output1 = image1.reshape((image1.shape[0] * image1.shape[1] * image1.shape[2], image1.shape[3]))

         descriptors0 = np.concatenate((descriptors0, output0), axis=0)
         descriptors1 = np.concatenate((descriptors1, output1), axis=0)

      descriptors0 = np.delete(descriptors0, 0, axis=0)
      descriptors1 = np.delete(descriptors1, 0, axis=0)
   #对特征做归一化
      descriptors0_mean = sum(descriptors0) / len(descriptors0)
      descriptors1_mean = sum(descriptors1) / len(descriptors1)

    #对特征做PCA
      pca0 = PCA(n_components=1)
      pca0.fit(descriptors0)
      trans_vec0 = pca0.components_[0]

      pca1 = PCA(n_components=1)
      pca1.fit(descriptors1)
      trans_vec1 = pca1.components_[0]


      return (trans_vec0, trans_vec1), (descriptors0_mean, descriptors1_mean)


   def co_locate(self, DDTimage,trans_vectors,descriptor_mean_tensors,savedir):

      '''
       载入载入测试数据集和训练集做DDT
      '''
      model = self.CreateModel()
      # model.load_weights(self.weight)
      model0 = Model(inputs=model.input, outputs=model.get_layer(name='block14_sepconv2').output)
      model1 = Model(inputs=model.input, outputs=model.get_layer(name="MaxPooling2D").output)
      img = image.load_img(DDTimage)
      #, target_size = (self.input_shape[0], self.input_shape[1])
      img = image.img_to_array(img)
      img = np.array(img, dtype=np.uint8)
      # plt.imshow(img)
      # plt.show()
      origin_image = img.copy()
      origin_height, origin_width = img.shape[0], img.shape[1]
      img = np.expand_dims(img, axis=0)
      img = preprocess_input(img)


      image0 = model0.predict(img)
      image1 = model1.predict(img)

      h0, w0 = image0.shape[1], image0.shape[2]
      h1, w1 = image1.shape[1], image1.shape[2]

      image0 = image0.reshape((image0.shape[1]*image0.shape[2],image0.shape[3]))
      image0 -= np.repeat(descriptor_mean_tensors[0],image0.shape[0]).reshape((image0.shape[0],image0.shape[1]))

      image1 = image1.reshape((image1.shape[1]*image1.shape[2],image1.shape[3]))
      image1 -= np.repeat(descriptor_mean_tensors[1], image1.shape[0]).reshape((image1.shape[0], image1.shape[1]))

      P0 = np.dot(trans_vectors[0], image0.transpose()).reshape(h0, w0)
      P1 = np.dot(trans_vectors[1], image1.transpose()).reshape(h1, w1)

      mask0 = self.max_conn_mask(P0, origin_height, origin_width)
      mask1 = self.max_conn_mask(P1, origin_height, origin_width)
      mask = mask0 + mask1
      mask[mask == 1] = 0
      mask[mask == 2] = 1
      # mask = mask1
      mask_3 = np.concatenate(
          (np.zeros((2, origin_height, origin_width), dtype=np.uint16), mask * 255), axis=0)
      # 将原图同mask相加并展示
      mask_3 = np.transpose(mask_3, (1, 2, 0))
      mask_3 = origin_image + mask_3
      mask_3[mask_3[:, :, 2] > 254, 2] = 255
      mask_3 = np.array(mask_3, dtype=np.uint8)
      print("save the  image. ")
      cv2.imwrite(savedir + "/adiadsresult5.jpg", mask_3)


   def max_conn_mask(self, P, origin_height, origin_width):
      h, w = P.shape[0], P.shape[1]
      highlight = np.zeros(P.shape)
      for i in range(h):
          for j in range(w):
              if P[i][j] > 0:
                  highlight[i][j] = 1

      # 寻找最大的全联通分量
      labels = measure.label(highlight, neighbors=4, background=0)
      props = measure.regionprops(labels)
      max_index = 0
      for i in range(len(props)):
          if props[i].area > props[max_index].area:
              max_index = i
      max_prop = props[max_index]
      highlights_conn = np.zeros(highlight.shape)
      for each in max_prop.coords:
          highlights_conn[each[0]][each[1]] = 1

      # 最近邻插值：
      highlight_big = cv2.resize(highlights_conn,
                                 (origin_width, origin_height),
                                 interpolation=cv2.INTER_NEAREST)

      highlight_big = np.array(highlight_big, dtype=np.uint16).reshape(1, origin_height, origin_width)
      # highlight_3 = np.concatenate((np.zeros((2, origin_height, origin_width), dtype=np.uint16), highlight_big * 255), axis=0)
      return highlight_big



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    DDTplus = DDTplus()
    trans_vectors, descriptor_means = DDTplus.fit(DDTtrain)
    DDTplus.co_locate(DDTimage,trans_vectors,descriptor_means,DDT_savedir)
