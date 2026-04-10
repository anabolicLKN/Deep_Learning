#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
# 필요한 라이브러리 가져오기
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

train_dir = './Petimages/train'
test_dir = './Petimages/test'

# base model 수정, 전이학습 추가
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(512,activation='relu')(x)
x=Dense(256,activation='relu')(x)
preds=Dense(2,activation='softmax')(x)

model=Model(inputs=base_model.input, outputs=preds)

for layer in model.layers[:150]:
    layer.trainable=False
for layer in model.layers[150:]:
    layer.trainable=True

# ImageDataGenerator 활용하여 전처리 옵션 설정
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
 rotation_range=10, brightness_range=[0.9,1.1], width_shift_range=0.1, zoom_range=[0.9, 1.1], height_shift_range=0.1,
 validation_split=0.2)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_dir,
 target_size=(224,224), batch_size=20, class_mode='categorical',
 subset='training')

valid_generator = train_datagen.flow_from_directory(train_dir,
 target_size=(224,224), batch_size=20, class_mode='categorical',
 subset='validation')

test_generator = test_datagen.flow_from_directory(test_dir,
 target_size=(224,224), batch_size=20, class_mode='categorical')
                                                  
###################################################################################################
# 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 학습
step_size_train=train_generator.n//train_generator.batch_size
history = model.fit(train_generator, steps_per_epoch=step_size_train, 
                    validation_data=valid_generator, 
                    validation_steps=valid_generator.n//valid_generator.batch_size, 
                    epochs=10,
                   verbose=1)                  
# 평가
step_size_test = test_generator.n//test_generator.batch_size
loss, accuracy = model.evaluate(test_generator, steps=step_size_test)
print('Loss = {:.5f}'.format(loss))
print('Accuracy = {:.5f}'.format(accuracy))
                                                  
# 훈련 손실과 검증 손실 그래프
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
                                                  
# 훈련 정확도와 검증 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


사용한 사전학습 모델 :imagenet으로 사전학습된 InceptionV3 모델을 사용하였음.
include_top=False로 불러와서 분류용 헤드는 제거하고 cnn기반 특징추출부만 가져옴.
분류기헤드는 GlobalAvergePooling2D 로 차원축소, FC 2층을 쌓고 softmax로 2클래스 다중분류함
과적합 방지를 위해 150까지 freeze하고 그 이후만 미세조정함.
회전,밝기, 높이, 너비 줌 등을 통해 데이터 증강함.

