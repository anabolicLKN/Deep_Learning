#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
#데이터 읽기
red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

#column 추가
red_wine['color'] = 1.
white_wine['color'] = 0.

#데이터 합치기
wine_data = pd.concat([red_wine, white_wine])
wine_data.reset_index(drop=True, inplace=True)

#결손치가 있는 데이터행 삭제
wine_data.dropna(inplace=True)

#특성과 레이블 분리
x = wine_data.drop(['color'], axis=1)
y = wine_data['color']

#데이터분할 : train, valid, test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

#keras 모델생성
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'])

#Early Stopping 콜백추가
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    verbose=1,
    restore_best_weights=True
)

#모델훈련
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=512,
    validation_split=0.1,
    verbose=2,
    callbacks=[early_stopping])

#모델평가
predictions = model.predict(X_test)
test_loss,test_accuracy = model.evaluate(X_test, y_test, verbose=2)

#평가 출력
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')


# In[36]:


#훈련 손실과 검증 손실그래프
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

#훈련 정확도와 검증 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[37]:


#선택한 하이퍼파라미터1 : epochs=100 
#100회의 충분한 학습을 통해 모델이 데이터를 잘 파악할 수 있도록 함
#만약 패턴이 복잡하더라도 깊이있는 학습가능

#선택한 하이퍼파라미터2: patience=20
#과적합방지 전략으로 드롭아웃을 선택했기 때문에 검증곡선이 들쑥날쑥 할 동안 어느정도의 여유 제공


# In[ ]:


#선택한 과적합 방지전략1 : 드롭아웃(Dropout)
#한 줄로 간단히 추가할 수 있으며, 무작위 뉴런 비활성화를 통해 뉴런간 의존도 줄임.
#앙상블 효과 낼 수 있음

#선택한 과적합 방지전략2: 조기종료(Early Stopping)
#에폭수가 100이므로 과적합 될 가능성있지만 조기종료를 통해 검증곡선의 개선이 없다면 조기중단함.
#또한 최적의 epoch를 자동선택해주므로 편리함
#드롭아웃과 조기종료는 같이 사용하였을때, 드롭아웃은 모델의 과적합 억제, 조기종료는 학습시점을 최적화해줌
#서로 상호보완적인 효과가 있다고 판단

