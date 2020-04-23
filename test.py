import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()

# pandas 이용해 csv 데이터 가져오고 Timestamp 부분을 드랍 시킨다
df = pd.read_csv("market-price.csv")
df_norm = df.drop(['Timestamp'], 1, inplace=True)

print(df)
# 예측을 30 일로 잡는다
# 항상 데이터 트래이닝 80, 검증 20, 테스트 20으로 잡아야 한다
prediction_days = 30

df_train = df[:len(df) - prediction_days]
print(df_train)
df_test = df[len(df) - prediction_days:]
# fit_transform 사용시 평균 0 오류 1
training_set = df_train.values
training_set = min_max_scaler.fit_transform(training_set)

# data 를 x와 y로 나누어준다
x_train = training_set[0:len(training_set) - 1]
y_train = training_set[1:len(training_set)]
x_train = np.reshape(x_train, (len(x_train), 1, 1))

# train model
# LSTM cell 에 사용된 단위
num_units = 4
# 함활성화 기능, 현재 sigmoid 가 사용
activation_function = 'sigmoid'
# 손실 최소화 위한 옵티마이저 현재 아담
optimizer = 'adam'
# 네트워크 무게와 손실 최소화 위해서 현재 Mean squared error
loss_function = 'mean_squared_error'
# 배치 사이즈, 반복 시 마다 훈련 세트에서 5선택
batch_size = 5
# 반복 횟수
num_epochs = 100

# 입출력 숨김의 순차적 모델 정의
# Initialize the RNN
regressor = Sequential()

# LSTM 셀을 모델에 추가
# Adding the input layer and the LSTM layer
regressor.add(LSTM(units=num_units, activation=activation_function, input_shape=(None, 1)))
# regressor.add(LSTM(units=4, activation='sigmoid, input_shape=(None, 1)))

# 출력레이어 추가   기본 선형 활성화 치수 1
# Adding the output layer
regressor.add(Dense(units=1))

# 손실 방지를 위한 옵티마이저와 loss 설정
# Compiling the RNN
regressor.compile(optimizer=optimizer, loss=loss_function)

# x_train과 y_train 이용하여 모델 학습
# Using the training set to train the model
regressor.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs)

# Predict price
# min_max_transform 을 사용하여 데이터 스케일한 후 예측 위해 데이터 재구성
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = min_max_scaler.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))

# 예측
predicted_price = regressor.predict(inputs)
# 실제 값과 일치하도록 예측된 데이터 크기 조겅
predicted_price = min_max_scaler.inverse_transform(predicted_price)

# 시각화
plt.figure(figsize=(25, 25), dpi=80, facecolor='w', edgecolor='k')

plt.plot(test_set[:, 0], color='red', label='Test 실제')
plt.plot(predicted_price[:, 0], color='blue', label='Test 예측')

plt.title('BTC 가격 예측', fontsize=40)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC price', fontsize=40)
plt.show()

# 학습된 모델 저장
regressor.save('model.h5')