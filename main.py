import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


# file_names = list()


def read_data(directory):
    accel = list()
    gyro = list()
    barometer = list()
    pedometer = list()
    data = list()
    file_names = os.listdir(directory)
    file_names.pop(0)

    for file_name in file_names:
        directory = 'data/Real_Data/' + file_name + '/' + file_name
        accel.append(pd.read_csv(directory + '-' + 'android.sensor.accelerometer.txt', header=None))
        gyro.append(pd.read_csv(directory + '-' + 'android.sensor.gyroscope.txt', header=None))
        barometer.append(pd.read_csv(directory + '-' + 'android.sensor.pressure.txt', header=None))
        pedometer.append(pd.read_csv(directory + '-' + 'android.sensor.step_detector.txt', header=None))
        data.append(pd.read_csv(directory + '-' + 'android.sensor.data.txt', header=None))

    return file_names, accel, gyro, barometer, pedometer, data

def raw(directory, show=True):
    file_names, accel, gyro, barometer, pedometer, data = read_data(directory)
    accelTotalList = list()
    gyroTotalList = list()
    barList = list()
    stepsList = list()
    timestampsList = list()
    for i in range(len(accel)):
        accelTotal = pd.DataFrame()
        accelTotal.insert(loc=0, column=0, value=accel[i].iloc[:,0])
        accelTotal.insert(loc=1, column=1,
                          value=np.sqrt(accel[i].iloc[:, 1] **2 + accel[i].iloc[:, 2] **2 + accel[i].iloc[:, 3] **2))
        accelTotal = (accelTotal.iloc[:, 1].groupby(accelTotal.iloc[:, 0]).sum())
        accelTotal = (accelTotal.reindex(range(accelTotal.index.max() + 1)))
        accelTotal = (accelTotal.fillna(0))
        accelTotal = accelTotal.reset_index()
        accelTotal = accelTotal.values
        accelTotalList.append(accelTotal)

        gyroTotal = pd.DataFrame()
        gyroTotal.insert(loc=0, column=0, value=gyro[i].iloc[:, 0])
        gyroTotal.insert(loc=1, column=1,
                          value=np.sqrt(gyro[i].iloc[:, 1] ** 2 + gyro[i].iloc[:, 2] ** 2 + gyro[i].iloc[:, 3] ** 2))
        gyroTotal = (gyroTotal.iloc[:, 1].groupby(gyroTotal.iloc[:, 0]).sum())
        gyroTotal = (gyroTotal.reindex(range(gyroTotal.index.max() + 1)))
        gyroTotal = (gyroTotal.fillna(0))
        gyroTotal = gyroTotal.reset_index()
        gyroTotal = gyroTotal.values
        gyroTotalList.append(gyroTotal)

        bar = barometer[i].iloc[:, 1].groupby(barometer[i].iloc[:, 0]).sum()
        bar = bar.reindex(range(bar.index.max() + 1))
        bar = (bar.fillna(0))
        bar = bar.reset_index()
        bar = bar.values
        barList.append(bar)

        steps = pd.DataFrame(pedometer[i].iloc[:, 1].groupby(pedometer[i].iloc[:, 0]).sum())
        steps = steps.reindex(range(steps.index.max() + 1))
        steps = (steps.fillna(0))
        steps = steps.values
        stepsList.append(steps)

        timestampsList.append(np.arange(0, len(accelTotal), len(accelTotal)/1000))





    if show==True:
        for i in range(len(accelTotalList)):
            accelTotalList[i] = np.interp(timestampsList[i], accelTotalList[i][:, 0], accelTotalList[i][:, 1])
            gyroTotalList[i] = np.interp(timestampsList[i], gyroTotalList[i][:, 0], gyroTotalList[i][:, 1])
            barList[i] = np.interp(timestampsList[i], barList[i][:, 0], barList[i][:, 1]).reshape(-1, 1)

        plt.subplot(2, 1, 1)
        for i in range(len(accelTotalList)):
            plt.plot(accelTotalList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.ylabel('Acceleration (m/s^2)', fontsize=8)
        plt.yticks(fontsize=8)
        plt.gca().set_title('Accelerometer Magnitude', fontsize=8)
        plt.legend()

        plt.subplot(2, 1, 2)
        for i in range(len(barList)):
            plt.plot(barList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.ylabel('Pressure (mbar)', fontsize=8)
        plt.yticks(fontsize=8)
        plt.gca().set_title('Barometric Pressure', fontsize=8)
        plt.legend()

        plt.show()

        plt.subplot(2, 1, 1)
        for i in range(len(stepsList)):
            plt.plot(stepsList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.ylabel('Steps', fontsize=8)
        plt.yticks(fontsize=8)
        plt.gca().set_title('Pedometer', fontsize=8)
        plt.legend()

        plt.subplot(2, 1, 2)
        for i in range(len(gyroTotalList)):
            plt.plot(gyroTotalList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.ylabel('Gyro Rotation', fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend()

        plt.show()


        plt.subplot(1, 1, 1)

        plt.ylabel('Seconds(s)', fontsize=8)
        plt.xlabel('File', fontsize=8)
        plt.gca().set_title('Time', fontsize=8)
        plt.xticks(range(len(data)), file_names)
        x = range(len(data))
        y = list()
        for i in range(len(data)):
            dt_obj = datetime.strptime(data[i].iloc[1].values[0],
                                       '%M:%S:%f').time()
            seconds = dt_obj.hour*60 +dt_obj.minute*60 + dt_obj.second+(dt_obj.microsecond/1000000)
            y.append(seconds)

        plt.scatter(x, y)
        plt.show()

def features(directory, show=True):
    file_names, accel, gyro, barometer, pedometer, data = read_data(directory)
    accelMeanList = list()
    accelVarList = list()
    gyroMeanList = list()
    gyroVarList = list()
    barMeanList = list()
    barVarList = list()
    speedList = list()
    timestampsList = list()
    for i in range(len(file_names)):
        accelMean = pd.DataFrame()
        accelMean.insert(loc=0, column=0, value=accel[i].iloc[:, 0])
        accelMean.insert(loc=1, column=1,
                          value=np.sqrt(accel[i].iloc[:, 1] ** 2 + accel[i].iloc[:, 2] ** 2 + accel[i].iloc[:, 3] ** 2))
        accelMean = (accelMean.iloc[:, 1].groupby(accelMean.iloc[:, 0]).mean())
        accelMean = (accelMean.reindex(range(accelMean.index.max() + 1)))
        accelMean = (accelMean.fillna(0)).values
        accelMeanList.append(accelMean)

        accelVar = pd.DataFrame()
        accelVar.insert(loc=0, column=0, value=accel[i].iloc[:, 0])
        accelVar.insert(loc=1, column=1,
                         value=np.sqrt(accel[i].iloc[:, 1] ** 2 + accel[i].iloc[:, 2] ** 2 + accel[i].iloc[:, 3] ** 2))
        accelVar = (accelVar.iloc[:, 1].groupby(accelVar.iloc[:, 0]).var())
        accelVar = (accelVar.reindex(range(accelVar.index.max() + 1)))
        accelVar = (accelVar.fillna(0)).values
        accelVarList.append(accelVar)

        gyroMean = pd.DataFrame()
        gyroMean.insert(loc=0, column=0, value=gyro[i].iloc[:, 0])
        gyroMean.insert(loc=1, column=1,
                         value=np.sqrt(gyro[i].iloc[:, 1] ** 2 + gyro[i].iloc[:, 2] ** 2 + gyro[i].iloc[:, 3] ** 2))
        gyroMean = (gyroMean.iloc[:, 1].groupby(gyroMean.iloc[:, 0]).mean())
        gyroMean = (gyroMean.reindex(range(gyroMean.index.max() + 1)))
        gyroMean = (gyroMean.fillna(0)).values
        gyroMeanList.append(gyroMean)

        gyroVar = pd.DataFrame()
        gyroVar.insert(loc=0, column=0, value=gyro[i].iloc[:, 0])
        gyroVar.insert(loc=1, column=1,
                        value=np.sqrt(gyro[i].iloc[:, 1] ** 2 + gyro[i].iloc[:, 2] ** 2 + gyro[i].iloc[:, 3] ** 2))
        gyroVar = (gyroVar.iloc[:, 1].groupby(gyroVar.iloc[:, 0]).var())
        gyroVar = (gyroVar.reindex(range(gyroVar.index.max() + 1)))
        gyroVar = (gyroVar.fillna(0)).values
        gyroVarList.append(gyroVar)

        barMean = barometer[i].iloc[:, 1].groupby(barometer[i].iloc[:, 0]).mean()
        barMean = barMean.reindex(range(barMean.index.max() + 1))
        barMean = (barMean.fillna(0)).values
        barMeanList.append(barMean)

        barVar = barometer[i].iloc[:, 1].groupby(barometer[i].iloc[:, 0]).mean()
        barVar = barVar.reindex(range(barVar.index.max() + 1))
        barVar = (barVar.fillna(0)).values
        barVarList.append(barVar)

        steps = pd.DataFrame(pedometer[i].iloc[:, 1].groupby(pedometer[i].iloc[:, 0]).sum())
        steps = steps.reindex(range(steps.index.max() + 1))
        steps = (steps.fillna(0))
        stride = (float)(data[i].iloc[0,0])
        speed = steps.iloc[:,0].values*stride
        speedList.append(speed)
        # timestamps = range(len(accelMean) + 1)
        timestampsList.append(range(len(accelMean) + 1))

    if show==True:
        plt.subplot(2, 1, 1)
        for i in range(len(accelMeanList)):
            plt.plot(accelMeanList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('m/s^2', fontsize=8)
        plt.gca().set_title('Mean Magnitude of Acceleration', fontsize=8)
        plt.legend()

        plt.subplot(2, 1, 2)
        for i in range(len(accelVarList)):
            plt.plot(accelVarList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('m/s^2', fontsize=8)
        plt.gca().set_title('Variance of Acceleration', fontsize=8)
        plt.legend()

        plt.show()

        plt.subplot(2, 1, 1)
        for i in range(len(gyroMeanList)):
            plt.plot(gyroMeanList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('degrees/s', fontsize=8)
        plt.gca().set_title('Mean Magnitude of Gyroscopic Rotation', fontsize=8)
        plt.legend()

        plt.subplot(2, 1, 2)
        for i in range(len(gyroVarList)):
            plt.plot(gyroVarList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('degrees/s', fontsize=8)
        plt.gca().set_title('Variance of Gyroscopic Rotation', fontsize=8)
        plt.legend()

        plt.show()

        plt.subplot(2, 1, 1)
        for i in range(len(barMeanList)):
            plt.plot(barMeanList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('mbar', fontsize=8)
        plt.gca().set_title('Mean Pressure Per Second', fontsize=8)
        plt.legend()

        plt.subplot(2, 1, 2)
        for i in range(len(barVarList)):
            plt.plot(barVarList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('mbar', fontsize=8)
        plt.gca().set_title('Variance of Pressure Per Second', fontsize=8)
        plt.legend()

        plt.show()

        plt.subplot(2,1,1)
        for i in range(len(speedList)):
            plt.plot(speedList[i], label=file_names[i])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('m/s', fontsize=8)
        plt.gca().set_title('Speed', fontsize=8)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.ylabel('Speed (m/s^2)', fontsize=8)
        plt.xlabel('File', fontsize=8)
        plt.gca().set_title('Average Speed', fontsize=8)
        plt.xticks(range(len(speedList)), file_names)
        x = range(len(speedList))
        y = list()
        for i in range(len(speedList)):
            speedList[i] = pd.DataFrame(speedList[i])
            y.append(speedList[i].mean())

        plt.scatter(x, y)

        plt.show()

        plt.subplot(2, 1, 1)

        plt.ylabel('Distance (m)', fontsize=8)
        plt.xlabel('File', fontsize=8)
        plt.gca().set_title('Total Distance', fontsize=8)
        plt.xticks(range(len(speedList)), file_names)
        x = range(len(speedList))
        y = list()
        for i in range(len(speedList)):
            speedList[i] = pd.DataFrame(speedList[i])
            y.append(pedometer[i].iloc[:,1].sum()*stride)
        plt.scatter(x, y)

        plt.subplot(2, 1, 2)
        plt.ylabel('Seconds(s)', fontsize=8)
        plt.xlabel('File', fontsize=8)
        plt.gca().set_title('Time', fontsize=8)
        plt.xticks(range(len(data)), file_names)
        x = range(len(data))
        y = list()
        for i in range(len(data)):
            dt_obj = datetime.strptime(data[i].iloc[1].values[0],
                                       '%M:%S:%f').time()
            seconds = dt_obj.hour * 60 + dt_obj.minute * 60 + dt_obj.second + (dt_obj.microsecond / 1000000)
            y.append(seconds)
        plt.scatter(x, y)

        plt.show()
# average speed
# gyroscopic rotation per second
# Accelemetor per sec
def train_model(directory):
    file_names, accel, gyro, barometer, pedometer, data = read_data(directory)
    accelMeanList = list()
    gyroMeanList = list()
    speedList = list()
    timestampsList = list()
    df = pd.DataFrame(columns=["speed", "accel", "gyro", "time"])
    # df = pd.DataFrame(
    #     {
    #         "speed":[0],
    #         "accel":[0],
    #         "gyro":[0]
    #     }
    # )
    for i in range(len(accel)):
        accelMean = pd.DataFrame()
        accelMean.insert(loc=0, column=0, value=accel[i].iloc[:, 0])
        accelMean.insert(loc=1, column=1,
                         value=np.sqrt(accel[i].iloc[:, 1] ** 2 + accel[i].iloc[:, 2] ** 2 + accel[i].iloc[:, 3] ** 2))
        accelMean = (accelMean.iloc[:, 1].groupby(accelMean.iloc[:, 0]).mean())
        accelMean = (accelMean.reindex(range(accelMean.index.max() + 1)))
        accelMean = (accelMean.fillna(0)).values


        gyroMean = pd.DataFrame()
        gyroMean.insert(loc=0, column=0, value=gyro[i].iloc[:, 0])
        gyroMean.insert(loc=1, column=1,
                        value=np.sqrt(gyro[i].iloc[:, 1] ** 2 + gyro[i].iloc[:, 2] ** 2 + gyro[i].iloc[:, 3] ** 2))
        gyroMean = (gyroMean.iloc[:, 1].groupby(gyroMean.iloc[:, 0]).mean())
        gyroMean = (gyroMean.reindex(range(gyroMean.index.max() + 1)))
        gyroMean = (gyroMean.fillna(0)).values


        steps = pd.DataFrame(pedometer[i].iloc[:, 1].groupby(pedometer[i].iloc[:, 0]).sum())
        steps = steps.reindex(range(steps.index.max() + 1))
        steps = (steps.fillna(0))
        stride = (float)(data[i].iloc[0,0])
        speed = steps.iloc[:,0].values*stride
        speedList.append(speed)
        # timestamps = range(len(accelMean) + 1)
        timestampsList.append(range(len(accelMean) + 1))


        dfi = pd.DataFrame(
            {
                "speed":speed,
                "accel":accelMean,
                "gyro":gyroMean
            }
        )
        dfi = dfi.iloc[1:]
        dfi["time"] = len(dfi.index)
        df = pd.concat([df, dfi])
    df = df.sample(frac=1).reset_index(drop=True)

    train_size = round((len(df)/100) * 80)
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)
    x_train = train_dataset.copy()
    x_test = test_dataset.copy()
    y_train = x_train.pop("time")
    y_test = x_test.pop("time")
    x_train = np.asarray(x_train).astype('float32')
    x_test = np.asarray(x_test).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')
    # print(train_dataset.describe().transpose()[['mean', 'std']])
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(x_train))
    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    linear_model.fit(
        x_train, y_train,
        epochs=100,
        verbose=0,
        validation_data=(x_test, y_test),
    )
    print(x_test[:1])
    print(linear_model.predict(x_test[:1]))
    # tf.saved_model.save(linear_model, "")
    # linear_model.save("mnist.h5")
    converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(linear_model)
    tflite_model = converter.convert()
    open('linear.tflite', 'wb').write(tflite_model)


if __name__ == '__main__':
    raw('data/Real_Data/', True)
    # features('data/Real_Data/', False)
    # train_model('data/Real_Data/')


