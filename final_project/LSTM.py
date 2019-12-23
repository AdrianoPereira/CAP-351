import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from scipy.signal import savgol_filter

gdf = gpd.read_file('project/shapefiles/estadosl_2007.shp')
path_glm = "/home/adriano/earth-observation/.data/csv/glm"

files = [os.path.join(path_glm, file) for file in os.listdir(path_glm)]
dfs = [pd.read_csv(file) for file in files]
dfs = pd.concat(dfs, sort=False)

group = dfs.groupby(['start_scan', 'state']).count()

total_lightning = group['flash_lon'].values.tolist()
p25 = int(np.percentile(total_lightning, 25))
p50 = int(np.percentile(total_lightning, 50))
p75 = int(np.percentile(total_lightning, 75))
p100 = int(np.percentile(total_lightning, 100))

x, y = .575, np.max(total_lightning)-10

plt.figure(figsize=(12, 8))
plt.boxplot(total_lightning)
plt.text(x, y, "25º Percentil %s"%p25, fontdict=dict(size=16))
y-=12
plt.text(x, y, "50º Percentil %s"%p50, fontdict=dict(size=16))
y-=12
plt.text(x, y, "75º Percentil %s"%p75, fontdict=dict(size=16))
y-=12
plt.text(x, y, "100º Percentil %s"%p100, fontdict=dict(size=16))
plt.xticks([])


times = sorted(np.unique(dfs['timestamp']))
ts_size = len(times)
time_serie = {}
for _, row in gdf.iterrows():
    time_serie[row['NOMEUF2']] = np.zeros(ts_size)
    
group = dfs.groupby(['timestamp', 'state']).agg({'state': 'count'})
group.head()
idx = 0
old_index = group.index.get_level_values(0)[0]
for index, value in group.iterrows():
    if index[1] == 'OUTSIDE':
        continue
    if index[0] != old_index:
        old_index = index[0]
        idx += 1
    time_serie[index[1]][idx] = value[0]

time_serie = pd.DataFrame(time_serie)
time_serie.set_index(pd.Series(times), inplace=True)
time_serie.head()

metrics = {}
for state in gdf['NOMEUF2'].values:

    base = time_serie[state].values
    norm = MinMaxScaler(feature_range=(0, 1))
    basen = norm.fit_transform(base.reshape(-1, 1))
    
    v_pred = []
    v_real = []
    size = 4
    for i in range(size, len(base)):
        v_pred.append(basen[i-size:i, 0])
        v_real.append(basen[i, 0])
    
    v_real = np.array(v_real)
    v_pred = np.array(v_pred)
    v_pred = np.reshape(v_pred, (v_pred.shape[0], v_pred.shape[1], 1))
    
    X_train, X_test, y_train, y_test = train_test_split(v_pred, v_real, test_size=.15)
    
    
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(v_pred.shape[1], 1)))
    model.add(Dropout(.2))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(.2))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(.2))
    
    model.add(LSTM(units=50))
    model.add(Dropout(.2))
    
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))
    
    
    val_loss, loss = history.history['val_loss'], history.history['loss']
    val_acc, acc = history.history['val_mean_absolute_error'], history.history['mean_absolute_error']
    
    fig, ax = plt.subplots(1, 2, figsize=(22, 5))
    ax[0].plot(loss, label='train', color='blue')
    ax[0].plot(val_loss, label='val', color='green')
    ax[0].legend()
    ax[0].set_title('categorical_crossentropy')
    
    ax[1].plot(acc, label='train', color='blue')
    ax[1].plot(val_acc, label='val', color='green')
    ax[1].legend()
    ax[1].set_title('accuracy')
    
    plt.savefig('lstm_images_metrics/%s.pdf'%state, dpi=600, \
           bbox_inches='tight', transparent="False", pad_inches=0.1)
    metrics[state] = dict(val_loss=val_loss, loss=loss, val_acc=val_acc, 
           acc=acc)
    
    
    y_pred = model.predict(X_test)
    
    dy_pred = norm.inverse_transform(y_pred.reshape(-1, 1))
    dy_test = norm.inverse_transform(y_test.reshape(-1, 1))
    
    state = state.replace(' ', '_')
    fig, ax = plt.subplots(1, 2, figsize=(17, 5))
    ax[0].plot(dy_test, dy_pred, 's', color='blue', markeredgecolor='black', markeredgewidth='1', alpha=.6)
    ax[0].set_title('%s\nLightning observed X Lightning Predicted'%state, fontdict=dict(size=14))
    ax[0].set_xlabel('Total observed')
    ax[0].set_ylabel('Total predicted')
    ax[0].plot([np.min(dy_test), np.max(dy_test)], [np.min(dy_pred), np.max(dy_pred)], color='r', linestyle='--', linewidth=2, alpha=.75)
    
    ax[1].plot(dy_pred, label='Predicted', color='red', linestyle='--', marker='o')
    ax[1].plot(dy_test, label='Observed', color='blue', marker='.')
    ax[1].set_title('Lightning observed X Lightning Predicted', fontdict=dict(size=14))
    # ax[1].set_xlabel('Time')
    ax[1].set_xticks([])
    ax[1].set_ylabel('Total lightning')
    ax[1].legend()
    plt.savefig('lstm_images/%s.pdf'%state, dpi=600, \
           bbox_inches='tight', transparent="False", pad_inches=0.1)
