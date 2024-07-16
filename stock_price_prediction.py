import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Flatten, MaxPooling1D, RepeatVector, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import yfinance as yf

class Company:
    def __init__(self,name,df,related_company):
        self.name = name
        self.df = df
        self.related_company = related_company
        self.related_feature = []
        self.target = 'RSI'
        self.train_data = ()
        self.date_price_target = pd.DataFrame()
        self.y_scaler = None

    def sort_date(self):
        self.df = self.df.rename(columns={"time":"Date"})
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(by='Date')
        self.df = self.df.reset_index(drop=True)

    def add_price(self):
        start = str(self.df['Date'].iloc[0])[:-9]
        end = str(self.df['Date'].iloc[-1])[:-9]
        data = yf.download(self.name, start=start, end=end)
        self.df = self.df.join(data, on='Date', how='inner')

    def add_related_company(self):
        for i in self.related_company:
            start = str(self.df['Date'].iloc[0])[:-9]
            end = str(self.df['Date'].iloc[-1])[:-9]
            data = yf.download(i, start=start, end=end)
            rename = i + 'Adj_Close'
            data = data.rename(columns={'Adj Close' : rename})
            data = data[[rename]]
            self.df = self.df.join(data, on='Date', how='left')

    def corrMat(self,target='RSI',figsize=(25,0.5),ret_id=True):
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        corr_mat = self.df.corr().round(2);shape = corr_mat.shape[0]
        corr_mat = corr_mat.transpose()
        corr = corr_mat.loc[:, self.df.columns == target].transpose().copy()
    
        if(ret_id is False):
            f, ax = plt.subplots(figsize=figsize)
            sns.heatmap(corr,vmin=-0.3,vmax=0.3,center=0, 
                         cmap=cmap,square=False,lw=2,annot=True,cbar=False)
            plt.title(f'Feature Correlation to {target} in {self.name}')

        if(ret_id):
            return corr
        
    def rmv_date_company(self):
        self.df.fillna(0, inplace=True)
        self.df = self.df.drop(columns=['Date','Company'])
        
    def get_related_feature(self,ftr_thr,company_thr,filter_ftr=True):
        corr = self.corrMat(target='RSI',figsize=(25,0.5),ret_id=True)
        # take the features with >= ftr_thr correlation, for the related company, with >= company_thr
        corr_ = corr.columns.to_list()
        for cor in corr_:
            if corr.loc[self.target,cor] >= ftr_thr:
                self.related_feature.append(cor)
            elif corr.loc[self.target,cor] >= company_thr:
                if cor in self.related_company:
                    self.related_feature.append(cor)
        if filter_ftr:
            to_drop = set(self.df.columns) - set(self.related_feature)
            self.df.drop(columns=to_drop,inplace=True)
        else:
            return self.related_feature

def load_data(file_path,to_train):
    # Get data for each companies
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Unnamed: 0'])
    df.fillna(0, inplace=True)
    grouped = df.groupby('Company')

    # get name,df
    dict_comp = {}
    for company, data in grouped:
        if company not in dict_comp:
            dict_comp[company]={}
        dict_comp[company]['name'] = company
        dict_comp[company]['df'] = data

    # get related company
    dict_comp['LTH']['related'] = ['^SP500-25','PLNT','XPOF']
    dict_comp['ETD']['related'] = ['LOVE','FLXS','HOFT']
    dict_comp['METC']['related'] = ['HAYN','RYI','ASTL']

    # Define Company objects
    for key in to_train: 
        name = dict_comp[key]['name']
        df = dict_comp[key]['df']
        related = dict_comp[key]['related']
        dict_comp[key]['obj'] = Company(name,df,related)

    return dict_comp

        
def create_target_rsi(df):
    delta = df['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    window_length = 5
    avg_gain = gain.rolling(window=window_length, min_periods=1).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

def windowing(x,y,x_length,y_length):
    if y is None:
        x_windows = [x[i:i + x_length] for i in range(len(x) - x_length + 1)]
        x = np.array(x_windows)
        return x
    else:
        x_windows = [x[i:i + x_length] for i in range(len(x) - x_length + 1)]
        y_windows = [y[i + x_length : i + x_length + y_length] for i in range(len(y) - x_length - y_length + 1)]
        x = np.array(x_windows)
        y = np.array(y_windows)
        return x,y
    
def plot_price_target(instance,start,end, figsize,str1='Adj Close', str2='RSI'):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    ax1.plot(instance.date_price_target['Date'].iloc[start:end],instance.date_price_target[str1].iloc[start:end], marker='o', label='Close Price')
    ax1.set_title(str1)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(instance.date_price_target['Date'].iloc[start:end],instance.date_price_target[str2].iloc[start:end], marker='o', label='RSI', color='orange')
    ax2.set_title(str2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel(str2)
    ax2.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

def create_model(x,y):
    model = Sequential()
    model.add(tf.keras.Input((x.shape[1],x.shape[2])))
    model.add(Conv1D(filters=32, kernel_size=7, activation='relu'))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train(to_train, dict_comp, val_size, epochs):
    # Train for each company
    for key in to_train:
        instance = dict_comp[key]['obj']
        x, y =  instance.train_data

        # Split to train and validation data
        val_idx = len(x) - int(len(x)*val_size)
        X_val = x[val_idx:]
        y_val = y[val_idx:]
        X_train = x[:val_idx]
        y_train = y[:val_idx]

        # Create and train model
        model = create_model(X_train,y_train) 

        model_checkpoint_callback = ModelCheckpoint(
            filepath= f'model/{key}.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0)

        history = model.fit(X_train, y_train, validation_data=(X_val,y_val), verbose=0,epochs=epochs,
                  callbacks=[model_checkpoint_callback])
        
        # Get loss and validation loss to plot
        loss = history.history['loss']
        val_loss =  history.history['val_loss']
        min_val_loss = np.min(val_loss)
        print(f'minimum validation loss : {min_val_loss}')
        epoch = range(1,len(loss)+1)

        plt.figure(figsize=(12, 4))
        plt.plot(epoch, loss, 'b', label='Training loss')
        plt.plot(epoch, val_loss, '-r', label='Validation loss')
        plt.title(f'{key} Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        # Plot validation with the desired number of samples
        plot_prediction(instance,model,num_sample=30,figsize=(12,8))

def unwindowing(windowed_data, starting_index=0):
    # unwindowing for stride=1 windowed data
    first_window = windowed_data[starting_index]
    unwindow_list = []
    for i in range(starting_index+1,windowed_data.shape[0]):
        unwindow_list.append(windowed_data[i][-1])
    unwindowed = np.concatenate((first_window,unwindow_list))
    return unwindowed

def plot_prediction(instance,model,num_sample,figsize=(12,4)):
    # Extract training data and actual prices
    x,y = instance.train_data
    price = instance.date_price_target['Adj Close'][-num_sample:]
    date = instance.date_price_target['Date'][-num_sample:]
    
    # Get the last num_samples of data to predict
    x_val = x[-num_sample:]
    y_val = y[-num_sample:]

    # Predict with the model
    y_hat = model.predict(x_val)

    # Return the prediction values to the original scale of the RSI
    y_val_orig = instance.y_scaler.inverse_transform(y_val)
    y_hat_orig = instance.y_scaler.inverse_transform(y_hat)

    # Unwindowing
    unw_y_val = unwindowing(y_val_orig)
    unw_y_hat = unwindowing(y_hat_orig)
    
    # Pad zeros to y. Aligning the price and y 
    trunc = price.shape[0]-unw_y_val.shape[0]
    zeros_array = np.zeros(trunc)
    y_val = np.concatenate((zeros_array, unw_y_val))
    y_hat = np.concatenate((zeros_array, unw_y_hat))

    # Plot the result
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    ax1.plot(date,y_val,marker='o', linestyle='-', color='b', label='Real RSI')
    ax1.plot(date,y_hat,marker='o', linestyle='-', color='r', label='Predicted RSI')

    ax1.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
    ax1.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
    
    name = instance.name
    ax1.set_title(f'{name} Real and Predicted RSI')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(date,price,marker='o', linestyle='-', color='g', label='Price')
    ax2.legend()
    ax2.grid(True)
    plt.show()

def main():
    file_path = 'data/lth_etd_metc_Technical_indicator.csv'
    company_to_train = ['LTH','ETD','METC']
    company_dict = load_data(file_path,company_to_train)

    # Data preprocessing
    for key in company_to_train:
        instance = company_dict[key]['obj']
        instance.sort_date()
        instance.add_price()
        instance.add_related_company()
        instance.date_price_target = create_target_rsi(instance.df)
        file_name = f'{key}.csv'
        instance.df.to_csv(file_name)
        instance.rmv_date_company()
        instance.get_related_feature(ftr_thr=0.5,company_thr=0.1)
        
        # put RSI as the last column
        data = instance.df.pop('RSI')
        instance.df['RSI'] = data
        
        # normalize the data
        scaler =  StandardScaler()
        key_np = scaler.fit_transform(instance.df)
        
        # get y scaler
        rsi = data.to_numpy()
        rsi = rsi.reshape(-1,1)
        scaler_y = StandardScaler()
        instance.y_scaler = scaler_y.fit(rsi)
        
        # windowing
        label = key_np[:,-1]
        ftr = key_np[:,:-1]
        x, y = windowing(ftr,label,30,1)
        num_window = y.shape[0]
        x_trunc = x[:num_window]
        print(x_trunc.shape)
        print(y.shape)
        instance.train_data = (x_trunc,y)

        plot_price_target(instance,50,100,figsize=(10,8), str1='Adj Close', str2='RSI')
    print('Start training!')
    train(company_to_train, company_dict, val_size=0.25,epochs=25)

if __name__=="__main__": 
    main() 