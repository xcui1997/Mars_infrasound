import pandas as pd 
import numpy as np
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt 
from scipy import signal
import glob
import time
from multiprocessing import Pool

# 全局变量
lowfrequency= 0.1
highfreq = 0.5
fs = 10
win_len = 15 # min

# 滤波
def bandpass_filter(data, lowcut=lowfrequency, highcut=highfreq, fs=fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

# 画psd
def plot_psd_csv(data, fs=fs, window_len=win_len*60, over_rate=0.5):
    '''
    plot the short time fourier transfrom of the data by using the scipy.signal.stft function.
    data is a pandas dataframe with UTC, Mars time and Pressure columns;
    sample_rate is the sample rate in Hz;
    window_len is the window length in seconds;
    over_rate is the overlapping rate.
    '''
    # plot the spectrogram by the signal.stft function
    f, t, Zxx = signal.stft(data["PRESSURE"].values, fs=fs, nperseg=int(window_len*fs), noverlap=int(window_len*fs*over_rate))
    plt.figure(figsize=(20, 10))
    plt.pcolormesh(t/3600, f, np.abs(Zxx), cmap='hot_r', vmin=0, vmax=0.005)
    plt.xlabel('Time [hour]\n Since {}'.format(data.index[0]), fontsize=20)
    plt.ylabel("Frequency (Hz)", fontsize=20)
    plt.title("Spectrogram of the pressure signal", fontsize=20)
    # plt.xticks(np.arange(0, len(t), int(len(t)/3)), np.array(data["LMST"].values)[np.arange(0, len(t), int(len(t)/3))])
    # plt.yticks(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
    plt.ylim(0.1, highfreq)
    plt.colorbar()
    plt.savefig("/home/lilab/xcui/data/Mars/out/png/stft_{}_{}_{}HZ.png".format(data.index[0], data.index[-1], highfreq), dpi=600)
    np.savetxt("/home/lilab/xcui/data/Mars/out/data/stft_{}_{}_{}Hz.txt".format(data.index[0], data.index[-1], highfreq), np.abs(Zxx), fmt="%.4f")
    plt.close()

    # generate the synthetic waveform with random phase and the same np.abs(Zxx)
    np.random.seed(0)
    phase = np.random.uniform(0, 2*np.pi, size=Zxx.shape)
    Zxx_syn = np.abs(Zxx)*np.exp(1j*phase)
    syn = signal.istft(Zxx_syn, fs=fs, nperseg=int(window_len*fs), noverlap=window_len*fs*over_rate)[1]
    np.savetxt("/home/lilab/xcui/data/Mars/out/data/syn_istft_{}_{}_{}Hz.txt".format(data.index[0], data.index[-1], highfreq), syn.real, fmt="%.4f")
    syn_norm = np.sign(syn.real)
    return syn_norm

# 计算自相关
def auto_corr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]/np.sum(x**2)

# 画自相关函数
def plot_correlation(time_, data_, win_len=win_len*60, overrate=0.5, samprate=fs, cor_t = 60, title="norm_data"):
    '''
        plot the autocorrelation of the data.
        time_ is the time array;
        data_ is the data array;
        win_len is the window length in seconds;
        overrate is the overlapping rate;
        samprate is the sample rate in Hz;
        cor_t is the plotted autocorrelation window length in seconds.
    '''

    auto_result = []
    Mars_time = []
    auto_all = []
    for i in range(1, int(len(data_)/(win_len*overrate*samprate))):
        if (i+1)*win_len*samprate*overrate >= len(data_):
            break

        x = data_[(i-1)*int(win_len*samprate*0.5):(i+1)*int(win_len*samprate*0.5)]
        auto_result.append(auto_corr(x)[:int(cor_t*samprate)])
        Mars_time.append(time_[i*int(win_len*samprate*0.5)])
        auto_all.append(auto_corr(x))

    plt.figure(figsize=(20, 10))
    plt.imshow(np.array(auto_result).T, aspect='auto', cmap='seismic_r', vmin=-0.5, vmax=0.5)
    plt.xlabel("Mars time", fontsize=20)
    plt.ylabel("Time (s)", fontsize=20)
    plt.title("Autocorrelation of the pressure signal", fontsize=20)
    plt.xticks(np.arange(0, len(Mars_time), int(len(Mars_time)/3)), np.array(Mars_time)[np.arange(0, len(Mars_time), int(len(Mars_time)/3))])
    plt.yticks(np.arange(0, cor_t*samprate, 10*samprate), np.arange(0, cor_t, 10))
    plt.ylim(0, cor_t*samprate)
    plt.colorbar()
    plt.savefig("/home/lilab/xcui/data/Mars/out/png/corr_{}_{}_{}_{}HZ.png".format(title, Mars_time[0], Mars_time[-1], highfreq), dpi=600)
    np.savetxt("/home/lilab/xcui/data/Mars/out/data/corr_{}_{}_{}_{}HZ.txt".format(title, Mars_time[0], Mars_time[-1], highfreq), np.array(auto_all), fmt="%.4f")
    np.savetxt("/home/lilab/xcui/data/Mars/out/data/corr_marstime_{}_{}_{}_{}HZ.txt".format(title, Mars_time[0], Mars_time[-1], highfreq), np.array(Mars_time), fmt="%s")



# save the data to a csv file
def filter_segmet(data):
    '''
    data is a pandas dataframe with UTC, Mars time and Pressure columns;
    lowcut is the lower bound of the bandpass filter;
    highcut is the upper bound of the bandpass filter;
    fs is the sample rate in Hz;
    order is the order of the butterworth filter.
    '''

    # bandpass filter
    data["PRESSURE"] = bandpass_filter(data["PRESSURE"])

    # one-bit normalization
    data["PRESSURE_norm"] = np.sign(data["PRESSURE"])

    # save the data to a csv file
    return data


# 对数据缺失值进行插值
def psd_with_interpolation(pressure_df, resamp=1/fs, min_gap=30, n_jobs=150):
    '''
    # presser_df is a pandas dataframe with UTC, Mars time and Pressure columns;
    # resmap is the resampling rate in Hz;
    # min_gap is the minimum gap in seconds to interpolate over. 
    '''
    # Convert UTC time to a datetime object and set it as the index
    pressure_df.set_index(pd.to_datetime(pressure_df["UTC"], format='%Y-%jT%H:%M:%S.%fZ').dt.round(freq='0.1S'), inplace=True)
    pressure_df = pressure_df.resample("{}S".format(resamp)).first()

    # interpolate over gaps less than min_gap
    helper = pd.DataFrame(index=pd.date_range(start=pressure_df.index.min(), end=pressure_df.index.max(), freq="{}S".format(resamp)))
    interpolate_df = pd.merge(helper, pressure_df, left_index=True, right_index=True, how="left")
    interpolate_df = interpolate_df.interpolate(method="linear", limit= int(min_gap/resamp))
    
    chunks = np.array_split(interpolate_df, n_jobs)
    pool = Pool(processes=n_jobs)
    results = pool.map(filter_segmet, [chunk for chunk in chunks])
    pool.close()
    pool.join()

    filter_norm_df = pd.concat(results)
    filter_norm_df.sort_index(inplace=True)
    filter_norm_df.to_csv("/home/lilab/xcui/data/Mars/out/data/data_{}_{}_{}HZ.csv".format(filter_norm_df.index[0], filter_norm_df.index[-1], highfreq), index=False)
    return filter_norm_df


if __name__ == "__main__":
    time0 = time.time()
    # 滤波过长返回值是nan

    files = glob.glob("/home/lilab/xcui/data/Mars/data/ps_bundle/data_calibrated/sol_0301_0389/ps_calib_*.csv")
    ps_calib = pd.concat([pd.read_csv(f) for f in files])
    ps_calib = ps_calib.sort_values("AOBT")

    interpolate_df = psd_with_interpolation(ps_calib[["UTC", "PRESSURE", "LMST", "AOBT"]])

    syn_data = plot_psd_csv(interpolate_df)

    plot_correlation(interpolate_df["UTC"].values, interpolate_df["PRESSURE_norm"].values)
    plot_correlation(interpolate_df["UTC"].values, syn_data, title="syn_data")
    print(time.time()-time0)