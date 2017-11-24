import pandas as pd
import matplotlib.pyplot as plt

def load_raman_map():


    meas = pd.read_csv('data/raman_100x100_wavenumbers1000_hotspots10_0dB_withinteractions.csv', header=None)
    print(meas.shape)
    substance1_map = meas.iloc[0,:]
    substance1_r = substance1_map.values.reshape(1000,10000)

    substance2_map = meas.iloc[1,:]
    substance2_r = substance2_map.values.reshape(1000,10000)

    mix_map = meas.iloc[2,:]
    mix_r = mix_map.values.reshape(1000,10000)



    return substance1_r,substance2_r,mix_r


if __name__=="__main__":
    substance1_r,substance2_r,mix_r = load_raman_map()

    plt.figure(figsize=(12,12))
    plt.subplot(3,1,1)
    plt.plot(substance1_r)

    plt.subplot(3,1,2)
    plt.plot(substance2_r)
    plt.subplot(3,1,3)
    plt.plot(mix_r)

    plt.show(block=True)