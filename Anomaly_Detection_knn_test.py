import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

def main():
    # Current Directory
    current_directory = os.path.abspath(os.path.dirname(__file__))
    
    # csvファイルの読み込み
    df_csv = pd.read_csv(current_directory + "\\knn_test.csv")
    df_train_csv = pd.read_csv(current_directory + "\\knn_train_data\\06P_SMOOTHING_DATA\\06P_SMOOTHING_All_DATA.csv")
    df_test_csv = pd.read_csv(current_directory + "\\knn_test_data\\06P_SMOOTHING_DATA\\06P_SMOOTHING_All_DATA.csv")
    
    #　訓練データ
    train_data = df_csv.iloc[1:3000, 2]
    train_data2 = df_train_csv.iloc[1, 4:]
    print(train_data2)
    
    # 検証データ
    test_data = df_csv.iloc[3001:6000, 2]
    test_data2 = df_test_csv.iloc[1, 4:]
    
    # 訓練データの波形
    plt.figure()
    plt.plot(train_data, '-g')
    plt.title("train_plot")
    plt.show()
    
    train_fig = plt.figure()
    train_ax = train_fig.add_subplot(1, 1, 1)
    train_ax.set_ylim([0, 5000])
    train_ax.plot(train_data2.to_numpy(), '-g')
    plt.title("train_plot")
    plt.show()
    
    # 検証データの波形
    plt.figure()
    plt.plot(test_data, '-g')
    plt.title("test_plot")
    plt.show()
    
    test_fig = plt.figure()
    test_ax = test_fig.add_subplot(1, 1, 1)
    test_ax.set_ylim([0, 5000])
    test_ax.plot(test_data2.to_numpy(), '-g')
    plt.title("test_plot")
    plt.show()

    # 窓幅(window size)
    width = 100
    
    # K 近傍法のK
    nk = 1
    
    # 窓幅を使ってベクトルの集合を作成
    train = embed(train_data2, width)
    test = embed(test_data2, width)
    
    # k近傍法でクラスタリング
    neigh = NearestNeighbors(n_neighbors=nk)
    neigh.fit(train)
    
    # 距離を計算
    d = neigh.kneighbors(test)[0]

    # 距離をmax1にするデータ整形
    mx = np.max(d)
    d = d / mx

    # グラフ作成
    # DataFrameをnumpyのndarrayに変換する。
    #test_for_plot = df_csv.iloc[3001+width:6000, 2].to_numpy()
    #test_for_plot = test_data2.to_numpy()
    test_for_plot = df_test_csv.iloc[1, 4+width:].to_numpy()
    
    fig = plt.figure()
    # 1x1グラブの1番目のグラフ
    ax1 = fig.add_subplot(1,1,1)
    # ax2 and ax1 will have common x axis and different y axis
    ax2 = ax1.twinx()
    
    #distance graph, blue
    p1, = ax1.plot(d, '-b')
    ax1.set_ylabel('distance')
    ax1.set_ylim(0, 1.2)
    #test graph, green
    p2, = ax2.plot(test_for_plot, '-g')
    ax2.set_ylabel('original')
    ax2.set_ylim(0, 5000)
    
    plt.title("Nearest Neighbors")
    ax1.legend([p1, p2], ["distance", "original"])
    plt.savefig('./results/knn.png')
    plt.show()


def embed(lst, dim):
    emb = np.empty((0,dim), float) #np.empty는 초기화가 없는 값으로 배열을 반환
    for i in range(lst.size - dim + 1):
        tmp = np.array(lst[i:i+dim])[::-1].reshape((1,-1)) 
        emb = np.append( emb, tmp, axis=0)
    return emb

if __name__ == '__main__':
    main()
    