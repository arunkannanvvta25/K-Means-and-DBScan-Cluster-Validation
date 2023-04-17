import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import math
import datetime
from scipy.stats import entropy,iqr
from scipy import signal
from scipy.fftpack import fft


def calculate_psd_Feature(row):
    f,power_values=signal.periodogram(row)
    psd1=power_values[:5].mean()
    psd2=power_values[5:10].mean()
    psd3=power_values[10:16].mean()
    return psd1,psd2,psd3

def calculate_fft_Feature(row):
    fft_x = fft(row)
    power_spectrum = np.abs(fft_x)**2
    sorted_indices = np.argsort(power_spectrum)[::-1]
    sorted_indices = sorted_indices[1:]
    selected_indices = sorted_indices[:6]
    new_power=[]
    for ind in selected_indices:
        new_power.append(power_spectrum[ind])
    return new_power

def choose_bin(x, minimum_CarbValue, bins_Count):
    partition = float((x - minimum_CarbValue)/20)
    bin =  math.floor(partition)
    if bin == bins_Count:
        bin = bin - 1
    return bin

def ComputeGroundTruthMatrix(insulin_dataFrame, glucose_dataFrame):
    meals = []
    mealsDataFrame = pd.DataFrame()
    mealMatrix = pd.DataFrame()
    twoHourInSeconds = 60 * 60 * 2
    thirtyMinutesInSeconds = 30 * 60
    sensor_time_interval = 30
    groundTruthMatrix = []
    bins = []
    minimum_CarbValue = 0
    maximum_CarbValue = 0
    bins_Count = 0
    modified_InsulinDf = insulin_dataFrame.copy()
    modified_GlucoseDf = glucose_dataFrame.copy()
    valid_carb_input = modified_InsulinDf['BWZ Carb Input (grams)'].notna() & modified_InsulinDf['BWZ Carb Input (grams)'] != 0.0
    modified_InsulinDf = modified_InsulinDf.loc[valid_carb_input][['Date_Time', 'BWZ Carb Input (grams)']]
    modified_InsulinDf.set_index(['Date_Time'], inplace = True)
    modified_InsulinDf = modified_InsulinDf.sort_index().reset_index()
    valid_glucose = modified_GlucoseDf['Sensor Glucose (mg/dL)'].notna()
    modified_GlucoseDf = modified_GlucoseDf.loc[valid_glucose][['Date_Time', 'Sensor Glucose (mg/dL)']]
    modified_GlucoseDf.set_index(['Date_Time'], inplace = True)
    modified_GlucoseDf = modified_GlucoseDf.sort_index().reset_index()
    minimum_CarbValue = modified_InsulinDf['BWZ Carb Input (grams)'].min()
    maximum_CarbValue = modified_InsulinDf['BWZ Carb Input (grams)'].max()
    bins_Count = math.ceil((maximum_CarbValue - minimum_CarbValue) / 20)
    for i in range(len(modified_InsulinDf)):
        carb_input = modified_InsulinDf['BWZ Carb Input (grams)'][i]
        selected_bin = choose_bin(carb_input, minimum_CarbValue, bins_Count)
        bins.append(selected_bin)
    
    modified_InsulinDf['bin'] = bins

    for i in range(0, len(modified_InsulinDf)-1):
        time_diff_seconds = (modified_InsulinDf.iloc[i + 1]['Date_Time'] - modified_InsulinDf.iloc[i]['Date_Time']).total_seconds()
        if(time_diff_seconds > twoHourInSeconds):
            meals.append(True)
        else:
            meals.append(False)
        
    meals.append(True)
    mealsDataFrame = modified_InsulinDf[meals]
    
    for i in range(len(mealsDataFrame)):
        lower_bound = mealsDataFrame.iloc[i]['Date_Time'] - datetime.timedelta(seconds=thirtyMinutesInSeconds)
        upper_bound = mealsDataFrame.iloc[i]['Date_Time'] + datetime.timedelta(seconds=twoHourInSeconds)
        is_within_bounds = (modified_GlucoseDf['Date_Time'] >= lower_bound) & (modified_GlucoseDf['Date_Time'] < upper_bound)
        bin = mealsDataFrame.iloc[i]['bin']
        filtered_glucose_dataFrame = modified_GlucoseDf[is_within_bounds]        
        if len(filtered_glucose_dataFrame.index) == sensor_time_interval:
            filtered_glucose_dataFrame = filtered_glucose_dataFrame.T
            filtered_glucose_dataFrame.drop('Date_Time', inplace=True)            
            filtered_glucose_dataFrame.reset_index(drop=True, inplace=True)
            filtered_glucose_dataFrame.columns = list(range(1, 31))          
            mealMatrix = pd.concat([mealMatrix, filtered_glucose_dataFrame], ignore_index=True)
            groundTruthMatrix.append(bin)
    mealMatrix = mealMatrix.apply(pd.to_numeric)
    groundTruthMatrix = np.array(groundTruthMatrix)
    return mealMatrix, groundTruthMatrix, bins_Count

def buildClusterMatrix(k, clusters, ground_truth):
    cluster_matrix = np.zeros((k, k))
    for i, j in enumerate(ground_truth):
        row = clusters[i]
        col = j
        cluster_matrix[row][col] += 1
    return cluster_matrix

def ComputeEntropy(groundTruthMatrix):
    groundTruthMatrix_sum = groundTruthMatrix.sum()
    bins = groundTruthMatrix.shape[0]
    entropyValue = 0
    c_Sum = 0
    clusterEntropyArray = []
    for i in range(bins):
        c_Sum = np.sum(groundTruthMatrix[i])
        if c_Sum == 0:
            continue
        for j in range(bins):
            if groundTruthMatrix[i,j] == 0:
                continue
            col_sum = groundTruthMatrix[i,j] / c_Sum
            entropy = -1 * col_sum * np.log2(col_sum)
            entropyValue = entropyValue + entropy
        clusterEntropyArray.append((c_Sum / groundTruthMatrix_sum) * entropyValue)
    return np.sum(clusterEntropyArray)

def computeSSE(dbscan,scaledFeature_Matrix):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    labels = dbscan.labels_

    # compute SSE for each cluster
    sse = 0
    for i in np.unique(labels):
        cluster_points = scaledFeature_Matrix[labels == i]
        core_distances = np.linalg.norm(cluster_points[core_mask[labels == i]] - cluster_points[:, None], axis=2)
        sse += np.sum(np.min(core_distances, axis=0) ** 2)
    return sse

def ComputeFeatureMatrix(df):
    features = pd.DataFrame()

    for i in range(0, df.shape[0]):
        x = df.iloc[i, :].tolist()
        fft_powerValues=calculate_fft_Feature(x)
        psd1,psd2,psd3=calculate_psd_Feature(x)
        features = features.append({
            "VelocityMin": np.diff(x).min(),
            "VelocityMax": np.diff(x).max(),
            "VelocityMean": np.diff(x).mean(),
            "AccelerationMin":np.diff(np.diff(x)).min(),
            "AccelerationMax":np.diff(np.diff(x)).max(),
            "AccelerationMean":np.diff(np.diff(x)).mean(),
            "Entropy": entropy(x, base=2),
            "iqr":iqr(x),
            "fft1": fft_powerValues[0],
            "fft2": fft_powerValues[1],
            "fft3": fft_powerValues[2],
            "fft4": fft_powerValues[3],
            "fft5": fft_powerValues[4],
            "fft6": fft_powerValues[5],
            "psd1":psd1,
            "psd2":psd2,
            "psd3":psd3
        },
        ignore_index=True
        )
    return features
def main(): 

    insulin_data = pd.read_csv('InsulinData.csv', parse_dates=[['Date','Time']], keep_date_col=True, low_memory=False)
    insulin_dataFrame = insulin_data[['Date_Time', 'Index', 'BWZ Carb Input (grams)']]
    insulin_dataFrame.loc[:, 'Index']

    glucose_data = pd.read_csv('CGMData.csv', parse_dates=[['Date','Time']], keep_date_col=True, low_memory=False)
    glucose_dataFrame = glucose_data[['Date_Time', 'Sensor Glucose (mg/dL)']]

    mealMatrix, groundTruthMatrix, bins_Count = ComputeGroundTruthMatrix(insulin_dataFrame, glucose_dataFrame)
    feature_matrix = ComputeFeatureMatrix(mealMatrix).to_numpy()
    scaler = StandardScaler()
    scaledFeature_Matrix = scaler.fit_transform(feature_matrix)

    #K_Means
    kmeans = KMeans(n_clusters=bins_Count, n_init=10,random_state=0).fit(scaledFeature_Matrix)
    y_true=groundTruthMatrix
    y_pred = kmeans.labels_
    kmeans_sse = kmeans.inertia_
    cm = confusion_matrix(y_true, y_pred)
    kmeans_purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    ClusterMatrix = buildClusterMatrix(int(bins_Count), kmeans.labels_, groundTruthMatrix)
    #kmeans_entropy = ComputeEntropy(kmeans_groundTruthMatrix)
    kmeans_entropy = -np.sum((np.sum(cm, axis=0)/np.sum(cm)) * np.log2(np.sum(cm, axis=0)/np.sum(cm)))

    #DBSCAN
    dbscan = DBSCAN(eps=0.522, min_samples=3,metric='euclidean').fit(scaledFeature_Matrix)
    y_pred = dbscan.fit_predict(scaledFeature_Matrix)
    cm_dbscan = confusion_matrix(y_true,y_pred)
    dbscan_labels = dbscan.labels_
    dbscan_sse =computeSSE(dbscan,scaledFeature_Matrix)    
    dbscan_groundTruthMatrix = buildClusterMatrix(int(bins_Count), dbscan_labels, groundTruthMatrix)
    dbscan_entropy = ComputeEntropy(dbscan_groundTruthMatrix)
    dbscan_purity = np.sum(np.amax(cm_dbscan, axis=0)) / np.sum(cm_dbscan)   

    result = pd.DataFrame(
        [
            [
                kmeans_sse,dbscan_sse,kmeans_entropy,dbscan_entropy,kmeans_purity,dbscan_purity
            ]
        ],
        columns=[
            "SSE for KMeans","SSE for DBSCAN","Entropy for KMeans","Entropy for DBSCAN","Purity for KMeans","Purity for DBSCAN"
        ],
    )
    result = result.fillna(0)
    result.to_csv("Result.csv", index=False, header=None)

if __name__ == '__main__':
    main()