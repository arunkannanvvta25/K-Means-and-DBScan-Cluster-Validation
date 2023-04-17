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


def ComputeFeatureMatrix(df):
    features = pd.DataFrame()

    for i in range(0, df.shape[0]):
        x = df.iloc[i, :].tolist()
        fft_powerValues=calculate_fft(x)
        psd1,psd2,psd3=calculate_psd(x)
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

def calculate_psd(row):
    f,power_values=signal.periodogram(row)
    psd1=power_values[:5].mean()
    psd2=power_values[5:10].mean()
    psd3=power_values[10:16].mean()
    return psd1,psd2,psd3

def calculate_fft(row):
    fft_x = fft(row)
    power_spectrum = np.abs(fft_x)**2
    sorted_indices = np.argsort(power_spectrum)[::-1]
    sorted_indices = sorted_indices[1:]
    selected_indices = sorted_indices[:6]
    new_power=[]
    for ind in selected_indices:
        new_power.append(power_spectrum[ind])
    return new_power

def choose_bin(x, min_carb, bins_Count):
    partition = float((x - min_carb)/20)
    bin =  math.floor(partition)
    if bin == bins_Count:
        bin = bin - 1
    return bin

def ComputeGroundTruthMatrix(insulin_dataFrame, glucose_dataFrame):
    meals = []
    meals_df = pd.DataFrame()
    meal_matrix = pd.DataFrame()
    two_hours = 60 * 60 * 2
    thirty_min = 30 * 60
    sensor_time_interval = 30

    groundTruthMatrix = []
    bins = []
    min_carb = 0
    max_carb = 0
    bins_Count = 0

    processed_insulin_dataFrame = insulin_dataFrame.copy()
    processed_glucose_dataFrame = glucose_dataFrame.copy()

    valid_carb_input = processed_insulin_dataFrame['BWZ Carb Input (grams)'].notna() & processed_insulin_dataFrame['BWZ Carb Input (grams)'] != 0.0
    processed_insulin_dataFrame = processed_insulin_dataFrame.loc[valid_carb_input][['Date_Time', 'BWZ Carb Input (grams)']]
    processed_insulin_dataFrame.set_index(['Date_Time'], inplace = True)
    processed_insulin_dataFrame = processed_insulin_dataFrame.sort_index().reset_index()

    valid_glucose = processed_glucose_dataFrame['Sensor Glucose (mg/dL)'].notna()
    processed_glucose_dataFrame = processed_glucose_dataFrame.loc[valid_glucose][['Date_Time', 'Sensor Glucose (mg/dL)']]
    processed_glucose_dataFrame.set_index(['Date_Time'], inplace = True)
    processed_glucose_dataFrame = processed_glucose_dataFrame.sort_index().reset_index()

    min_carb = processed_insulin_dataFrame['BWZ Carb Input (grams)'].min()
    max_carb = processed_insulin_dataFrame['BWZ Carb Input (grams)'].max()
    bins_Count = math.ceil((max_carb - min_carb) / 20)

    for i in range(len(processed_insulin_dataFrame)):
        carb_input = processed_insulin_dataFrame['BWZ Carb Input (grams)'][i]
        selected_bin = choose_bin(carb_input, min_carb, bins_Count)
        bins.append(selected_bin)
    
    processed_insulin_dataFrame['bin'] = bins

    for i in range(0, len(processed_insulin_dataFrame)-1):
        time_diff_seconds = (processed_insulin_dataFrame.iloc[i + 1]['Date_Time'] - processed_insulin_dataFrame.iloc[i]['Date_Time']).total_seconds()
        if(time_diff_seconds > two_hours):
            meals.append(True)
        else:
            meals.append(False)
        
    meals.append(True)
    meals_df = processed_insulin_dataFrame[meals]
    
    for i in range(len(meals_df)):
        lower_bound = meals_df.iloc[i]['Date_Time'] - datetime.timedelta(seconds=thirty_min)
        upper_bound = meals_df.iloc[i]['Date_Time'] + datetime.timedelta(seconds=two_hours)
        is_within_bounds = (processed_glucose_dataFrame['Date_Time'] >= lower_bound) & (processed_glucose_dataFrame['Date_Time'] < upper_bound)
        bin = meals_df.iloc[i]['bin']
        filtered_glucose_dataFrame = processed_glucose_dataFrame[is_within_bounds]
        
        if len(filtered_glucose_dataFrame.index) == sensor_time_interval:
            filtered_glucose_dataFrame = filtered_glucose_dataFrame.T
            filtered_glucose_dataFrame.drop('Date_Time', inplace=True)
            
            filtered_glucose_dataFrame.reset_index(drop=True, inplace=True)
            filtered_glucose_dataFrame.columns = list(range(1, 31))
            
            
            meal_matrix = pd.concat([meal_matrix, filtered_glucose_dataFrame], ignore_index=True)
            groundTruthMatrix.append(bin)

    meal_matrix = meal_matrix.apply(pd.to_numeric)
    groundTruthMatrix = np.array(groundTruthMatrix)

    return meal_matrix, groundTruthMatrix, bins_Count

def buildClusterMatrix(k, clusters, ground_truth):
    cluster_matrix = np.zeros((k, k))
    for i, j in enumerate(ground_truth):
        row = clusters[i]
        col = j
        cluster_matrix[row][col] += 1
    return cluster_matrix

def ComputeEntropy(gtm):
    gtm_sum = gtm.sum()
    bins = gtm.shape[0]
    cluster_entropy = 0
    cluster_sum = 0
    cluster_entropies = []
    for i in range(bins):
        cluster_sum = np.sum(gtm[i])
        if cluster_sum == 0:
            continue
        for j in range(bins):
            if gtm[i,j] == 0:
                continue
            col_sum = gtm[i,j] / cluster_sum
            entropy = -1 * col_sum * np.log2(col_sum)
            cluster_entropy = cluster_entropy + entropy
        cluster_entropies.append((cluster_sum / gtm_sum) * cluster_entropy)
    return np.sum(cluster_entropies)

def squaredError(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred)
    counts = np.count_nonzero(y_true)
    return MSE*counts

def main(): 

    insulin_data = pd.read_csv('InsulinData.csv', parse_dates=[['Date','Time']], keep_date_col=True, low_memory=False)
    insulin_dataFrame = insulin_data[['Date_Time', 'Index', 'BWZ Carb Input (grams)']]
    insulin_dataFrame.loc[:, 'Index']

    glucose_data = pd.read_csv('CGMData.csv', parse_dates=[['Date','Time']], keep_date_col=True, low_memory=False)
    glucose_dataFrame = glucose_data[['Date_Time', 'Sensor Glucose (mg/dL)']]

    meal_matrix, groundTruthMatrix, bins_Count = ComputeGroundTruthMatrix(insulin_dataFrame, glucose_dataFrame)
    feature_matrix = ComputeFeatureMatrix(meal_matrix).to_numpy()
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
    #kmeans_entropy = ComputeEntropy(kmeans_gtm)
    kmeans_entropy = -np.sum((np.sum(cm, axis=0)/np.sum(cm)) * np.log2(np.sum(cm, axis=0)/np.sum(cm)))

    #DBSCAN
    dbscan = DBSCAN(eps=0.522, min_samples=3).fit(scaledFeature_Matrix)
    y_pred = dbscan.fit_predict(scaledFeature_Matrix)
    cm_dbscan = confusion_matrix(y_true,y_pred)
    dbscan_sse   = squaredError(y_true, y_pred)
    dbscan_labels = dbscan.labels_
    dbscan_gtm = buildClusterMatrix(int(bins_Count), dbscan_labels, groundTruthMatrix)
    dbscan_entropy = ComputeEntropy(dbscan_gtm)
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