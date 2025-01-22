import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def calc_mean_erp(trial_points, ecog_data):
    """
    This function calculates the average Event-Related Potential (ERP) for each finger movement
    Parameters:
    trial_points + ecog_data 
    Returns:
    fingers_erp_mean + 5x1201 matrix containing average brain responses for each finger
    """
    
    trial_data = pd.read_csv(trial_points).astype(int)
    brain_data = pd.read_csv(ecog_data).values.flatten()
    
    fingers_erp_mean = np.zeros((5, 1201))
    
    for finger in range(1, 6):
        finger_starts = trial_data[trial_data.iloc[:, 2] == finger].iloc[:, 0]
        all_windows = np.zeros((len(finger_starts), 1201))
        
        for i, start in enumerate(finger_starts):
            window_start = start - 200
            window_end = start + 1001
            
            if window_start >= 0 and window_end <= len(brain_data):
                all_windows[i] = brain_data[window_start:window_end]
        
        fingers_erp_mean[finger-1] = np.mean(all_windows, axis=0)
    
    plt.figure(figsize=(12, 8))
    time = np.arange(-200, 1001)
    for finger in range(5):
        plt.plot(time, fingers_erp_mean[finger], label=f'Finger {finger+1}')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Brain Response')
    plt.title('Average Brain Response for Each Finger Movement')
    plt.legend()
    plt.grid(True)
    plt.savefig('finger_responses.png')
    plt.close()
    
    return fingers_erp_mean

if __name__ == "__main__":
    fingers_erp_mean = calc_mean_erp('events_file_ordered.csv', 'brain_data_channel_one.csv')
