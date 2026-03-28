import h5py
import numpy as np

def load_sar_complex(file_path):
    """提取 HDF5 格式的复数基带数据并重构共轭张量"""
    with h5py.File(file_path, 'r') as f:
        data_struct = f['SAR_Data_Complex']
        real_part = data_struct['complex_matrix']['real'][:]
        imag_part = data_struct['complex_matrix']['imag'][:]
        return (real_part + 1j * imag_part).T

def adaptive_peak_search_and_extract(complex_mat, init_y, init_x, search_window=128, feature_window=64):
    """基于先验坐标计算局部电磁散射特征 (SCR 与相位方差)"""
    h_mat, w_mat = complex_mat.shape
    y_s, y_e = max(0, init_y - search_window//2), min(h_mat, init_y + search_window//2)
    x_s, x_e = max(0, init_x - search_window//2), min(w_mat, init_x + search_window//2)
    
    search_roi = complex_mat[y_s:y_e, x_s:x_e]
    local_y, local_x = np.unravel_index(np.argmax(np.abs(search_roi)), search_roi.shape)
    peak_y, peak_x = y_s + local_y, x_s + local_x
    
    fy_s, fy_e = max(0, peak_y - feature_window//2), min(h_mat, peak_y + feature_window//2)
    fx_s, fx_e = max(0, peak_x - feature_window//2), min(w_mat, peak_x + feature_window//2)
    
    feature_roi = complex_mat[fy_s:fy_e, fx_s:fx_e]
    amplitude, phase = np.abs(feature_roi), np.angle(feature_roi)
    
    fh, fw = amplitude.shape
    center_mask = np.zeros((fh, fw), dtype=bool)
    center_mask[fh//4 : 3*fh//4, fw//4 : 3*fw//4] = True
    
    P_t = np.mean(amplitude[center_mask]**2) + 1e-9 
    P_c = np.mean(amplitude[~center_mask]**2) + 1e-9 
    return peak_y, peak_x, 10 * np.log10(P_t / P_c), np.var(phase[center_mask])