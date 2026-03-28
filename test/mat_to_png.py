import h5py
import numpy as np
import cv2
import os

def convert_mat_to_png_strict_math(mat_filepath, png_filepath):
    # 1. 读取复数矩阵
    with h5py.File(mat_filepath, 'r') as f:
        data_struct = f['SAR_Data_Complex']
        real_part = data_struct['complex_matrix']['real'][:]
        imag_part = data_struct['complex_matrix']['imag'][:]
        complex_mat = real_part + 1j * imag_part
        complex_mat = complex_mat.T  # 矩阵转置对齐坐标系

    # 2. 取绝对幅度并做几何翻转
    amplitude = np.abs(complex_mat)
    amplitude = np.flipud(amplitude)

    # 3. 严格复刻 MATLAB 的数学映射逻辑：乘以 0.01 并硬截断
    # 相当于设定固定阈值 vmax = 100
    amplitude_scaled = amplitude * 0.01
    amplitude_clipped = np.clip(amplitude_scaled, 0.0, 1.0)

    # 4. 线性量化到 [0, 255] 的 uint8 空间
    img_8bit = (amplitude_clipped * 255.0).astype(np.uint8)

    # 5. 保存图像
    cv2.imwrite(png_filepath, img_8bit)
    print("报告学长！已严格按照固定比例缩放完成矩阵映射并保存！")

if __name__ == '__main__':
    MAT_INPUT_PATH = os.path.join(os.path.dirname(__file__), "mat", "rigui_001.mat")
    PNG_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "png", "rigui_001.png")
    convert_mat_to_png_strict_math(MAT_INPUT_PATH, PNG_OUTPUT_PATH)