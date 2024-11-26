import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import csv
import platform


def set_korean_font():
    """한글 폰트 설정"""
    if platform.system() == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    elif platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Linux':
        plt.rc('font', family='NanumGothic')

    # 음수 표시를 위한 설정
    plt.rc('axes', unicode_minus=False)


def analyze_audio(file_path, freq_min=0, freq_max=5000, time_interval=0.05, nfft=441000, csv_output="output.csv"):
    """
    오디오 파일을 분석하고 특정 주파수 범위의 스펙트럼을 표시하고 스펙트로그램을 추가하며 CSV 파일에 기록합니다.

    Parameters:
    file_path (str): WAV 파일 경로
    freq_min (int): 표시할 최소 주파수 (Hz)
    freq_max (int): 표시할 최대 주파수 (Hz)
    csv_output (str): 출력할 CSV 파일 경로
    """
    # 한글 폰트 설정
    set_korean_font()

    # WAV 파일 읽기
    sample_rate, data = wavfile.read(file_path)

    # 스테레오인 경우 모노로 변환
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    # 데이터 정규화
    data = data / np.max(np.abs(data))

    # FFT 수행
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, 1 / sample_rate)

    # 양의 주파수만 선택
    positive_freq_mask = xf > 0
    xf = xf[positive_freq_mask]
    yf = np.abs(yf[positive_freq_mask])

    # 주파수 범위 필터링
    freq_mask = (xf >= freq_min) & (xf <= freq_max)
    # 시간 단위 계산 (윈도우의 길이)
    window_size = int(sample_rate * time_interval)
    num_windows = len(data) // window_size

    # CSV 파일 작성
    with open(csv_output, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Time (s)", "Peak Frequency (Hz)", "Amplitude"])

        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window_data = data[start:end]

            # # FFT 수행
            window_yf = fft(window_data, n=nfft)
            window_xf = fftfreq(nfft, 1 / sample_rate)

            # 양의 주파수만 선택
            positive_freq_mask = window_xf > 0
            window_xf = window_xf[positive_freq_mask]
            window_yf = np.abs(window_yf[positive_freq_mask])

            # 양의 주파수만 선택
            positive_freq_mask = window_xf > 0
            window_xf = window_xf[positive_freq_mask]
            window_yf = np.abs(window_yf[positive_freq_mask])

            # 주파수 범위 필터링
            window_freq_mask = (window_xf >= freq_min) & (window_xf <= freq_max)
            window_xf_filtered = window_xf[window_freq_mask]
            window_yf_filtered = window_yf[window_freq_mask]

            # 가장 큰 진폭을 가진 주파수 찾기
            if len(window_yf_filtered) > 0:
                max_index = np.argmax(window_yf_filtered)
                peak_frequency = window_xf_filtered[max_index]
                amplitude = window_yf_filtered[max_index]
                csv_writer.writerow([round(i * time_interval, 3), round(peak_frequency, 3), round(amplitude, 3)])
    # CSV 파일에서 시간과 주파수 데이터를 읽어와 그래프 그리기
    times = []
    peak_frequencies = []

    with open(csv_output, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # 헤더 스킵

        for row in csv_reader:
            time = float(row[0])
            peak_frequency = float(row[1])
            times.append(time)
            peak_frequencies.append(peak_frequency)

    # 그래프 생성
    plt.figure(figsize=(8, 8))

    # 시간에 따른 피크 주파수 그래프
    plt.subplot(3, 1, 1)
    plt.plot(times, peak_frequencies, color='blue')
    plt.title('시간에 따른 피크 주파수')
    plt.xlabel('시간 (초)')
    plt.ylabel('피크 주파수 (Hz)')
    plt.ylim(freq_min, freq_max)
    plt.grid(True)

    # 주파수 스펙트럼
    plt.subplot(3, 1, 2)
    plt.plot(window_xf_filtered, window_yf_filtered)
    plt.title('주파수 스펙트럼')
    plt.xlabel('주파수 (Hz)')
    plt.ylabel('진폭')
    plt.grid(True)

    # 원본 파형
    plt.subplot(3, 1, 3)
    time = np.arange(len(data)) / sample_rate
    plt.plot(time, data)
    plt.title('원본 파형')
    plt.xlabel('시간 (초)')
    plt.ylabel('진폭')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # 사용 예시
    for i in range(11, 15):
        print(i)
        file = f"경기북과학고등학교 {i}"
        file_path = f"audios/{file}.wav"
        output_path = f"datas/{file}.csv"
        analyze_audio(file_path, freq_min=7750, freq_max=8250, time_interval=0.05, csv_output=output_path)


if __name__ == "__main__":
    main()
