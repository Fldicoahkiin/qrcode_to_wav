import matplotlib.pyplot as plt
import numpy as np
import qrcode
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import spectrogram


# --------------------------
# 二维码生成函数
# --------------------------
def text_to_qrmatrix(text, box_size=10, version=3):
    """生成二维码矩阵"""
    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=box_size,
        border=4
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('L')
    width, height = img.size
    return [[0 if img.getpixel((x, y)) > 127 else 1 for x in range(width)] for y in range(height)]


# --------------------------
# 文本图像生成函数
# --------------------------
def text_to_matrix(text, size=29):
    """生成黑底白字的横向文本矩阵"""
    # 创建横向画布（宽度为正方形尺寸的2倍）
    canvas_width = size * 2
    canvas_height = size
    img = Image.new('L', (canvas_width, canvas_height), 0)  # 黑色背景
    draw = ImageDraw.Draw(img)

    # 强制使用默认字体
    font = ImageFont.load_default()

    # 动态确定最大字体尺寸
    max_font_size = 1
    while True:
        test_font = ImageFont.load_default().font_variant(size=max_font_size)

        # 使用textbbox方法
        bbox = draw.textbbox((0, 0), text, font=test_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 同时满足高度和宽度限制
        if text_height > canvas_height * 0.8 or text_width > canvas_width * 0.9:
            break
        max_font_size += 1
    max_font_size = max(1, max_font_size - 2)

    # 创建最终字体
    final_font = ImageFont.load_default().font_variant(size=max_font_size)

    # 获取最终文本尺寸
    bbox = draw.textbbox((0, 0), text, font=final_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 居中绘制白色文本
    x = (canvas_width - text_width) // 2
    y = (canvas_height - text_height) // 2
    draw.text((x, y), text, fill=255, font=final_font)  # 白色文字

    # 裁剪为正方形
    left = (canvas_width - size) // 2
    right = left + size
    img = img.crop((left, 0, right, size))

    # 生成反转矩阵（文字区域为1，背景为0）
    return [
        [1 if img.getpixel((x, y)) > 127 else 0  # 反转阈值判断
         for x in range(size)]
        for y in range(size)
    ]


# --------------------------
# 核心编码函数
# --------------------------
def generate_square_spectral_qr(qr_matrix, output_file="qr_square.wav", f_start=100.0, f_end=4000.0):
    size = len(qr_matrix)

    # 时间参数设置
    TIME_PER_PIXEL = 0.04
    total_time = size * TIME_PER_PIXEL

    # 计算频率参数
    bandwidth_per_pixel = (f_end - f_start) / (size - 1) if size > 1 else 0

    # 生成音频信号
    sample_rate = 44100
    t_total = np.linspace(0, total_time, int(sample_rate * total_time))
    audio = np.zeros_like(t_total)

    # 时频编码
    for y in range(size):
        for x in range(size):
            if qr_matrix[y][x] == 1:
                freq = f_start + x * bandwidth_per_pixel
                t_start = y * TIME_PER_PIXEL
                t_end = (y + 1) * TIME_PER_PIXEL

                mask = (t_total >= t_start) & (t_total < t_end)
                time_segment = t_total[mask]

                # 生成带相移的正弦波
                tone = np.sin(2 * np.pi * freq * time_segment + np.pi / 4)

                # 应用凯撒窗
                window = np.kaiser(len(time_segment), beta=14)
                audio[mask] += window * tone * 0.6

    # 标准化并保存
    audio = np.asarray(audio)
    audio /= np.max(np.abs(audio)) * 1.1
    sf.write(output_file, audio, sample_rate)
    return audio, (f_start, f_end)


# --------------------------
# 优化可视化函数
# --------------------------
def plot_square_spectrum(audio, freq_range):
    sample_rate = 44100
    nperseg = 2048

    try:
        f, t, Sxx = spectrogram(audio, fs=sample_rate,
                                nperseg=nperseg,
                                noverlap=int(nperseg * 0.8),
                                window='blackmanharris',
                                scaling='density',
                                mode='psd')

        Sxx = np.clip(Sxx, 1e-25, None)
        log_Sxx = 10 * np.log10(Sxx)

        db_range = 60
        vmin = np.percentile(log_Sxx, 20)
        vmax = vmin + db_range

        plt.figure(figsize=(12, 6))  # 调整为宽屏比例
        ax = plt.gca()
        mesh = ax.pcolormesh(t, f, log_Sxx,
                             shading='gouraud',
                             cmap='inferno',  # 改用更清晰的配色
                             vmin=vmin,
                             vmax=vmax)

        # 精确设置频率范围
        plt.ylim(max(0, freq_range[0] - 100), freq_range[1] + 100)  # 自动扩展边界
        plt.xlim(0, len(audio) / sample_rate)

        # 优化比例计算
        time_span = t[-1] - t[0]
        freq_span = freq_range[1] - freq_range[0]
        ax.set_aspect(time_span / freq_span * 0.3)  # 手动调整显示比例

        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(mesh, label='Power (dB)')
        plt.title(f'Spectral Code ({freq_range[0]:.0f}-{freq_range[1]:.0f} Hz)')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"可视化错误: {str(e)}")
        raise


# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    # 用户输入文本
    input_text = input("请输入要编码的文本内容（建议长度<150字符）: ").strip()

    # 选择生成模式
    mode = input("请选择生成类型（1为二维码，2为文本）: ").strip()
    if mode not in ['1', '2']:
        print("无效选择，请输入1或2")
        exit(1)

    try:
        # 根据模式生成矩阵
        if mode == '1':
            qr_matrix = text_to_qrmatrix(input_text)
        else:
            qr_matrix = text_to_matrix(input_text)
    except Exception as e:
        print(f"矩阵生成错误: {str(e)}")
        exit(1)

    # 频率范围输入提示
    try:
        default_start = 100.0
        default_end = 4000.0

        start_input = input(f"请输入起始频率（Hz，默认{default_start}）: ").strip()
        end_input = input(f"请输入结束频率（Hz，默认{default_end}）: ").strip()

        # 处理输入值
        f_start = float(start_input) if start_input else default_start
        f_end = float(end_input) if end_input else default_end

        # 验证频率范围
        if f_start >= f_end:
            raise ValueError("起始频率必须小于结束频率")
        if f_end > 22050:
            raise ValueError("结束频率不能超过采样率的一半（22050Hz）")
        if f_start < 20:
            print("警告：极低频可能无法被设备准确播放")

    except ValueError as e:
        print(f"输入错误: {str(e)}")
        exit(1)

    try:
        # 生成音频
        audio, freq_range = generate_square_spectral_qr(
            qr_matrix,
            f_start=f_start,
            f_end=f_end
        )

        # 显示频谱
        plot_square_spectrum(audio, freq_range)

        print(f"生成成功！频率范围：{freq_range[0]:.1f}-{freq_range[1]:.1f}Hz")
        print(f"音频文件已保存为 qr_square.wav")

    except Exception as e:
        print(f"运行错误: {str(e)}")
        exit(1)
