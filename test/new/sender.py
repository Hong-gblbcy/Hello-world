# sender_save_images_optimized.py
import os
import lzma  # 【优化】使用lzma库，压缩率更高
import base64
import json
import hashlib
import qrcode
import sys

# --- 配置参数 ---
# 【优化】显著增大数据块，因为压缩和纠错等级优化后，单个QR码能容纳更多数据
CHUNK_SIZE = 2000
OUTPUT_DIR = "qr_codes_output_optimized"

def main(filepath):
    if not os.path.exists(filepath):
        print(f"错误：文件 '{filepath}' 不存在。")
        return

    print("正在读取文件...")
    with open(filepath, 'rb') as f:
        file_data = f.read()
    
    # 【优化】使用lzma进行高效压缩
    print("正在使用LZMA进行高强度压缩...")
    compressed_data = lzma.compress(file_data)
    
    file_hash = hashlib.md5(file_data).hexdigest()
    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    data_chunks = [encoded_data[i:i + CHUNK_SIZE] for i in range(0, len(encoded_data), CHUNK_SIZE)]
    total_chunks = len(data_chunks)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建文件夹: {OUTPUT_DIR}")
    else:
        # 清空旧文件，避免混淆
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))
        print(f"已清空并使用文件夹: {OUTPUT_DIR}")

    print("-" * 30)
    print(f"文件 '{os.path.basename(filepath)}'")
    print(f"原始大小: {len(file_data) / 1024:.2f} KB")
    print(f"LZMA压缩后大小: {len(compressed_data) / 1024:.2f} KB")
    print(f"MD5 校验和: {file_hash}")
    print(f"将生成 {total_chunks} 个优化后的QR码图片。")
    print("-" * 30)
    print("\n开始生成图片...")

    num_digits = len(str(total_chunks - 1)) 

    for i, chunk in enumerate(data_chunks):
        packet = {
            "f": os.path.basename(filepath),
            "h": file_hash,
            "t": total_chunks,
            "p": i,
            "d": chunk
        }
        packet_str = json.dumps(packet)
        
        # 【优化】创建QR对象，并设置纠错等级为L(最低)，以获取最大容量
        qr = qrcode.QRCode(
            version=None, # 自动判断版本大小
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(packet_str)
        qr.make(fit=True)

        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        filename = f"qr_code_{i:0{num_digits}d}.png"
        filepath_out = os.path.join(OUTPUT_DIR, filename)
        qr_img.save(filepath_out)
        
        print(f"已保存: {filename} ({i + 1}/{total_chunks})")

    print(f"\n✅ 所有QR码图片生成完毕，已保存在 '{OUTPUT_DIR}' 文件夹中。")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法: python sender_save_images_optimized.py <你的文件名>")
        sys.exit(1)
    
    file_to_send = sys.argv[1]
    main(file_to_send)