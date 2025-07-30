# sender_save_images.py
import os
import zlib
import base64
import json
import hashlib
import qrcode
import sys
from PIL import Image

# --- 配置参数 ---
CHUNK_SIZE = 1024  # 每块数据的大小（字节），可以调整
OUTPUT_DIR = "qr_codes_output" # 输出文件夹名称

def main(filepath):
    # 1. 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"错误：文件 '{filepath}' 不存在。")
        return

    # 2. 读取和压缩文件
    print("正在读取和压缩文件...")
    with open(filepath, 'rb') as f:
        file_data = f.read()
    
    compressed_data = zlib.compress(file_data)
    
    # 3. 计算文件的MD5校验和
    file_hash = hashlib.md5(file_data).hexdigest()
    
    # 4. Base64编码
    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    
    # 5. 将数据分块
    data_chunks = [encoded_data[i:i + CHUNK_SIZE] for i in range(0, len(encoded_data), CHUNK_SIZE)]
    total_chunks = len(data_chunks)
    
    # 6. 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建文件夹: {OUTPUT_DIR}")
    else:
        print(f"输出文件夹: {OUTPUT_DIR}")


    print(f"文件 '{os.path.basename(filepath)}' 已准备好。")
    print(f"文件大小: {len(file_data) / 1024:.2f} KB")
    print(f"压缩后大小: {len(compressed_data) / 1024:.2f} KB")
    print(f"MD5 校验和: {file_hash}")
    print(f"总共将生成 {total_chunks} 个QR码图片。")
    print("\n开始生成图片...")

    # 7. 循环生成并保存所有QR码图片
    # 使用4位数字补齐（如0001, 0002），确保文件名按顺序排列
    num_digits = len(str(total_chunks - 1)) 

    for i, chunk in enumerate(data_chunks):
        packet = {
            "f": os.path.basename(filepath), # 文件名
            "h": file_hash,                  # 完整文件的MD5
            "t": total_chunks,               # 总块数
            "p": i,                          # 当前块编号
            "d": chunk                       # 数据块
        }
        packet_str = json.dumps(packet)
        
        # 生成QR码PIL图像对象
        qr_img = qrcode.make(packet_str)
        
        # 定义文件名并保存
        filename = f"qr_code_{i:0{num_digits}d}.png"
        filepath_out = os.path.join(OUTPUT_DIR, filename)
        qr_img.save(filepath_out)
        
        # 打印进度
        print(f"已保存: {filename} ({i + 1}/{total_chunks})")

    print(f"\n✅ 所有QR码图片生成完毕，已保存在 '{OUTPUT_DIR}' 文件夹中。")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法: python sender_save_images.py <你的文件名>")
        sys.exit(1)
    
    file_to_send = sys.argv[1]
    main(file_to_send)