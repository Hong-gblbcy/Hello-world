# receiver_from_images.py
import os
import cv2
import zlib
import base64
import json
import hashlib
import sys
from pyzbar import pyzbar

def main(input_dir):
    # 1. 检查文件夹是否存在
    if not os.path.isdir(input_dir):
        print(f"错误：文件夹 '{input_dir}' 不存在。")
        return

    # 2. 获取文件夹中所有图片文件并排序
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print(f"错误：文件夹 '{input_dir}' 中没有找到图片文件。")
            return
    except Exception as e:
        print(f"读取文件夹错误: {e}")
        return

    print(f"将在文件夹 '{input_dir}' 中查找 {len(image_files)} 个图片文件...")

    received_chunks = {}
    total_chunks = None
    file_hash = None
    filename = None

    # 3. 循环读取和解码每张图片
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        print(f"正在处理: {image_file}...", end='')
        
        # 读取图片文件
        image = cv2.imread(image_path)
        if image is None:
            print(" [读取失败]")
            continue

        # 解码QR码
        qrcodes = pyzbar.decode(image)
        if not qrcodes:
            print(" [未找到QR码]")
            continue
        
        # 通常一张图只有一个QR码
        qr_data_str = qrcodes[0].data.decode('utf-8')
        
        try:
            # 4. 解析数据包
            packet = json.loads(qr_data_str)
            part_index = packet['p']

            if total_chunks is None: # 从第一张有效图片中获取元数据
                total_chunks = packet['t']
                file_hash = packet['h']
                filename = packet['f']
                print(f"\n检测到文件传输任务: {filename} (共 {total_chunks} 块)")
            
            # 5. 收集数据块
            if part_index not in received_chunks:
                received_chunks[part_index] = packet['d']
                print(f" [成功解码第 {part_index + 1} 块]")
            else:
                print(" [块重复，已忽略]")

        except (json.JSONDecodeError, KeyError):
            print(" [QR码内容格式错误]")
            continue

    # 6. 检查是否接收完成
    if total_chunks is None:
        print("\n未成功解码任何有效的数据块。")
        return
        
    if len(received_chunks) == total_chunks:
        print(f"\n所有 {total_chunks} 个数据块已成功解码！正在重组文件...")
        
        # 7. 按照顺序重组
        sorted_chunks = [received_chunks[i] for i in range(total_chunks)]
        encoded_data = "".join(sorted_chunks)
        
        try:
            # Base64解码 -> zlib解压
            compressed_data = base64.b64decode(encoded_data)
            original_data = zlib.decompress(compressed_data)
            
            # 8. 校验文件
            new_file_hash = hashlib.md5(original_data).hexdigest()
            print(f"原始文件校验和: {file_hash}")
            print(f"重组文件校验和: {new_file_hash}")
            
            if new_file_hash == file_hash:
                output_filename = f"received_{filename}"
                with open(output_filename, 'wb') as f:
                    f.write(original_data)
                print(f"✅ 文件校验成功！已保存为: {output_filename}")
            else:
                print("❌ 文件校验失败！文件可能已损坏。")

        except Exception as e:
            print(f"❌ 文件重组或解码失败: {e}")
    else:
        print(f"\n❌ 文件接收不完整。预期接收 {total_chunks} 块，实际只成功解码 {len(received_chunks)} 块。")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # 如果未提供文件夹，则默认使用 `OUTPUT_DIR`
        default_dir = "qr_codes_output"
        print(f"未指定文件夹，将尝试从默认文件夹 '{default_dir}' 读取。")
        print(f"使用方法: python receiver_from_images.py <包含QR码图片的文件夹>")
        input_directory = default_dir
    else:
        input_directory = sys.argv[1]
    
    main(input_directory)