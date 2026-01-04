import cv2
import numpy as np
import os
import time
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw, ImageFont

# ================= 配置部分 =================
# 解决 Matplotlib 中文乱码
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 模型路径
MODEL_PATH = 'runs/train/yolo11-640-50-64/weights/best.pt'
IMG_PATH = "test.jpg"
CONFIDENCE_THRESHOLD = 0.6  # 识别结果采纳的置信度阈值


def four_point_transform(image, pts):
    """透视变换函数 (保持不变)"""
    rect = np.array(pts, dtype="float32")
    width = int(np.linalg.norm(rect[1] - rect[0]))
    height = int(np.linalg.norm(rect[3] - rect[0]))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def cv2_add_chinese_text(img, text, position, text_color=(255, 0, 0), text_size=30):
    """
    在 OpenCV 图片上绘制中文
    """
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    # 尝试加载中文字体，如果失败则使用默认
    try:
        font = ImageFont.truetype("simhei.ttf", text_size, encoding="utf-8")
    except OSError:
        print("未找到 simhei.ttf，使用默认字体")
        font = ImageFont.load_default()

    draw.text(position, text, fill=text_color, font=font)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ================= 主逻辑 =================

def main():
    # 1. 初始化模型
    print("正在加载模型...")
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    model = YOLO(MODEL_PATH, task='detect')

    # 2. 读取并预处理图像
    if not os.path.exists(IMG_PATH):
        print(f"错误: 找不到图片 {IMG_PATH}")
        return

    original_image = cv2.imread(IMG_PATH)
    # 调整大小以确保与ONNX/YOLO模型输入尺寸一致
    process_img = cv2.resize(original_image, (640, 640))
    display_img = process_img.copy()  # 用于绘制结果

    # 3. 创建结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 获取当前脚本所在目录的根目录
    project_root = os.getcwd()
    result_folder = os.path.join(project_root, "results", f"picture_{timestamp}")
    os.makedirs(result_folder, exist_ok=True)
    print(f"结果将保存在: {result_folder}")

    # 4. 运行 YOLO 车牌检测
    print("正在进行检测...")
    # 设置 conf 和 iou
    results = model(process_img, conf=0.25, iou=0.45)

    boxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf.cpu().numpy()

    detected_plates_info = []  # 存储结果用于后续显示和保存文本

    if len(boxes) == 0:
        print("未检测到任何车牌。")
    else:
        # 5. 遍历检测结果
        for i, result in enumerate(boxes):
            x1, y1, x2, y2 = map(int, result[:4])
            confidence = confidences[i]

            # 增加 padding
            padding = 10
            x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
            x2, y2 = min(x2 + padding, process_img.shape[1] - 1), min(y2 + padding, process_img.shape[0] - 1)

            # 裁剪
            # crop_rect = process_img[y1:y2, x1:x2] # 简单裁剪

            # 透视变换准备
            pts = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype="float32")

            # 执行透视变换获取矫正后的车牌
            warped_plate = four_point_transform(process_img, pts)

            # 6. OCR 识别
            ocr_result = ocr.ocr(warped_plate, cls=True)

            plate_text = ""
            if ocr_result and isinstance(ocr_result, list):
                try:
                    plate_text = "".join([word[1][0] for line in ocr_result for word in line if word])
                except Exception as e:
                    print(f" 车牌 {i} OCR 解析异常: {e}")
                    plate_text = "识别失败"
            else:
                plate_text = "未检测到文本"

            print(f"检测框 {i}: {plate_text} (置信度: {confidence:.2f})")

            # 7. 过滤与保存逻辑
            # 只有当识别内容有效且置信度达标时才进行保存和绘制
            if plate_text not in ["识别失败", "未检测到文本"] and plate_text.strip() != "":
                if confidence >= CONFIDENCE_THRESHOLD:
                    # --- A. 绘制到大图上 ---
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # 使用 PIL 绘制中文标签
                    label = f"{plate_text} ({confidence:.2f})"
                    display_img = cv2_add_chinese_text(display_img, label, (x1, y1 - 30), (255, 0, 0))

                    # --- B. 保存单张车牌小图 ---
                    plate_filename = os.path.join(result_folder, f"plate_{i}_{plate_text}.jpg")
                    cv2.imwrite(plate_filename, warped_plate)

                    # --- C. 收集信息 ---
                    detected_plates_info.append({
                        "text": plate_text,
                        "conf": confidence,
                        "coords": (x1, y1, x2, y2),
                        "img": warped_plate  # 用于 Matplotlib 显示
                    })

    # 8. 保存最终结果大图和文本日志
    if detected_plates_info:
        # 保存完整效果图
        final_img_path = os.path.join(result_folder, "result_overview.jpg")
        cv2.imwrite(final_img_path, display_img)

        # 保存文本日志
        txt_path = os.path.join(result_folder, "results.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for info in detected_plates_info:
                f.write(f"车牌: {info['text']}, 置信度: {info['conf']:.4f}, 坐标: {info['coords']}\n")
        print(f"结果保存完成。共保存 {len(detected_plates_info)} 个有效车牌。")

    # 9. 使用 Matplotlib 显示结果
    # 如果检测到多个车牌，动态调整 subplot
    num_plates = len(detected_plates_info)
    if num_plates > 0:
        plt.figure(figsize=(10, 4 * num_plates))

        # 显示完整大图
        plt.subplot(num_plates + 1, 1, 1)
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.title(f"检测概览 (共 {num_plates} 个有效车牌)", color='red')
        plt.axis('off')

        # 显示每个车牌的详情
        for idx, info in enumerate(detected_plates_info):
            plt.subplot(num_plates + 1, 2, 2 * (idx + 1) + 1)  # 左侧放裁剪图
            plt.imshow(cv2.cvtColor(info['img'], cv2.COLOR_BGR2RGB))
            plt.title(f"车牌 {idx + 1} 矫正细节", color='blue')
            plt.axis('off')

            plt.subplot(num_plates + 1, 2, 2 * (idx + 1) + 2)  # 右侧放文字信息
            plt.text(0.1, 0.5, f"识别结果: {info['text']}\n置信度: {info['conf']:.2f}", fontsize=14)
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        # 如果没检测到，只显示原图
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB))
        plt.title("未检测到有效车牌", color='red')
        plt.show()


if __name__ == "__main__":
    main()