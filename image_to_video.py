import cv2
import os

def images_to_video(image_folder, output_file, fps):
    # 获取图片路径
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]
    image_files.sort()  # 确保图片按顺序读取
    # print(image_files)
    # 读取第一张图片以获取宽度和高度
    first_image_path = image_folder+'/'+ image_files[0]
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 遍历所有图片并写入视频
    for image in image_files:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 释放资源
    video_writer.release()

# 设置参数
image_folder = 'output/Yingrenshi_add_path/renderability/view_render/travesal/202408051135'  # 图片所在的文件夹路径
output_file = 'output/Yingrenshi_add_path/renderability/view_render/travesal/video11.mp4'  # 输出视频文件名
fps = 30  # 每秒帧数

# 调用函数
images_to_video(image_folder, output_file, fps)