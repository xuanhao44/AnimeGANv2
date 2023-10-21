import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import onnxruntime as ort
import pygame  # pip install pygame
from pygame import mixer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    desc = "Tensorflow implementation of AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--video', type=str, default='video/input/'+ '2.mp4',
                        help='video file or number for webcam')
    parser.add_argument('--output', type=str, default='video/output/' + 'Paprika',
                        help='output path')
    parser.add_argument('--model', type=str, default='Shinkai',
                        help='model name')
    parser.add_argument('--onnx', type=str, default='pb_and_onnx_model/Shinkai_53.onnx',
                        help='path of onnx')
    parser.add_argument('--output_format', type=str, default='mp4v',
                        help='codec used in VideoWriter when saving video to file')
    return parser.parse_args()


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def post_precess(img, wh):
    img = (img.squeeze() + 1.0) / 2 * 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (wh[0], wh[1]))
    return img

def cvt2anime_video(video_path, output, model, onnx = 'model.onnx', output_format='mp4v'):  # 小写就不报错了，只是仍然无法在浏览器上播放

    # check onnx model
    exists = os.path.isfile(onnx)
    if not exists:
        print('Model file not found:', onnx)
        return

    # 加载模型，若有 GPU, 则用 GPU 推理
    # 参考：https://zhuanlan.zhihu.com/p/645720587
    # 慎入！https://zhuanlan.zhihu.com/p/492040015
    if ort.get_device()=='GPU':
        print('use gpu')
        providers = ['CUDAExecutionProvider','CPUExecutionProvider',]
        session = ort.InferenceSession(onnx, providers=providers)
        session.set_providers(['CUDAExecutionProvider'], [ {'device_id': 0}]) #gpu 0
    else:
        print('use cpu')
        providers = ['CPUExecutionProvider',]
        session = ort.InferenceSession(onnx, providers=providers)

    input_name = session.get_inputs()[0].name

    # load video
    video_in = cv2.VideoCapture(video_path)
    video_in_name = os.path.basename(video_path)  # 只取文件名
    # https://blog.csdn.net/lsoxvxe/article/details/131999217
    # https://pythonjishu.com/python-os-28/

    total = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_in.get(cv2.CAP_PROP_FPS))
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*output_format)

    # 输出视频名称、路径
    video_out_name = video_in_name.rsplit('.', 1)[0] + '_' + model + '.mp4'
    video_out_path = os.path.join(output, video_out_name)

    video_out = cv2.VideoWriter("tmp.mp4", fourcc, fps, (width, height))

    pbar = tqdm(total=total, ncols=80)
    pbar.set_description(f"Making: {video_out_name}")

    while True:
        ret, frame = video_in.read()
        if not ret:
            break
        # frame = np.expand_dims(process_image(frame),0)
        frame = np.asarray(np.expand_dims(process_image(frame), 0))
        fake_img = session.run(None, {input_name : frame})
        fake_img = post_precess(fake_img[0], (width, height))
        video_out.write(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))
        pbar.update(1)

    pbar.close()
    video_in.release()
    video_out.release()

    # When your video is ready, just run the following command
    # You can actually just write the command below in your terminal

    # https://snipit.io/public/snippets/43806
    # os.system("ffmpeg -i Video.mp4 -vcodec libx264 Video2.mp4")
    # os.system("ffmpeg -i tmp.mp4 -vcodec libx264 " + video_out_path + " -y")

    # https://stackoverflow.com/questions/12938581/ffmpeg-mux-video-and-audio-from-another-video-mapping-issue
    # ffmpeg -an -i tmp.mp4 -vn -i video_path -c:a copy -c:v copy video_out_path
    # os.system("ffmpeg -an -i tmp.mp4 -vn -i " + video_path + " -c:a copy -c:v copy " + video_out_path + " -y")

    # 合成大西瓜！
    os.system("ffmpeg -an -i tmp.mp4 -vn -i " + video_path + " -c:a copy -c:v copy -vcodec libx264 " + video_out_path + " -y")

    return video_out_path

if __name__ == '__main__':
    # python onnx_video2anime.py --video video/input/お花見.mp4 --output video/output --model Shinkai --onnx pb_and_onnx_model/Shinkai_53.onnx 新海诚 (v2)

    # v3
    # python onnx_video2anime.py --video examples/2.mp4 --output output --model JP_face --onnx pb_and_onnx_model/AnimeGANv3_JP_face_v1.0.onnx 日漫脸
    # python onnx_video2anime.py --video examples/2.mp4 --output output --model PortraitSketch --onnx pb_and_onnx_model/AnimeGANv3_PortraitSketch_25.onnx 素描
    # 加密
    # python onnx_video2anime.py --video examples/2.mp4 --output output --model Haoyao --onnx pb_and_onnx_model/animeganv3_H64_model0.onnx 宫崎骏

    arg = parse_args()
    check_folder(arg.output)
    info = cvt2anime_video(arg.video, arg.output, arg.model, arg.onnx)
    print(f'output video: {info}')
