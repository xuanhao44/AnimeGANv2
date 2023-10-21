import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import AnimeGANv3_src

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    desc = "Tensorflow implementation of AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--video', type=str, default='video/input/'+ '2.mp4',
                        help='video file or number for webcam')
    parser.add_argument('--output', type=str, default='video/output/' + 'Paprika',
                        help='output path')
    parser.add_argument('--style', type=str, default='Shinkai',
                        help='style name')
    parser.add_argument('--output_format', type=str, default='mp4v',
                        help='codec used in VideoWriter when saving video to file')
    return parser.parse_args()


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def cvt2anime_video(video_path, output, style, output_format='mp4v'):  # 小写就不报错了，只是仍然无法在浏览器上播放
    print(video_path, style)

    # load video
    video_in = cv2.VideoCapture(video_path)
    video_in_name = os.path.basename(video_path)  # 只取文件名

    total = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_in.get(cv2.CAP_PROP_FPS))
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*output_format)

    # 输出视频名称、路径
    video_out_name = video_in_name.rsplit('.', 1)[0] + '_' + style + '.mp4'
    video_out_path = os.path.join(output, video_out_name)

    video_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    pbar = tqdm(total=total, ncols=80)
    pbar.set_description(f"Making: {video_out_name}")

    while True:
        ret, frame = video_in.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0

        if style == "AnimeGANv3_Arcane":
            f = "A"
        elif style == "AnimeGANv3_Trump v1.0":
            f = "T"
        elif style == "AnimeGANv3_Shinkai":
            f = "S"
        elif style == "AnimeGANv3_PortraitSketch":
            f = "P"
        elif style == "AnimeGANv3_Hayao":
            f = "H"
        elif style == "AnimeGANv3_Disney v1.0":
            f = "D"
        elif style == "AnimeGANv3_JP_face v1.0":
            f = "J"
        else:
            f = "U"

        output = AnimeGANv3_src.Convert(frame, f, False)

        video_out.write(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        pbar.update(1)

    pbar.close()
    video_in.release()
    video_out.release()

    return video_out_path

if __name__ == '__main__':

    # v3

    # python so_video2anime.py --video examples/2.mp4 --output output --style AnimeGANv3_Arcane
    # python so_video2anime.py --video examples/2.mp4 --output output --style AnimeGANv3_Trump v1.0
    # python so_video2anime.py --video examples/2.mp4 --output output --style AnimeGANv3_Shinkai
    # python so_video2anime.py --video examples/2.mp4 --output output --style AnimeGANv3_PortraitSketch
    # python so_video2anime.py --video examples/2.mp4 --output output --style AnimeGANv3_Hayao
    # python so_video2anime.py --video examples/2.mp4 --output output --style AnimeGANv3_Disney v1.0
    # python so_video2anime.py --video examples/2.mp4 --output output --style AnimeGANv3_JP_face v1.0

    arg = parse_args()
    check_folder(arg.output)
    info = cvt2anime_video(arg.video, arg.output, arg.style)
    print(f'output video: {info}')
