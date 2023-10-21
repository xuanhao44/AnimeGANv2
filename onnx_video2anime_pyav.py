import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import onnxruntime as ort
import av

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

def cvt2anime_video(video_path, output, model, onnx = 'model.onnx'):

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
        session.set_providers(['CUDAExecutionProvider'], [ {'device_id': 0}])  # gpu 0
    else:
        print('use cpu')
        providers = ['CPUExecutionProvider',]
        session = ort.InferenceSession(onnx, providers=providers)

    input_name = session.get_inputs()[0].name

    # 载入视频
    in_container = av.open(video_path)
    video_in_name = os.path.basename(video_path)  # 只取文件名
    # https://blog.csdn.net/lsoxvxe/article/details/131999217
    # https://pythonjishu.com/python-os-28/

    in_stream = in_container.streams.video[0]
    fps = in_stream.base_rate  # 帧率
    frame_width = in_stream.width  # 帧宽
    frame_height = in_stream.height  # 帧高
    total_time_in_second = in_stream.duration * 1.0 * in_stream.time_base  # 视频总长
    total_frame = int(total_time_in_second * fps)  # 视频总帧数

    # 输出视频名称、路径
    video_out_name = video_in_name.rsplit('.', 1)[0] + '_' + model + '.mp4'
    video_out_path = os.path.join(output, video_out_name)

    out_container = av.open(video_out_path, mode="w")
    out_stream = out_container.add_stream("h264", rate=fps)
    out_stream.width = frame_width
    out_stream.height = frame_height
    out_stream.pix_fmt = "yuv420p"

    pbar = tqdm(total=total_frame, ncols=80)
    pbar.set_description(f"Making: {video_out_name}")

    for frame in in_container.decode(video=0):
        frame = frame.to_ndarray(format="rgb24")

        frame = np.asarray(np.expand_dims(process_image(frame), 0))
        fake_img = session.run(None, {input_name: frame})
        fake_img = post_precess(fake_img[0], (frame_width, frame_height))

        frame = av.VideoFrame.from_ndarray(fake_img, format="rgb24")
        for packet in out_stream.encode(frame):
            out_container.mux(packet)

        pbar.update(1)

    pbar.close()

    # Flush stream
    for packet in out_stream.encode():
        out_container.mux(packet)

    # Close the file
    in_container.close()
    out_container.close()

    return video_out_path

if __name__ == '__main__':
    # python onnx_video2anime.py --video video/input/お花見.mp4 --output video/output --model Shinkai_53 --onnx pb_and_onnx_model/Shinkai_53.onnx 新海诚 (v2)

    # v3
    # python onnx_video2anime.py --video examples/2.mp4 --output output --model JP_face_v1.0 --onnx pb_and_onnx_model/AnimeGANv3_JP_face_v1.0.onnx 日漫脸
    # python onnx_video2anime.py --video examples/2.mp4 --output output --model PortraitSketch_25 --onnx pb_and_onnx_model/AnimeGANv3_PortraitSketch_25.onnx 素描
    # 加密
    # python onnx_video2anime.py --video examples/2.mp4 --output output --model H64_model0 --onnx pb_and_onnx_model/Animeganv3_H64_model0.onnx 宫崎骏

    arg = parse_args()
    check_folder(arg.output)
    info = cvt2anime_video(arg.video, arg.output, arg.model, arg.onnx)
    print(f'output video: {info}')
