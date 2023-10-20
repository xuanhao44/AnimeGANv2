import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import onnxruntime as ort

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
    img = (img.squeeze() + 1.) / 2 * 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (wh[0], wh[1]))
    return img

def cvt2anime_video(video_filepath, output, model, onnx = 'model.onnx', output_format='mp4v'):  # 小写就不报错了，只是仍然无法在浏览器上播放

    # 调用 onnx：https://blog.csdn.net/songpeiying/article/details/133084413

    # check onnx model
    exists = os.path.isfile(onnx)
    if not exists:
        print('Model file not found:', onnx)
        return

    # 加载模型，若有 GPU, 则用 GPU 推理
    # https://zhuanlan.zhihu.com/p/645720587
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
    vid = cv2.VideoCapture(video_filepath)
    vid_name = os.path.basename(video_filepath)  # 只取文件名
    # https://blog.csdn.net/lsoxvxe/article/details/131999217
    # https://pythonjishu.com/python-os-28/
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*output_format)

    # 输出视频名称、路径
    video_out_name = vid_name.rsplit('.', 1)[0] + '_' + model + '.mp4'
    video_out_path = os.path.join(output, video_out_name)

    video_out = cv2.VideoWriter(video_out_path, codec, fps, (width, height))

    pbar = tqdm(total=total, ncols=80)
    pbar.set_description(f"Making: {video_out_name}")

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = np.expand_dims(process_image(frame),0)
        fake_img = session.run(None, {input_name : frame})
        fake_img = post_precess(fake_img[0], (width, height))
        video_out.write(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))
        pbar.update(1)

    pbar.close()
    vid.release()
    video_out.release()
    return os.path.join(output, video_out_path)

if __name__ == '__main__':
    # python onnx_video2anime.py --video video/input/お花見.mp4 --output video/output --model Shinkai --onnx pb_and_onnx_model/Shinkai_53.onnx
    arg = parse_args()
    check_folder(arg.output)
    info = cvt2anime_video(arg.video, arg.output, arg.model, arg.onnx)
    print(f'output video: {info}')
