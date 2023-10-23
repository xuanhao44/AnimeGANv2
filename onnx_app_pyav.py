import os

import av
import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def process_image_alter(img, x32=True):
    h, w = img.shape[:2]
    if x32:  # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = img.astype(np.float32) / 127.5 - 1.0  # 注意修改
    return img


def post_precess(img, wh):
    img = (img.squeeze() + 1.0) / 2 * 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (wh[0], wh[1]))
    return img


def cvt2anime_video(video_path, output, model, onnx='model.onnx'):
    # check onnx model
    exists = os.path.isfile(onnx)
    if not exists:
        print('Model file not found:', onnx)
        return

    # 加载模型，若有 GPU, 则用 GPU 推理
    # 参考：https://zhuanlan.zhihu.com/p/645720587
    # 慎入！https://zhuanlan.zhihu.com/p/492040015
    if ort.get_device() == 'GPU':
        print('use gpu')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider', ]
        session = ort.InferenceSession(onnx, providers=providers)
    else:
        print('use cpu')
        providers = ['CPUExecutionProvider', ]
        session = ort.InferenceSession(onnx, providers=providers)

    video_in_name = os.path.basename(video_path)  # 只取文件名
    # 输出视频名称、路径
    video_out_name = video_in_name.rsplit('.', 1)[0] + '_' + model + '.mp4'
    video_out_path = os.path.join(output, video_out_name)

    # 载入视频
    in_container = av.open(video_path, 'r')
    in_video_stream = next(s for s in in_container.streams if s.type == 'video')
    in_audio_stream = next(s for s in in_container.streams if s.type == 'audio')

    fps = in_video_stream.base_rate  # 帧率
    width = in_video_stream.width  # 帧宽
    height = in_video_stream.height  # 帧高
    total_time_in_second = in_video_stream.duration * 1.0 * in_video_stream.time_base  # 视频总长
    total_frame = int(total_time_in_second * fps)  # 视频总帧数

    out_container = av.open(video_out_path, 'w')
    out_video_stream = out_container.add_stream("h264", rate=fps)
    out_audio_stream = out_container.add_stream(template=in_audio_stream)

    out_video_stream.width = width
    out_video_stream.height = height

    pbar = tqdm(total=total_frame, ncols=80)
    pbar.set_description(f"Making: {video_out_name}")

    for packet in in_container.demux(in_video_stream, in_audio_stream):

        _type = packet.stream.type

        for frame in packet.decode():
            if _type == 'video':
                frame = frame.to_ndarray(format="rgb24")  # 这里 frame 得到了 rgb 格式
                # https://www.zhihu.com/question/452884533 VideoCapture 读出来的图片默认是 BGR 格式，所以需要转
                # 但是这里 frame 可以指定格式，所以后面就修改原来的 process_image 函数，不用转换 cvtColor 了。

                frame = np.asarray(np.expand_dims(process_image_alter(frame), 0))
                fake_img = session.run(None, {session.get_inputs()[0].name: frame})
                fake_img = post_precess(fake_img[0], (width, height))

                frame = av.VideoFrame.from_ndarray(fake_img, format="rgb24")  # 接收 rgb
                out_container.mux(out_video_stream.encode(frame))

                pbar.update(1)  # bar 跟随 video frame

            elif _type == 'audio':
                # We need to skip the "flushing" packets that `demux` generates.
                if packet.dts is None:
                    continue
                # We need to assign the packet to the new stream.
                packet.stream = out_audio_stream
                out_container.mux(packet)

    pbar.close()

    # Close the file
    out_container.close()

    return video_out_path


def anime(video_filepath, style):
    try:
        output = "output"
        check_folder(output)  # 空文件夹真烦人

        model = "H64_model0"
        if style == "《起风了》（宫崎骏）":
            model = "H64_model0"
        elif style == "素描":
            model = "PortraitSketch_25"
        elif style == "日漫脸":
            model = "JP_face_v1.0"

        onnx = os.path.join("pb_and_onnx_model", "AnimeGANv3_" + model + ".onnx")

        try:
            output_filepath = cvt2anime_video(video_filepath, output, model, onnx)
            return output_filepath
        except RuntimeError as error:
            print('Error', error)
    except Exception as error:
        print('global exception', error)
        return None, None


title = "记录式影片迁移动漫创作平台"
description = r"""目前，很多纪录式电影片段具有很深的教育意义，然而这些影片往往趣味性较低，对于青少年受众群体尤甚；而动漫风格多变，色彩丰富，主角形体、语音都具有非常大的丰富性。<br>
动漫是我们日常生活中常见的艺术形式，被广泛应用于广告、电影和儿童教育等多个领域。目前，动漫的制作主要是依靠手工实现。<br>
然而，手工制作动漫非常费力，需要非常专业的艺术技巧。对于动漫艺术家来说，创作高质量的动漫作品需要仔细考虑线条、纹理、颜色和阴影，这意味着创作动漫既困难又耗时。<br>
因此，能够将真实世界的照片、视频自动转换为高质量动漫风格图像的自动技术是非常有价值的。它不仅能让艺术家们更多专注于创造性的工作，也能让普通人更容易创建自己的动漫作品。😊<br>
"""

demo = gr.Interface(
    fn=anime,
    inputs=[
        gr.Video(source="upload"),
        gr.Dropdown([
            '《起风了》（宫崎骏）',  # H64_model0
            '素描',  # PortraitSketch_25
            '日漫脸',  # JP_face_v1.0
        ],
            type="value",  # 默认
            value='《起风了》（宫崎骏）',
            label='style'),
    ],
    outputs=[
        gr.PlayableVideo(),
    ],
    title=title,
    description=description,
    allow_flagging='never',
    examples=[
        ["examples/1.mp4", "素描"],
        ["examples/2.mp4", "《起风了》（宫崎骏）"],
        ["examples/3.mp4", "日漫脸"],
        ["examples/4.mp4", "《起风了》（宫崎骏）"],
        ["examples/5.mp4", "素描"],
    ],
    cache_examples=True,  # 缓存示例以实现快速运行，如修改需要手动删除
)

if __name__ == "__main__":
    # https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance
    # queue 方法允许用户通过创建一个队列来控制请求的处理速率，从而实现更好的控制。用户可以设置一次处理的请求数量，并向用户显示他们在队列中的位置。
    demo.launch(share=True, show_error=True)
