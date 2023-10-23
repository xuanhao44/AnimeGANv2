# AnimeGANv2 部署、改进和展示

主要是学习 <https://github.com/TachibanaYoshino/AnimeGANv2>。学习过程见下面四个文档，是按照时间顺序来的。

## 1 [AnimeGANv2 部署测试](https://www.sheniao.top/tech/189.html)

（在 RTX 2080Ti 上，后续都是 RTX A4000）简单部署了 AnimeGANv2。并未做出其他调整。

环境：

```shell
conda create --prefix /cloud/animegan python=3.6 -y
conda activate /cloud/animegan
pip install --user tensorflow-gpu==1.15.0
pip install --user opencv-python==4.2.0.32
pip install --user tqdm
pip install --user numpy
pip install --user glob2
pip install --user argparse
pip install --user onnxruntime
conda install --prefix /cloud/animegan cudatoolkit==10.0.130 -y
conda install --prefix /cloud/animegan cudnn=7.6.0=cuda10.0_0 -y
```

自动化：

```shell
sudo -i -u featurize bash << EOF
cp -r ~/work/AnimeGANv2 ~/AnimeGANv2
EOF
```

## 2 [AnimeGANv2 + Gradio 轻量展示](https://www.sheniao.top/tech/191.html)

尝试使用 Gradio 来展示。在这篇文章中，并未解决输出视频编码的问题——但是之后解决了。

---

换成了 python 3.8，添加了一些包。

```shell
conda create --prefix /cloud/newanime python=3.8 -y
conda activate /cloud/newanime

pip install --user opencv-python==4.2.0.32
pip install --user tqdm
pip install --user numpy
pip install --user glob2
pip install --user argparse
pip install --user onnxruntime

pip install --user gradio
pip install --user socksio

conda install --prefix /cloud/newanime cudatoolkit==10.0.130 -y
conda install --prefix /cloud/newanime cudnn=7.6.0=cuda10.0_0 -y

pip install --user nvidia-pyindex
pip install --user nvidia-tensorboard==1.15
pip install --user nvidia-tensorflow
```

自动化：（这里已经接入了自己克隆的 AnimeGANv2 仓库）

```shell
cd ~/work/AnimeGANv2
git pull
cp -r ~/work/AnimeGANv2 ~/AnimeGANv2
```

## 3 [AnimeGANv2 onnx 模型调用尝试](https://www.sheniao.top/tech/194.html)

调用作者训练好的 onnx 模型，并尝试也调用 v3 中的模型，并成功。

---

卸载 onnxruntime，安装 onnxruntime-gpu。

```shell
conda create --prefix /cloud/newanime python=3.8 -y
conda activate /cloud/newanime

pip install --user opencv-python==4.2.0.32
pip install --user tqdm
pip install --user numpy
pip install --user glob2
pip install --user argparse
pip install --user onnxruntime-gpu

pip install --user gradio
pip install --user socksio

conda install --prefix /cloud/newanime cudatoolkit==10.0.130 -y
conda install --prefix /cloud/newanime cudnn=7.6.0=cuda10.0_0 -y

pip install --user nvidia-pyindex
pip install --user nvidia-tensorboard==1.15
pip install --user nvidia-tensorflow
```

自动化：（加密的 onnx 模型不在仓库里放出，请自行破解并放到相应目录。这里是把 onnx 模型放到云盘然后复制到相应的位置）

```shell
sudo -i -u featurize bash << EOF
cd ~/work/AnimeGANv2
git pull
cp -r ~/work/AnimeGANv2 ~/AnimeGANv2
cp ~/work/AnimeGANv3_H64_model0.onnx ~/AnimeGANv2/pb_and_onnx_model/AnimeGANv3_H64_model0.onnx
EOF
```

此外注意修改 onnx 模型名称：AnimeGANv3_H64_model0.onnx，不要小写。

## 4 [为视频转编码以及添加音频](https://www.sheniao.top/tech/197.html)

如标题，解决了第二篇文档中的问题，且附上了原视频音轨。

### 4.1 FFMPEG

需要安装 FFMPEG。

```shell
sudo apt update
sudo apt install ffmpeg -y
```

```shell
conda create --prefix /cloud/newanime python=3.8 -y
conda activate /cloud/newanime

pip install --user opencv-python==4.2.0.32
pip install --user tqdm
pip install --user numpy
pip install --user glob2
pip install --user argparse
pip install --user onnxruntime-gpu

pip install --user gradio
pip install --user socksio

conda install --prefix /cloud/newanime cudatoolkit==10.0.130 -y
conda install --prefix /cloud/newanime cudnn=7.6.0=cuda10.0_0 -y

pip install --user nvidia-pyindex
pip install --user nvidia-tensorboard==1.15
pip install --user nvidia-tensorflow
```

自动化：

```shell
sudo -i -u featurize bash << EOF
cd ~/work/AnimeGANv2
git pull
cp -r ~/work/AnimeGANv2 ~/AnimeGANv2
cp ~/work/AnimeGANv3_H64_model0.onnx ~/AnimeGANv2/pb_and_onnx_model/AnimeGANv3_H64_model0.onnx
EOF
```

### 4.2 PyAV

```shell
conda create --prefix /cloud/newanime python=3.8 -y
conda activate /cloud/newanime

pip install --user opencv-python==4.2.0.32
pip install --user tqdm
pip install --user numpy
pip install --user glob2
pip install --user argparse
pip install --user onnxruntime-gpu

pip install --user gradio
pip install --user socksio

conda install --prefix /cloud/newanime cudatoolkit==10.0.130 -y
conda install --prefix /cloud/newanime cudnn=7.6.0=cuda10.0_0 -y

pip install --user nvidia-pyindex
pip install --user nvidia-tensorboard==1.15
pip install --user nvidia-tensorflow

pip install --user av
```

自动化：

```shell
sudo -i -u featurize bash << EOF
cd ~/work/AnimeGANv2
git pull
cp -r ~/work/AnimeGANv2 ~/AnimeGANv2
cp ~/work/AnimeGANv3_H64_model0.onnx ~/AnimeGANv2/pb_and_onnx_model/AnimeGANv3_H64_model0.onnx
EOF
```

## 环境安装总和（如需要）

- 不适用于特定平台。
- RTX A4000

```shell
# 软件更新下载
sudo apt update
sudo apt install ffmpeg -y

git clone https://github.com/xuanhao44/AnimeGANv2.git
# 加密 onnx 文件自己放 & 记得改名字

# 环境创建
conda create -n animeneo python=3.8 -y
conda activate animeneo

pip install opencv-python==4.2.0.32
pip install tqdm
pip install numpy
pip install glob2
pip install argparse
pip install onnxruntime-gpu

pip install gradio
pip install socksio

conda install cudatoolkit==10.0.130 -y
conda install cudnn=7.6.0=cuda10.0_0 -y

pip install nvidia-pyindex
pip install nvidia-tensorboard==1.15
pip install nvidia-tensorflow

pip install av
```

## 文件说明

- [`app.py`](app.py) 第二篇文章中 Gradio 展示，仍使用 tf + 导入 ckpt的方式，没有编码和音轨的后续处理
- [`onnx_video2anime.py`](onnx_video2anime.py) 第三篇文章中调用 onnx 版本的 video2anime.py，还加上了第四篇文章 1 的编码和音轨的后续处理
- [`onnx_app.py`](onnx_app.py) 第三篇文章中调用 onnx 版本的 app.py，还加上了第四篇文章 1 的编码和音轨的后续处理
- [`so_video2anime.py`](so_video2anime.py) 第三篇文章中调用 so 文件的 video2anime.py，因为 so 本身原因只能使用 CPU
- [`onnx_video2anime_pyav.py`](onnx_video2anime_pyav.py) 第四篇文章中使用 PyAV 处理视频的 onnx_video2anime.py（即第四篇文章 2 的编码和音轨的后续处理）
- [`onnx_app_pyav.py`](onnx_app_pyav.py) 第四篇文章中使用 PyAV 处理视频的 onnx_app.py（即第四篇文章 2 的编码和音轨的后续处理）

- [`AnimeGANv3_src.so`](AnimeGANv3_src.so) 从 hugging face 的 v3 仓库中得到的动态链接文件 <https://huggingface.co/spaces/TachibanaYoshino/AnimeGANv3/tree/main>
- [`AnimeGANv3_bin.so`](AnimeGANv3_bin.so) 从 hugging face 的 v3 仓库中得到的动态链接文件 <https://huggingface.co/spaces/TachibanaYoshino/AnimeGANv3/tree/main>

- [`pb_and_onnx_model/AnimeGANv3_JP_face_v1.0.onnx`](pb_and_onnx_model/AnimeGANv3_JP_face_v1.0.onnx) 从 GitHub 的 v3 仓库中得到的 onnx 模型
- [`pb_and_onnx_model/AnimeGANv3_PortraitSketch_25.onnx`](pb_and_onnx_model/AnimeGANv3_PortraitSketch_25.onnx) 从 GitHub 的 v3 仓库中得到的 onnx 模型
- [`pb_and_onnx_model/AnimeGANv2_Hayao.onnx`](pb_and_onnx_model/AnimeGANv2_Hayao.onnx) 从 hugging face 的 v2 仓库中得到的 onnx 模型 <https://huggingface.co/vumichien/AnimeGANv2_Hayao/tree/main>
- [`pb_and_onnx_model/AnimeGANv2_Paprika.onnx`](pb_and_onnx_model/AnimeGANv2_Paprika.onnx) 从 hugging face 的 v2 仓库中得到的 onnx 模型 <https://huggingface.co/vumichien/AnimeGANv2_Paprika/tree/main>

- [`examples/`](examples) 文件夹，里面存放用于样例输入，也是各类 app.py 读取样例的路径

