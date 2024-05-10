
### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks and [HINet](https://github.com/megvii-model/HINet) 

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
git clone https://github.com/megvii-research/NAFNet
cd NAFNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Quick Start 
* Image Denoise Colab Demo: [<a href="https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing)
* Image Deblur Colab Demo: [<a href="https://colab.research.google.com/drive/1yR2ClVuMefisH12d_srXMhHnHwwA1YmU?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1yR2ClVuMefisH12d_srXMhHnHwwA1YmU?usp=sharing)
* Stereo Image Super-Resolution Colab Demo: [<a href="https://colab.research.google.com/drive/1PkLog2imf7jCOPKq1G32SOISz0eLLJaO?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1PkLog2imf7jCOPKq1G32SOISz0eLLJaO?usp=sharing)
* Single Image Inference Demo:
    * Image Denoise:
    ```
    python basicsr/demo.py -opt options/test/SIDD/NAFNet-width64.yml --input_path ./demo/noisy.png --output_path ./demo/denoise_img.png
  ```
    * Image Deblur:
    ```
    python basicsr/demo.py -opt options/test/REDS/NAFNet-width64.yml --input_path ./demo/blurry.jpg --output_path ./demo/deblur_img.png
    ```
    * ```--input_path```: the path of the degraded image
    * ```--output_path```: the path to save the predicted image
    * [pretrained models](https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models) should be downloaded. 
    * Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for single image restoration[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/chuxiaojie/NAFNet)
* Stereo Image Inference Demo:
    * Stereo Image Super-resolution:
    ```
    python basicsr/demo_ssr.py -opt options/test/NAFSSR/NAFSSR-L_4x.yml \
    --input_l_path ./demo/lr_img_l.png --input_r_path ./demo/lr_img_r.png \
    --output_l_path ./demo/sr_img_l.png --output_r_path ./demo/sr_img_r.png
    ```
    * ```--input_l_path```: the path of the degraded left image
    * ```--input_r_path```: the path of the degraded right image
    * ```--output_l_path```: the path to save the predicted left image
    * ```--output_r_path```: the path to save the predicted right image
    * [pretrained models](https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models) should be downloaded. 
    * Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for stereo image super-resolution[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/chuxiaojie/NAFSSR)
* Try the web demo with all three tasks here: [![Replicate](https://replicate.com/megvii-research/nafnet/badge)](https://replicate.com/megvii-research/nafnet)

### Results and Pre-trained Models

| name | Dataset|PSNR|SSIM| pretrained models | configs |
|:----|:----|:----|:----|:----|-----|
|NAFNet-GoPro-width32|GoPro|32.8705|0.9606|[gdrive](https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1AbgG0yoROHmrRQN7dgzDvQ?pwd=so6v)|[train](./options/train/GoPro/NAFNet-width32.yml) \| [test](./options/test/GoPro/NAFNet-width32.yml)|
|NAFNet-GoPro-width64|GoPro|33.7103|0.9668|[gdrive](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1g-E1x6En-PbYXm94JfI1vg?pwd=wnwh)|[train](./options/train/GoPro/NAFNet-width64.yml) \| [test](./options/test/GoPro/NAFNet-width64.yml)|
|NAFNet-SIDD-width32|SIDD|39.9672|0.9599|[gdrive](https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1Xses38SWl-7wuyuhaGNhaw?pwd=um97)|[train](./options/train/SIDD/NAFNet-width32.yml) \| [test](./options/test/SIDD/NAFNet-width32.yml)|
|NAFNet-SIDD-width64|SIDD|40.3045|0.9614|[gdrive](https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/198kYyVSrY_xZF0jGv9U0sQ?pwd=dton)|[train](./options/train/SIDD/NAFNet-width64.yml) \| [test](./options/test/SIDD/NAFNet-width64.yml)|
|NAFNet-REDS-width64|REDS|29.0903|0.8671|[gdrive](https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1vg89ccbpIxg3mK9IONBfGg?pwd=9fas)|[train](./options/train/REDS/NAFNet-width64.yml) \| [test](./options/test/REDS/NAFNet-width64.yml)|
|NAFSSR-L_4x|Flickr1024|24.17|0.7589|[gdrive](https://drive.google.com/file/d/1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1P8ioEuI1gwydA2Avr3nUvw?pwd=qs7a)|[train](./options/test/NAFSSR/NAFSSR-L_4x.yml) \| [test](./options/test/NAFSSR/NAFSSR-L_4x.yml)|
|NAFSSR-L_2x|Flickr1024|29.68|0.9221|[gdrive](https://drive.google.com/file/d/1SZ6bQVYTVS_AXedBEr-_mBCC-qGYHLmf/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1GS6YQSSECH8hAKhvzw6GyQ?pwd=2v3v)|[train](./options/test/NAFSSR/NAFSSR-L_2x.yml) \| [test](./options/test/NAFSSR/NAFSSR-L_2x.yml)|

### Image Restoration Tasks 

| Task                                 | Dataset | Train/Test Instructions            | Visualization Results                                        |
| :----------------------------------- | :------ | :---------------------- | :----------------------------------------------------------- |
| Image Deblurring                     | GoPro   | [link](./docs/GoPro.md) | [gdrive](https://drive.google.com/file/d/1S8u4TqQP6eHI81F9yoVR0be-DLh4cNgb/view?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/1yNYQhznChafsbcfHO44aHQ?pwd=96ii)|
| Image Denoising                      | SIDD    | [link](./docs/SIDD.md)  | [gdrive](https://drive.google.com/file/d/1rbBYD64bfvbHOrN3HByNg0vz6gHQq7Np/view?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/1wIubY6SeXRfZHpp6bAojqQ?pwd=hu4t)|
| Image Deblurring with JPEG artifacts | REDS    | [link](./docs/REDS.md)  | [gdrive](https://drive.google.com/file/d/1FwHWYPXdPtUkPqckpz-WBitpVyPuXFRi/view?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/17T30w5xAtBQQ2P3wawLiVA?pwd=put5) |
| Stereo Image Super-Resolution | Flickr1024+Middlebury    | [link](./docs/StereoSR.md)  | [gdrive](https://drive.google.com/drive/folders/1lTKe2TU7F-KcU-oaF8jqgoUwIMb6RW0w?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/1kov6ivrSFy1FuToCATbyrA?pwd=q263 ) |

