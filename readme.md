
### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks 

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
git clone https://github.com/Zjtao-lab/DSSRNet.git
cd DSSRNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Quick Start 

* Stereo Image Inference Demo:
    ```
   export CUDA_VISIBLE_DEVICES="0,1"
    nohup python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=4321 basicsr/train.py \
    -opt options/train/NAFSSR/NAFSSR-S_x2.yml --launcher pytorch \
    > outprint_simpleffb_v0.4.txt 2>&1 &
    ```



### Results and Pre-trained Models

Will be uploaded later