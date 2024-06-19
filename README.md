# 使用U-Net模型进行内质网与线粒体的荧光显微图像分割

## 实验设置
实验平台：北京超级云计算中心N32EA14P分区 `NVIDIA A100-PCIE-40GB` <br>
依赖环境：
``` shell
module load anaconda/2021.11
module load cuda/11.8
conda create --name unet4bip python=3.11
source activate unet4bip

## For U-Net
# https://download.pytorch.org/whl/cu118/torch-2.3.0%2Bcu118-cp311-cp311-linux_x86_64.whl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm~=4.66.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install albumentations~=1.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install opencv-python~=4.9.0.80 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install matplotlib~=3.5.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install PyYAML~=6.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple/

conda env remove -n unet4bip
```

运行方法：
``` shell
# 修改script.py设置自己希望运行的实验
# 北京超级云计算中心N32EA14P分区
dsub -s run.sh # 提交作业
djob # 查看作业ID
djob -T job_id # 取消该作业
```