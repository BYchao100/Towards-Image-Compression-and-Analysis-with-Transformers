## Towards End-to-End Image Compression and Analysis with Transformers

Source code of our AAAI 2022 paper "Towards End-to-End Image Compression and Analysis with Transformers".

## Usage
The code is run with `Python 3.7`, `Pytorch 1.8.1`, `Timm 0.4.9` and `Compressai 1.1.4`.

### Data preparation
Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/vision/stable/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder), and the training and validation data is expected to be in the `train` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

### Pretrained model
The `./pretrained_model` provides the pretrained model without compression.
* Test

Please adjust `--data-path` and run `sh test.sh`:
```
python main.py --eval --resume ./pretrain_s/checkpoint.pth --model pretrained_model --data-path /path/to/imagenet/ --output_dir ./eval
```
The `./pretrain_s/checkpoint.pth` can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1RFXeKEzRn7mWk7ay0mQh_Q), with access code `aaai`.
* Train

Please adjust `--data-path` and run `sh train.sh`:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model pretrained_model --no-model-ema --clip-grad 1.0 --batch-size 128 --num_workers 16 --data-path /path/to/imagenet/ --output_dir ./ckp_pretrain
```

### Full model
The `./full_model` provides the full model with compression.
* Test

Please adjust `--data-path` and `--resume`, respectively. Run `sh test.sh`:
```
python main.py --eval --resume ./ckp_s_q1/checkpoint.pth --model full_model --no-pretrained --data-path /path/to/imagenet/ --output_dir ./eval
```
The `./ckp_s_q1/checkpoint.pth`, `./ckp_s_q2/checkpoint.pth` and `./ckp_s_q3/checkpoint.pth` can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1RFXeKEzRn7mWk7ay0mQh_Q), with access code `aaai`.

* Train

Please adjust `--data-path` and `--quality`, respectively.

| quality | alpha | beta | 
| :---: | :---: | :---: |
| 1 | 0.1 | 0.001 |
| 2 | 0.3 | 0.003 |
| 3 | 0.6 | 0.006 |

Run `sh train.sh`.
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model full_model --batch-size 128 --num_workers 16 --clip-grad 1.0 --quality 1 --data-path /path/to/imagenet/ --output_dir ./ckp_full
```

## Citation
```
@InProceedings{Bai2022AAAI,
  title={Towards End-to-End Image Compression and Analysis with Transformers},
  author={Bai, Yuanchao and Yang, Xu and Liu, Xianming and Jiang, Junjun and Wang, Yaowei and Ji, Xiangyang and Gao, Wen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
