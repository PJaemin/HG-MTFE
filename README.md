# HG-MTFE

### Jaemin Park, An Gia Vien, Thuy Thi Pham, Hanul Kim, and Chul Lee
Official pytorch implementation for **"Image Enhancement via Cross-Attention-Based Multiple Transformation Function Estimation"**

This paper will be appeared in **IEEE Transactions on Consumer Electronics**.


<p float="left">
  &emsp;&emsp; <img src="overview.PNG" width="800" />
</p>

## Preparation
### Training data: [Download from GoogleDrive](https://drive.google.com/file/d/1jekxUtXmcU79DfnyTMbLEUm9y6vQwuVU/view?usp=sharing)
The ZIP file contains three test datasets:
- LOL dataset: 485 image pairs
- FiveK dataset: 4,500 image pairs
- EUVP dataset: 11,435 image pairs

### Testing samples: [Download from GoogleDrive](https://drive.google.com/file/d/1bnmfDTkcK-Sq2KGIWnv9QmEZUWyHg4x5/view?usp=sharing)
The ZIP file contains three test datasets:
- LOL dataset: 15 image pairs
- FiveK dataset: 500 image pairs
- EUVP dataset: 515 image pairs

### Pretrained weights: [Download from GoogleDrive](https://drive.google.com/file/d/1-GIGT_V3HjTYBCGvON8kjSWCFxWXWM-8/view?usp=sharing)
The ZIP file contains weight files trained with each training dataset.

## Training
1. Put low-quality images of training dataset in ./data/train_data/input
2. Put high-quality images of training dataset in ./data/train_data/gt
3. Put test images in ./data/test_data/LOL
4. Put ground-truths of test images in ./data/test_gt
5. Run below commend:
```
python lowlight_train.py
```
6. The trained model is saved at ./models
7. The result images are saved at ./data/analysis

## Testing
1. Put test images in ./data/test_data/LOL
2. Put ground-truths of test images in ./data/test_gt
3. Run below commend:
```
python lowlight_test.py
```
4. The result images are saved at ./data/analysis

## Citation (To be updated)
If you find this work useful for your research, please consider citing our paper:
```
@article{Park20224,
    author={{Park, Jaemin and Vien, An Gia and Pham, Thuy Thi and Kim, Hanul and Lee, Chul}},
    booktitle={},
    title={Image Enhancement via Cross-Attention-Based Multiple Transformation Function Estimation}, 
    year={2024},
    volume={},
    number={},
    pages={},
    doi={}}
}
```

## License
See [MIT License](https://github.com/PJaemin/MTFE/blob/main/LICENSE)


