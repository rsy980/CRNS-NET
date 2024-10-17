# CRNS-Net
The codes for the work "CRNS: CLIP-Driven Referring Nuclei Segmentation". 
Thanks to TransNeXt for publishing the excellent code。
## 1. Download pre-trained TransNeXt model (TransNeXt-Base)
* [Get pre-trained model in this link] (https://github.com/DaiShiResearch/TransNeXt): Put pretrained TransNeXt-Base into folder "pretrained_ckpt/"

## 2. Prepare data

- The two public datasets we used, CPM-17 and MoNuSeg, are available from the links below.  (https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK) and (https://monuseg.grand-challenge.org/Data/). 

- For the method of retrieving the text branch, please refer to utils/getText.

## 3. Environment

- Please prepare an environment with python=3.9, cuda=11.8, numpy=1.22, and install the remaining packages as needed.

- For TransNext environment dependencies, based on this link (https://github.com/DaiShiResearch/TransNeXt).

## 4. Train/Test

- The batch size we used is 8.  Our test found that batch_size has little effect on the final result. You can modify the value of batch_size according to the size of GPU memory.


## References
* [TransNeXt](https://github.com/DaiShiResearch/TransNeXt)
* [CLIP-Driven-Universal-Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model)



