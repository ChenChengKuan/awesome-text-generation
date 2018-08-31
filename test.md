# Introduction
This readme contains the usage of code and dataset for the body detection project (07.2018 - 08.2018) when I interned in the Deepfoce.Inc

# Dataset
* MPHB ([source link](http://parnec.nuaa.edu.cn/xtan/data/MPHB.html))
    * This dataset contains people (~1-5) with different poses and environments (both indoor and outdoor), which is suitable for building body deteciton modelfor simpler setting
    * Location: `/tank/body_data/MPHB`
    * Note: The dataset stored in the current folder has been pre-processed for yolo which contain following steps (Don't need to do it again):
        * Convert to yolo format: `python convert_label.py`
        * Generating train/val/test list: `python train_val_split.py`
* MPHB_neg (The best model is built upon this dataset)
    * This dataset is the combination of oirginal MPHB and negative samples (chairs, bottle) from VOC to lower the false positive rate of body detection
    * Location `/tank/body_data/MPHB_neg`
    * Note: To get the negative samples from VOC, please switch to `/tank/body_data/MyVOC/VOC_chair_bottle` then run (I already do it.) 
        ```       
        python collect_chair_bottle.py ../VOCdevkit/VOC2007/labels
        ```
* CrowdHuman ([source link](http://www.crowdhuman.org/))
    * This dataset contains mamy human in crowd setting, please refer to the original [paper links](http://www.crowdhuman.org/) for more details
    * Location `/tank/body_data/CrowdHuman`

# Training
All training related codes are in `/home/kuanchen/darknet_ryan`
## YOLO preparation
   * Feel free to skip following steps if you use the folder provied in the directory above.
   * Cloning the yolo projet: `git clone https://github.com/pjreddie/darknet`
   * Following the instruction in the [website](https://pjreddie.com/darknet/install/) to build the project
   * Note: I recommend to build it with OPENCV =1, if the original make file does not work for you, please replace line 35-37 by:
   
```
    ifeq ($(DEBUG), 1)

        OPTS=-O4 -g

    endif
```
## Train
  * Preparation
    * Generating anchors file (Feel free to skip if default setting is used)
      * Swith to `/tank/kuanchen/alex_darknet/darknet/scripts`
      * run `python gen_anchors.py -filelist xxx -output_dir xxx -num_clusters x`. For example, `python gen_anchors.py -filelist /tank/kuanchen/body_data/MPHB_neg/train.txt -output_dir ./ -num_clusters 5` will generate anchor file for MPHB_neg data with 5 cluster on current directory.
      * This step is only necessary if you chaneg the input size of (e.g, from 192 * 192 -> 418 * 418) of images.
      * The recommended anchors number for yolov2 and v3 are 5 and 9, however, it doesn't matter according to [this discussion](https://github.com/pjreddie/darknet/issues/597) 
    * Swith to `/home/kuanchen/darknet_ryan`
      * Set .data files (=body_mphb_neg.data, body_ch.data): you can just use default setting
      * Set .cfg files. Set `batch_size=64,`subdivision=8 or 16` for training. If memory error happens, use larger subdivision. These files are stored in:
        * `./cfg/MPHB_neg`: These folder stores some configs used for training yolo on MPHB_neg, the details can be found in result sheet
        * `./cfg/CrowdHuman`: This folder stores some configs used for training yolo on CrowdHuman dataset, the details can be found in the result sheet.
      * Set starting model: You can either choose to train from scratch by setting it to `./models/yolov2-tiny.con10` for yolov2-tiny and `./models/yolov3-tiny.conv13` or use the pre-trained weights stored in `/tank/kuanchen/alex_darknet/darknet/backup/yolov2-tiny_body_mphb_neg_v2/`, `/tank/kuanchen/alex_darknet/darknet/backup/yolov3-tiny_body_mphb_neg/`, `/tank/kuanchen/alex_darknet/darknet/backup/yolov2-tiny_body_ch_v2/` or `/tank/kuanchen/alex_darknet/darknet/backup/yolov3-tiny_body_ch/`
  * Training:
    Simply run 
    ```
    ./darknet detector train .data .cfg .model 
    ```
    For example, 
    ```
    ./darknet detector train ./cfg/body_mphb_neg.data ./cfg/MPHB_neg/yolov2-tiny-mphb_v2.cfg ./models/yolov2-tiny.conv10
    ```
    This will train yolov2-tiny model based on data specified in `./cfg/body_mphb_neg.data`, using configuration in `./cfg/MPHB_neg/yolov2-tiny-mphb_v2.cfg` and initialize the weight from `./models/yolov2-tiny.conv10`
  * Important note:
    * If change the input image size, you should re-compute the anchor **and** use the same height and width in the corresponding `.cfg` 
    * Make sure the backup folder in `.data` exist before training

# Inference
The inferece is defined as calculate MAP and get the bounding box location and print on the image, which are completed in the `/tank/kuanchen/alex_darknet` and `/tank/kuanchen/rafel_darknet` respectively.

## MAP calculation
  * Switch to `/tank/kuanchen/alex_darknet`
  * Change `batch_size=1` and `subdivision=1` in the `.cfg`file used in the training
  * Run
  ```
  ./darknet detector map .data .cfg .model
  ```
  For example,
  ```
  ./darknet detector map /home/kuanchen/darknet_ryan/cfg/body_mphb_neg.data /home/kuanchen/darknet_ryan/cfg/MPHB_neg/yolov2-tiny-mphb_v2.cfg ./backup/yolov2-tiny_mphb_neg_v2/ yolov2-tiny-mphb_v2_final.weights
  ```
  This will use the finla checkpoint to calculate MAP scores
## Get the bounding box result and position
  * Swith to `/tankkuanchen/rafel_darknet`
  * Make sure the result folder specified in `.data` file exist
  * Make sure change the name *valid* to *test* in `.data` file
  * Run
  ```
  ./darknet testimages .data .cfg .model -saveimg -savetxt
  ```
  For example,
  ```
  ./darknet testimages /home/kuanchen/darknet_ryan/cfg/body_mphb_neg.data /home/kuanchen/darknet_ryan/cfg/MPHB_neg/yolov2-tiny-mphb_v2.cfg ./backup/yolov2-tiny_mphb_neg_v2/ yolov2-tiny-mphb_v2_final.weights -saveimg -savetxt
  ```
  This will generate the images with bounding box with `.det` files that record the location of each bounding box in the result file specified in `.data`
  


# Results
The results can be found in this [sheet](Coming soon)

