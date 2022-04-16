# AIC22 MTMC (Track 1) -- Team 10 solution

## Solution pipeline
* Use an one-stage detector (YOLOv5) and three ReID models for vehicle detection and feature extraction.
* Use motion--based [ByteTrack](https://github.com/ifzhang/ByteTrack) to against the similar appearence problem. Meanwhile, improve ByteTrack with some occlusion handling strategies.  
* Use `Anomaly Masked Association` for mutli-cam tracklets association. The anomaly tracklets are detected based on vehicle motional information, time consumption bewtween two adjacent cameras, and so on.  
* Post-processing

## Environment
Python 3.8 or later with dependencies listed in `requirements.txt`. Run the below command for install dependencies:

```
pip install -r requirements.txt
```

## Preparation
* Dataset  
    If you want to run the whole pipeline of our solution,please download the datasets from [AI City Challenge 2022](https://www.aicitychallenge.org/2022-data-and-evaluation/). When you download the dataset successfully, move the dataset to `./datasets` . Make sure the directory tree like the below structure:  
    ```
    -- ROOT
     | -- datasets
        | -- AIC22_Track1_MTMC_Tracking
           | -- test
           | -- ...
    ```

* Intermidiate Results  
    Meanwhile, we provide the intermidiate results for reproducing rapidly, which include `Detection Boxes` and `ReID features`, you can download them from [Google Drive](https://drive.google.com/file/d/13eHo1gwa8TzD2JHfd6vFPZxxY9TqnuhI/view?usp=sharing) or [Baidu Drive (提取码:4dhe)](https://pan.baidu.com/s/1jPfQr7lAd63N0y2dKBQHfw).  

    Unzip the downloaded file, move the unzipped directory to `./datasets`. Make sure the directory tree like the below structure:  

    ```
    -- ROOT
     | -- datasets
        | -- AIC22_Track1_MTMC_Tracking
           | -- test
           | -- ...
        | -- test_reproduce
           | -- detect_reid1
           | -- detect_reid2
           | -- detect_reid3
           | -- detect_result
           | -- feature_merge
    ```

* Models  
    In the solution, we use [YOLOv5](https://github.com/ultralytics/yolov5)  for vehicle detection, and use three [ReID models](https://github.com/LCFractal/AIC21-MTMC) provided from the AI City Challenge 2021 winner. You can download all models form [Google Drive](https://drive.google.com/drive/folders/1WVRH_4d0Gwad3_SaDNI8oZ2xqbBW1BUj?usp=sharing) or [Baidu Drive (提取码:pfs6)](https://pan.baidu.com/s/1RsqcH2jRR9GMpMWDbKGJyA).  
    
    Put the models to `./models`, and make sure the dicectory tree like the below structure:  
    ```
    -- ROOT
     | -- models
        | -- reid
           | -- resnext101_ibn_a_2.pth
           | -- resnet101_ibn_a_3.pth
           | -- resnet101_ibn_a_2.pth
        | -- detector
           | -- yolov5
              | -- yolov5x.pt
    ```

## Config
Open the configurations `aic_all.yml`, `aic_reidx.yml (x=1,2,3)` and `aic_vis_mcmt.yml`, then set  `MAIN_DIR` as the project root path.  
```
MAIN_DIR: ${PROJECT_ROOT}
```


## Run
* Run the complete pipeline  
  ```
  ./run_all.sh
  ```

* Rapid Reproducing  
    Make sure the intermidate results downloaded, and run 
    ```
    ./run_mtmc.sh
    ```