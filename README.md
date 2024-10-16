# GI-Tract-Classification

This repo will use 3 segmentation approaches to classifiy this GI TRACT dataset

## Setup

you need to download your kaggle api token and copy it into ~/.kaggle/

then run 

```py
pip install kaggle
```

then you can run the download scripts provided

## Details

The reason why we are using this dataset, is because 

1. it doesnt blow up your hardrive LOL some of them are 100gb large
 
2. beyond being a classification problem, it requires that we use a segmenation approach. Segmentation uses localization which means that instead of classifying the entire image we classify each pixel, in this way we can create binary "masks" that are white for pixels that are in the class and black for pixels that arent in the class. Since there are 3 classes we will have 3 masks per "slice" or training image. This will allow us to use UNet