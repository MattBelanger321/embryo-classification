kaggle competitions download -c uw-madison-gi-tract-image-segmentation
mkdir .\data
Expand-Archive -Path "uw-madison-gi-tract-image-segmentation.zip" -DestinationPath ".\data"
del "uw-madison-gi-tract-image-segmentation.zip"
