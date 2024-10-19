kaggle competitions download -c uw-madison-gi-tract-image-segmentation
mkdir ./data
unzip uw-madison-gi-tract-image-segmentation.zip -d ./data
rm uw-madison-gi-tract-image-segmentation.zip
python3 ./generate_segment_masks.py