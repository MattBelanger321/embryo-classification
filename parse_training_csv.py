# This file should parse the "./data/train.csv" file
# and be able to return the information from each of the 3 segments
# in some standardized manner
# you should seralize the data into a boolean matrix segment[i,j]
# such that segment[i,j] = 0 if the pixel at coordinate (i,j) is not in the class
# and is 1 otherwise
# since there are 3 classes for each slice, we should have 3 matricies for each slice. Most of these matricies will be blank
# I recommened using opencv