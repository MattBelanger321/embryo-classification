import parse_training_csv as parser

# This application uses the function(s) exposed by parse_training_csv to
# generate images of pixels that will highlight the segment that they represent. This
# will give us a visual represenetation of what each class actually represents
# using the boolean matrix, you should basically be able to multiply the boolean matrix by 255
# and use imwrite() from open cv to create png image
# take care to write the image in the ./generated_images directory, making sure to miminc the file structure
# seen in .data