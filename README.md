# Image-Stitching

Python CV2 SIFT Image-Stitching

This works for two or more images.

## Requirements

First, install all required python packages:
```
pip install -r requirements.txt
```
We install opencv-python == 3.4.2.17 to avoid the patent issue.

## Run Code
Then, you can run the python by using the following line:
```
python ./original.py directory-of-image 
```


or if you want to stitch the images in reverse order, use the following line
```
python ./original.py directory-of-images 0
```


Sample run:
```
python ./original.py ./img
```

## Result
![Image of Result](https://raw.githubusercontent.com/mattc95/Image-Stitching/master/output/5_result.jpg)
It will ouput a matching image and a result image.
![Image of Result](https://raw.githubusercontent.com/mattc95/Image-Stitching/master/output/0_matching.jpg)

![Image of Result](https://raw.githubusercontent.com/mattc95/Image-Stitching/master/output/0_result.jpg)
