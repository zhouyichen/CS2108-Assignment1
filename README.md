# CS2108-Assignment1

Put all files in the ImageSeach_demo folder into the root folder of the repo, images and other files will be ignored.
Clear the original files in the 'dataset' folder, and put all files in the 'ImageData' folder into the 'dataset' folder of the repo.

Run the following in terminal to extract features of the training and testing dataset:

```
python color_hist.py
python deep_learning_features.py
python sift.py
python sift_clustering.py 300
python bag_of_words.py 300
```

Run the UI by: `python UI_search.py`