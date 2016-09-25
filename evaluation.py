from params import *

def get_category_and_id(image_id):
	s = image_id.find("/")
	return (image_id[:s], image_id[s+1:])

CH = 'Color Histogram'
DL = 'Deep Learning'
VW = 'Visual Keyword'
VC = 'Visual Concept'


# if __name__ == "__main__":
# 	print get_category_and_id('window/0096_128826595.jpg')