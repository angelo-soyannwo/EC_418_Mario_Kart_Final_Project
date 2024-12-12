from PIL import Image
from glob import glob
from os import path

dataset_path = 'road_data'

data = []
for f in glob(path.join(dataset_path, '*.png')):
    i = Image.open(f)
    i.load()
    #f.split('/')[1][0] is a 0, 1, or 2 denoting sharp left, approximately straight, and sharp right respectively
    if(int(f.split('/')[1][0]) == 0 or int(f.split('/')[1][0]) == 1 or int(f.split('/')[1][0]) == 2):
        data.append( int(f.split('/')[1][0]) )
    else:
        print(f)

print('0s: ' + str(data.count(0)))
print('1s: ' + str(data.count(1)))
print('2s: ' + str(data.count(2)))
