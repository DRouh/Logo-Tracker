from PIL import Image
from pylab import *
im = array(Image.open('C:/Users/dbubelnik\Desktop/test/test/Images for color clustering/pepsi.png'))
imshow(im)
print 'Please click 3 points'
x = ginput(3)
print 'you clicked:',x
show()