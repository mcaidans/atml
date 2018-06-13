import numpy
import scipy.misc as smp
from PIL import Image

file1 = "C:/Users/Aidan/Documents/Eager/Data/201805250700/PerthB01_201805250700_data.csv"
file2 = "C:/Users/Aidan/Documents/Eager/Data/201805250700/PerthB02_201805250700_data.csv"
file3 = "C:/Users/Aidan/Documents/Eager/Data/201805250700/PerthB03_201805250700_data.csv"


data1 = numpy.genfromtxt(file1, delimiter=',')
data2 = numpy.genfromtxt(file2, delimiter=',')
data3 = numpy.genfromtxt(file3, delimiter=',')

#data1 = data1.astype(int)
#data2 = data2.astype(int)
#data3 = data3.astype(int)

print(data1)

rgb = numpy.dstack((data3, data1, data2))
print(rgb.shape)
#img = Image.fromarray(data1)

img = smp.toimage(data1)#_uint8)       # Create a PIL image
img.save("b.jpg")
img = smp.toimage(data2)#_uint8)       # Create a PIL image
img.save("g.jpg")
img = smp.toimage(data3)#_uint8)       # Create a PIL image
img.save("r.jpg")
