from PIL import Image
import numpy
file1 = "C:/Users/Aidan/Documents/Eager/Data/201805250700/PerthB01_201805250700_data.csv"
file2 = "C:/Users/Aidan/Documents/Eager/Data/201805250700/PerthB02_201805250700_data.csv"
file3 = "C:/Users/Aidan/Documents/Eager/Data/201805250700/PerthB03_201805250700_data.csv"
b = numpy.genfromtxt(file1, delimiter=',')
g = numpy.genfromtxt(file2, delimiter=',')
r = numpy.genfromtxt(file3, delimiter=',')

b = b.astype(int)
g = g.astype(int)
r = r.astype(int)

rgbArray = numpy.zeros((444,376), 'uint8')
#rgbArray[..., 0] = r
#rgbArray[..., 1] = g
rgbArray = b
img = Image.fromarray(rgbArray)
img.show()
