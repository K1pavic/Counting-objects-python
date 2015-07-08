import numpy as np
import pylab # Dio od matplotlib.
import mahotas as mh
import cv2

image = cv2.imread('pictures/sky.png', 0) # Ucitavamo sliku u grayscale prikazu.

filtered = mh.gaussian_filter(image, 10) # Koristimo gauss filter za izostravanje slike.
result = filtered.astype('uint8') # Funckija mh.gaussian_filter() postavlja vrijednost varijable u float64.
rmax = mh.regmax(result) # Trazi maksimalne vrijednosti.

print rmax
pylab.imshow(mh.overlay(image, rmax))
pylab.show()

labeled, nr_objects = mh.label(rmax)
# mh.overlay() funckija postavlja img kao pozadinu a preko nje postavlja vrijednosti variable rmax u crvenom kanalu.

print ('Broj pronadenih objekata je {}.'.format(nr_objects))
pylab.imshow(labeled)
# pylab.gray()
pylab.show()

dist = mh.distance(result)
dist = dist.max() - dist
dist -= dist.min()
dist = dist/float(dist.ptp()) * 255
dist = dist.astype(np.uint8)
objects = mh.cwatershed(dist, labeled)
whole = mh.segmentation.gvoronoi(objects) # Voronoi segemtacija, svaki piksel poprima vrijednost najblizeg maksimuma.
pylab.imshow(objects)
pylab.show()

# print (labeled).shape
# print (labeled).max()
# print (labeled).min()

