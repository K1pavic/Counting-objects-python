import numpy as np
import pylab # Dio od matplotlib.
import mahotas as mh
import cv2


image = cv2.imread('pictures/sky.png', 0) # Ucitavamo sliku u grayscale prikazu.

pylab.imshow(image) # Ucitavanje prvotne slike.
pylab.gray() # Prebacujemo iz heatmap prikaza u obicni grayscale.
pylab.show() # Po defaultu prikazuje heatmapu slike.

# filtered = mh.gaussian_filter(image, 0.125)
# change = filtered.astype('uint8')

# Ako postoje mali komadici pored objekata koji se trebaju brojati ili slika nije dobre kvalitete
# treba koristit gauss filter, ali ako to nije potrebno bolje ga je ne koristiti u ovome primjeru,
# jer se objekti koji su blizu jedan drugom mogu stopiti.

labeled, nr_objects = mh.label(image) # Koristimo funckiju mh.label za oznacivanje i brojanje objekata.
print ('Broj pronadenih objekata je {}.'.format(nr_objects)) # Ispisujemo broj pronadenih objekata.
pylab.imshow(labeled)
pylab.gray()  # Ako smo koristili pylab.grayscale() vraca prikaz slike u heatmapu.
pylab.show() # Prikaz oznacene slike.

# labeled ima istu velicinu kao slika koja je dana mh.label funkciji s vrijednostima pozadine (0) do vrijednosti
# broja pronadenih objekata.

# print (labeled).shape
# print (labeled).max()
# print (labeled).min()
