from bs4 import BeautifulSoup
import urllib2

# It reads all the joker from this website

base = "http://www.jokes4us.com/pickuplines/random/pickupline"
pickup = []

for i in range(1,1328):
    print "at page {}".format(i)
    url = base + str(i) + ".html"
    f = urllib2.urlopen(url, timeout=1000)
    soup = BeautifulSoup(f.read(), "html.parser")
    pickup += soup.find_all('p', align='center')[0]

import pickle
pickle.dump(pickup, open("pickups.dat","wb"))
