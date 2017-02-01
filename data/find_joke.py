from bs4 import BeautifulSoup
import urllib

# It reads all the joker from this website

base = "http://onelinefun.com/"
f = urllib.urlopen(base)
soup = BeautifulSoup(f.read(), "html.parser")
jokes = soup.find_all('p', class_='')[1:-5]

for i in range(1,270):
    print "at page {}".format(i)
    url = base + str(i) + "/"
    f = urllib.urlopen(url)
    soup = BeautifulSoup(f.read(), "html.parser")
    jokes += soup.find_all('p', class_='')[1:-5]


