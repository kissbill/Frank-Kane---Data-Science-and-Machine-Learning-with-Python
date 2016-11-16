#KNN (K-Nearest-Neighbors)

#KNN is a simple concept: define some distance metric between the items in your dataset,
#and find the K closest items. You can then use those items to predict some property of a test item,
# by having them somehow "vote" on it.
#As an example, let's look at the MovieLens data.
# We'll try to guess the rating of a movie by looking at 
 #the 10 movies that are closest to it in terms of genres and popularity.
#To start, we'll load up every rating in the data set into a Pandas DataFrame:

import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('e:/sundog-consult/udemy/datascience/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3))
ratings.head()

#	user_id	movie_id	rating
#0	0	50	5
#1	0	172	5
#2	0	133	1
#3	196	242	3
#4	186	302	3

#Now, we'll group everything by movie ID, and compute the total number of 
#ratings (each movie's popularity) and the average rating for every movie:

import numpy as np

movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
movieProperties.head()


#	rating
#size	mean
#movie_id		
#1	452	3.878319
#2	131	3.206107
#3	90	3.033333
#4	209	3.550239
#5	86	3.302326

#The raw number of ratings isn't very useful for computing distances between movies, 
#so we'll create a new DataFrame that contains the normalized number of ratings. 
#So, a value of 0 means nobody rated it, and a value of 1 will mean it's the most popular movie there is.
#minimum es a maximum kozott fogjuk nezni es rate-elni az ossze film kozott -> lambda van erre 

movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
#az egesz dataframe-re lehet -> numpy[min and max] maximum es a minimum rate-et megtalalja az egesz data set -re 
#szoval megtalalja a legnepszerubb es legrosszabbat es range-et ad  es normalizal mindent ezen range alapjan  
movieNormalizedNumRatings.head()
#szoval 0-1 skalan megadja egy adott film nepszeruseget 
#0 senki , 1 mindenki megnezte


#Now, let's get the genre information from the u.item file. 
#The way this works is there are 19 fields, 
#each corresponding to a specific genre - a value of '0' means it is not in that genre, 
#and '1' means it is in that genre.
#A movie may have more than one genre associated with it.
#While we're at it, we'll put together everything into one big Python dictionary called movieDict.
#Each entry will contain the movie name, list of genre values, the normalized popularity score, 
#and the average rating for each movie:
#beolvassuk soronkent ezt a file-t

movieDict = {}
with open(r'e:/sundog-consult/udemy/datascience/ml-100k/u.item') as f:
    temp = ''
    for line in f:
    	#vegig iteralunk a file-on soronkent,
        fields = line.rstrip('\n').split('|')#levagjuk az ujsort es felvagjuk a py dilimeter alapjan
        movieID = int(fields[0]) # es kiszedjuk a move id-t stb stb
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        #dictionarit, ami a film ID koti a nevehez genrehez , popularity , avarege rating 
        movieDict[movieID] = (name, genres, movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))
#pelda sor
#5|Copycat (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Copycat%20(1995)|0|0|0|0|0|0|1|0|1|0|0|0|0|0|0|0|1|0|0
																					#genre-> ahol 0 ott nem scif pl


# ezen szamitasok alapjan, hogy szerezzuk meg a legkozelebbi szomszedot adott filmnek
#pl toystory

#Now let's define a function that computes the "distance" between two movies based on how similar their genres are, 
#and how similar their popularity is.
# Just to make sure it works, we'll compute the distance between movie ID's 2 and 4:

from scipy import spatial

def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)#cosine similarity matrixot hasznal a ket vektro []<-osszehasonlitasara->[] 
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)# a raw data erteket vesszuk nyers ertekbe
    return genreDistance + popularityDistance
  
#ez olyan mint egy ket tengelyes abra 1 tengely mufaj hasonlosag cosine matrix alapjan,
# popularity -> es a tavolsagot nezzuk a ket film kozott ennel
ComputeDistance(movieDict[2], movieDict[4])


#Now, we just need a little code to compute the distance between some given test movie 
#(Toy Story, in this example) and all of the movies in our data set. 
#When the sort those by distance, and print out the K nearest neighbors:

import operator

def getNeighbors(movieID, K): #ami erdekel movie es k szomszed 
    distances = []
    for movie in movieDict: #minden filmen vegig iteral 
        if (movie != movieID):#ha kulonboz film mint amit vizsgalunk
        	#kiszamolja a korabbi fuggvennyel a kulonbseget 
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist)) # hozzafuzi a listahoz az eredmenynek 
    distances.sort(key=operator.itemgetter(1)) #rendezzuk a result-0t
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors

K = 10
avgRating = 0
neighbors = getNeighbors(1, K) #film id es 10 szomszedja
for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]
    print movieDict[neighbor][0] + " " + str(movieDict[neighbor][3])
    
avgRating /= float(K)


#Star Trek III: The Search for Spock (1984) 3.11111111111
#Stargate (1994) 3.14173228346
#Star Trek IV: The Voyage Home (1986) 3.4472361809
#Star Trek: The Motion Picture (1979) 3.03418803419
#Star Trek: Generations (1994) 3.33620689655
#Lost World: Jurassic Park, The (1997) 2.94303797468
#Star Trek: The Wrath of Khan (1982) 3.81557377049
#Abyss, The (1989) 3.58940397351
#Spawn (1997) 2.61538461538
#Star Trek V: The Final Frontier (1989) 2.39682539683

#While we were at it, we computed the average rating of the 10 nearest neighbors to Toy Story:
#In [25]:

#avgRating
#Out[25]:
3.1430700237115254


#While we were at it, we computed the average rating of the 10 nearest neighbors to Toy Story:
#In [25]:

#avgRating
#Out[25]:
3.1430700237115254
