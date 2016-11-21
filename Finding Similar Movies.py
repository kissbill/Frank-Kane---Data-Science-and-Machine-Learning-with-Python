# Finding Similar Movies

#We'll start by loading up the MovieLens dataset. Using Pandas, 
#we can very quickly load the rows of the u.data and u.item files that we care about, 
#and merge them together so we can work with movie names instead of ID's. 
#(In a real production job, you'd stick with ID's and 
#worry about the names at the display layer to make things more efficient. 
#But this lets us understand what's going on better for now.)

In [1]:

import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('e:/sundog-consult/udemy/datascience/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3))
#ami el van valasztva tabulatorral, 3 db oszlop, azokat nevezi el

m_cols = ['movie_id', 'title']
movies = pd.read_csv('e:/sundog-consult/udemy/datascience/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2))
#itt a film id van osszekotve a cimmel, innen szedi ki hogy melyik movie id-hoz melyik film cim tartozik
ratings = pd.merge(movies, ratings)


ratings.head()


#movie_id	title	user_id	rating
#0	1	Toy Story (1995)	308	4
#1	1	Toy Story (1995)	287	5
#2	1	Toy Story (1995)	148	4
#3	1	Toy Story (1995)	280	4
#4	1	Toy Story (1995)	66	3


#Now the amazing pivot_table function on a DataFrame will construct a user / movie rating matrix. 
#Note how NaN indicates missing data - movies that specific users didn't rate.

movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
movieRatings.head()

title	'Til There Was You (1997)'
user_id																					
0			NaN
1			NaN
2			NaN

#minden usert es minden movie-t tartalmaz, 
#1 user tobb filmet is osztalyozhat, minden intersection-ban ahol van ertek ott adott osztalyzatot
#minden userre kivonhatjuk, hogy milyen filmeket osztalyzot es minden filmre kivonhatjuk ki osztalyozta oket
#user es item filter-ingre hasznos ez 
#ha kapcsolatot akarok keresni userek kozott korelaciokat kereshetek a user sorokban 
#item-re ha filmeket akarok ,akkor pedig oszlopokat figyelek ami a user vislekedesen alapszik

#Let's extract a Series of users who rated Star Wars:
#In [4]:

starWarsRatings = movieRatings['Star Wars (1977)']
starWarsRatings.head()
#Out[4]:
#user_id
#0     5
#1     5
#2     5
#3   NaN ---> o nem ertekelte pl 
#4     5
#Name: Star Wars (1977), dtype: float64

#egy adott oszlopot osszevet minden mas oszlopal ami benne van a data DataFrame-ben
#hogy megadja a korelaciokat es visszadja nekunk azt 

#Pandas' corrwith function makes it really easy to compute the pairwise correlation of Star Wars' v
#ector of user rating with every other movie! After that, 
#we'll drop any results that have no data, and construct a
# new DataFrame of movies and their correlation score (similarity) to Star Wars:
#In [5]:

similarMovies = movieRatings.corrwith(starWarsRatings) #megadjuk hogy csak a starWarsRatings-el correlaljon
similarMovies = similarMovies.dropna() # droppoljuk a NAN-akat -> azok maradnak csak amik korrelalnak a SW-vel
df = pd.DataFrame(similarMovies)# uj DataFrame-et csinalunk a data set-bol 
df.head(10)

#(That warning is safe to ignore.) 
#Let's sort the results by similarity score, and we should have the movies most similar to Star Wars! Except... we don't. 
#These results make no sense at all! This is why it's important to know your data - clearly we missed something important.

similarMovies.sort_values(ascending=False)


#Our results are probably getting messed up by movies that have only been viewed by 
#a handful of people who also happened to like Star Wars. 
#So we need to get rid of movies that were only watched by a few people that are producing spurious results.
# Let's construct a new DataFrame that counts up how many ratings exist for each movie, 
#and also the average rating while we're at it - that could also come in handy later.

import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
movieStats.head()

#kisiltazuk hany ember pontozta es mennyi az atlaga igy a filmnek
#rating
#size						mean
#title		
#'Til There Was You (1997)	9	2.333333
#1-900 (1994)	5	2.600000
#101 Dalmatians (1996)	109	2.908257
#12 Angry Men (1957)	125	4.344000
#187 (1997)	41	3.024390

#Let's get rid of any movies rated by fewer than 100 people, and check the top-rated ones that are left:
In [9]:

popularMovies = movieStats['rating']['size'] >= 100
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]

#100 might still be too low, but these results look pretty good as far as "well rated movies that people have heard of." 
#Let's join this data with our original set of similar movies to Star Wars:

df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
df.head()

#And, sort these new results by similarity score. That's more like it!

df.sort_values(['similarity'], ascending=False)[:15]

#Ideally we'd also filter out the movie we started from - of course Star Wars is 100% similar to itself. 
#But otherwise these results aren't bad.
#Activity
#100 was an arbitrarily chosen cutoff. Try different values - what effect does it have on the end results?