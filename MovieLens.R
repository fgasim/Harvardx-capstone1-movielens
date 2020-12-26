# This code was written to develop a recommendation system for the Movielens dataset
# as part of the HardvardX Data Science Program. Code developed by Fidan Gasim. 

# Set global options
options(repos="https://cran.rstudio.com")
options(timeout=10000, digits=10)

# Install packages
packages<-install.packages(c("caret", "data.table", "devtools", "dplyr", 
                   "DT", "ggplot2", "ggthemes", "h2o", "irlba", 
                   "kableExtra", "knitr", "lubridate", "Matrix.utils", 
                   "purrr", "RColorBrewer", "recommenderlab", "recosystem", 
                   "scales", "tidyr", "tidyverse", "splitstackshape", "ggrepel", 
                   "tinytex","latexpdf"))

# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Load libraries #####
library(tidyverse)
library(tidyr)
library(dplyr)
library(ggplot2)
library(caret)
library(data.table)
library(knitr) #A General-Purpose Package for Dynamic Report Generation in R
library(kableExtra) #to build common complex tables and manipulate table styles
library(lubridate)
library(Matrix.utils) #Data.frame-Like Operations on Sparse and Dense Matrix Objects.
library(DT) #provides an R interface to the JavaScript library DataTables.
library(RColorBrewer) #Provides color schemes for maps (and other graphics) 
library(ggthemes) #Some extra themes, geoms, and scales for 'ggplot2'.
library(scales)
library(recosystem) #Recommender System using Matrix Factorization
library(purrr)
library(devtools)
library(ggrepel)
library(splitstackshape)
library(tinytex)
library(latexpdf)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Source file #####

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Splitting the data #####

# Validation set will be 10% of MovieLens data 
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Split data into training and test sets - test set will be 10% of edx 
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# Data Exploration & Visualization #####

class(edx)

str(edx)

dim(edx)

head(edx)

# Ratings #####

# Count the number of ratings per rating

edx %>% 
  group_by(rating) %>% 
  summarize(n=n())

# Plot - Distribution of ratings

edx %>% 
  group_by(rating) %>% 
  summarize(n=n()) %>%
  ggplot(aes(x=rating, y=n))+
  geom_bar(stat="identity", color="white")+ 
  theme(axis.text.x =element_text(angle = 45, hjust = 1)) + 
  labs(x="Rating", y="Count", title="Distribution of Ratings by Count")+
  scale_y_continuous(breaks = c(0,1000000,2000000,3000000),
                     labels=c("0","1M","2M","3M"))+
  theme_economist()

# Movies #####

# Number of movies in edx dataset

n_distinct(edx$movieId)

# Number of ratings per movie 

edx %>% 
  group_by(movieId, title) %>%
  summarise(n=n())%>%
  head()

# Movies with the highest number of ratings (popular movies)

edx %>% 
  group_by(movieId, title) %>%
  summarise(n=n())%>%
  arrange(desc(n)) %>%
  head()

# Average number of ratings per movie 

avgratings<-edx %>% 
  group_by(movieId, title) %>%
  summarise(n=n())

mean(avgratings$n)

rm(avgratings)

# Plot - Distribution of movies by number of ratings  

edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "white") + 
  scale_x_log10() + 
  labs(x="Number of Ratings", y="Number of Movies", 
       title="Distribution of Movies by Number of Ratings")+
  theme_economist()

# Create movies dataframe - Extract year from movie title 

movies_df <- edx%>%
  mutate(title = str_trim(title)) %>%
  extract(title, c("temp_title", "year"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = F) %>%
  mutate(year = if_else(str_length(year) > 4, as.integer(str_split(year, "-", simplify = T)[1]), 
                        as.integer(year))) %>%
  mutate(title = if_else(is.na(temp_title), title, temp_title)) %>%
  select(-temp_title)  %>%
  mutate(genres = if_else(genres == "(no genres listed)", `is.na<-`(genres), genres))
head(movies_df)

# Number of movies released per year 

movies_per_year <- movies_df %>%
  na.omit() %>% 
  select(movieId, year) %>% 
  group_by(year) %>% 
  summarise(count = n())  %>% 
  arrange(year)

# Fill missing years

movies_per_year <- movies_per_year %>%
  complete(year = full_seq(year, 1), fill = list(count = 0))
print(movies_per_year)

# Plot - Number of movies released per year 

movies_per_year %>%
  ggplot(aes(x = year, y = count)) +
  geom_line(color="blue")+
  labs(x="Year",y="Count", title="Number of movies released per year")+
  scale_y_continuous(breaks = c(0,200000,400000,600000,800000),
                     labels=c("0","200,000","400,000","600,000","800,000"))+
  theme_economist()

rm(movies_per_year)

# Title #####

# Top 10 movie titles by number of ratings 

edx %>%
  group_by(title) %>%
  summarize(count=n()) %>%
  top_n(10,count) %>%
  arrange(desc(count))

# Plot top 10 movies by number of ratings 

edx %>%
  group_by(title) %>%
  summarize(count=n()) %>%
  top_n(10,count) %>%
  arrange(desc(count)) %>% 
  ggplot(aes(x=reorder(title, count), y=count)) +
  geom_bar(stat='identity', color="white") + 
  coord_flip()+
  labs(x="", y="Number of Ratings", 
       title="Top 10 Movies by Number of Ratings")+
  theme_economist()+
  theme(axis.text.y = element_text(size=6), 
        plot.title = element_text(size=12))

# Users #####

# Number of users in edx dataset

n_distinct(edx$userId)

# Number of movies rated per user

edx %>% 
  group_by(userId) %>%
  summarise(n=n())%>%
  head()

# Some users rate very few movies
edx %>% 
  group_by(userId) %>%
  summarise(n=n())%>%
  arrange(n) %>%
  head()

# Some users rate many movies
edx %>% 
  group_by(userId) %>%
  summarise(n=n())%>%
  arrange(desc(n)) %>%
  head()

# Average number of movies rated per user

userratings<-edx %>% 
  group_by(userId) %>%
  summarise(n=n())

mean(userratings$n)

# Percentage of users who rated less than 20 movies

mean(userratings$n<20)

rm(userratings)

# Distribution of Users by the number of movies they rated 

edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "white") + 
  scale_x_log10() +
  labs(x="Number of Ratings", y="Number of Users", 
       title="Distribution of Users by Number of Movies Rated")+
  theme_economist()

# Heatmap of users vs movies 

users <- sample(unique(edx$userId), 100)

edx %>% filter(userId %in% users) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% 
  select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")
title("User x Movie Matrix")

rm(users)

# Timestamp #####

# Convert timestamp to more readable format

timestamp_df<-edx%>%
  mutate(timestamp = as_datetime(timestamp))
head(timestamp_df)

# Rating period covered in dataset

start_year<-lubridate::year(min(timestamp_df$timestamp))
start_year

end_year<-lubridate::year(max(timestamp_df$timestamp))
end_year

rating_period<-tibble(`Start Date` = start_year,
                      `End Date` = end_year) %>%
              mutate(Period = end_year-start_year)
rating_period

rm(timestamp_df,start_year,end_year,rating_period)

# Plot Rating Distribution Per Year (number of ratings per year)
  
edx %>% mutate(year = lubridate::year(as_datetime(timestamp))) %>%
  ggplot(aes(x=year)) +
  geom_bar(color = "white") + 
  scale_x_continuous(breaks=1995:2010)+
  labs(x="Year", y="Number of Ratings", 
       title="Rating Distribution Per Year")+
  theme_economist()+
  theme(axis.text.x = element_text(angle = 90))

# list of dates with the most ratings - Blockbuster/ Popular films

edx %>% mutate(date = date(as_datetime(timestamp))) %>%
  group_by(date, title) %>%
  summarise(count = n()) %>%
  arrange(-count) %>%
  head(10)

# Plot average ratings over time

edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating))%>%
  ggplot(aes(date,rating))+
  geom_point()+
  geom_smooth()+
  labs(title="Average Ratings Over Time")+
  theme_economist()

# average rating per movie

avg_rating <- movies_df%>%
  na.omit() %>%
  select(title, rating, year) %>%
  group_by(title, year) %>%
  summarise(count = n(), mean = mean(rating)) %>%
  ungroup() %>%
  arrange(desc(mean))
head(avg_rating)

# Calculate Weighted Rating

# MR = average rating for the movie = Mean Rating
# C = number of ratings for the movie = Count
# Min = minimum count required to be listed in the Top 250
# OMR = the mean rating across all movies = Overall Mean Rating
weighted_rating <- function(MR, C, Min, OMR) {
  return (C/(C+Min))*MR + (Min/(C+Min))*OMR
}

avg_rating <- avg_rating %>%
  mutate(wr = weighted_rating(mean, count, 500, mean(mean))) %>%
  arrange(desc(wr)) %>%
  select(title, year, count, mean, wr)
head(avg_rating)

# best movie of every decade based on score

avg_rating %>%
  mutate(decade = year  %/% 10 * 10) %>%
  arrange(year, desc(wr)) %>%
  group_by(decade) %>%
  summarise(title = first(title), wr = first(wr), 
            mean = first(mean), count = first(count))

rm(avg_rating)

# Genres #####

# number of movies per genre 

edx %>% group_by(genres) %>% 
  summarise(n=n()) %>%
  head()

# split genres column into separate rows - takes long time to run

genre_df <- movies_df %>% separate_rows(genres, sep = "\\|") 
head(genre_df)

# top 15 genres by number of ratings

genre_df%>%group_by(genres)%>%
  summarize(count=n())%>%
  top_n(15,count) %>%
  arrange(desc(count))

# Plot - Ratings Distribution by Genre (number of ratings)

genre_df%>%group_by(genres)%>%
  summarize(count=n())%>%
  top_n(15,count) %>%
  ggplot()+
  geom_bar(aes(x=reorder((factor(genres)),count), y=count), 
           stat="identity", width=0.8) +
  labs(x="Genres", y="Number of Ratings", 
       title="Top Genres by Number of Ratings") +
  scale_y_continuous(breaks = c(0,1000000,2000000,3000000,4000000),
                     labels=c("0","1M","2M","3M","4M"))+
  theme_economist()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Plot - Average Rating per Genre

edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 100000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x="Genres",y="Average", title="Average Rating per Genre")

# Genre popularity over time

genre_popularity <- genre_df %>%
  na.omit() %>% 
  select(movieId, year, genres) %>% 
  mutate(genres = as.factor(genres)) %>% 
  group_by(year, genres) %>% 
  summarise(count = n()) %>% 
  complete(year = full_seq(year, 1), genres, fill = list(number = 0)) 

# Plot - genre popularity over time (select five for readability)

genre_popularity %>%
  filter(year > 1930) %>%
  filter(genres %in% c("Drama", "Comedy", "Action", "Thriller", "Adventure")) %>%
  ggplot(aes(x = year, y = count)) +
  geom_line(aes(color=genres)) + 
  scale_fill_brewer(palette = "Paired")+
  scale_y_continuous(breaks = c(0,100000,200000,300000,400000),
                     labels=c("0","100,000","200,000","300,000","400,000"))+
  labs(x="Year",y="Count", title="Genre Popularity Over Time (by Number of Ratings)")+
  theme_economist()

# Genre performance over time (as per user ratings)

genre_ratings <- genre_df %>%
  na.omit() %>%
  select(movieId, year, genres, rating) %>%
  mutate(decade = year  %/% 10 * 10) %>%
  group_by(year, genres) %>%
  summarise(count = n(), avg_rating = mean(rating)) %>%
  ungroup() %>%
  mutate(wr = weighted_rating(mean, count, 5000, mean(mean))) %>%
  arrange(year)

# Plot - Genre performance over time (as per user ratings)

genre_ratings %>%
  filter(genres %in% c("Drama", "Comedy", "Action", "Thriller",
                       "Adventure")) %>%
  ggplot(aes(x = year, y = wr)) +
  geom_line(aes(group=genres, color=genres)) +
  geom_smooth(aes(group=genres, color=genres)) +
  facet_wrap(~genres)+
  labs(x="Year", y="Weighted Rating",
       title="Genre Performance Over Time (by Average Ratings)")

rm(genre_popularity,genre_df,genre_ratings)

# Example User x Movie Matrix

x<-matrix(nrow=4, ncol=5)
colnames(x)<-c("Movie 1", "Movie 2", "Movie 3", "Movie 4", "Movie 5")
rownames(x)<-c("User 1", "User 2", "User 3", "User 4")
x[1,] <- c("?",3,2,"?",5)
x[2,] <- c(3,4,"?","?",2)
x[3,] <- c(5,"?",2,"?",3)
x[4,] <- c(4,"?","?",3,"?")

# Data Cleaning/ Pre-Processing #####

train_set<-train_set%>%select(userId, movieId, rating, title)
test_set<-test_set%>%select(userId, movieId, rating, title)

# Define Root Mean Squared Error (RMSE) - loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Modeling #####

# create a results table with all the RMSEs

rmse_results <- tibble(Method = "Project Goal", RMSE = 0.8649)

# Mean of observed ratings

mu<-mean(train_set$rating)
mu

# Naive RMSE - predict all unknown ratings with overall mean 

naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse

# Update results table with naive RMSE

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Naive RMSE", 
                                 RMSE = RMSE(test_set$rating, mu)))
rmse_results

# Calculate movie effect

bi <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
head(bi)

# Plot - Distribution of Movie Effect (bi)

bi %>% ggplot(aes(x = b_i)) + 
  geom_histogram(bins=10, col = I("white")) +
  labs(x="Movie Effect (bi)", y="Count", 
       title="Distribution of Movie Effect")+
  theme_economist()

# Predict ratings with mean + bi  
y_hat_bi <- mu + test_set %>% 
  left_join(bi, by = "movieId") %>% 
  .$b_i

RMSE(test_set$rating, y_hat_bi)

# Update results table with movie effect RMSE

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Movie Effect", 
                                 RMSE = RMSE(test_set$rating, y_hat_bi)))
rmse_results

# Calculate user effect

bu <- train_set %>% 
  left_join(bi, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Plot - User Effect Distribution

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "white")+
  labs(x="User Effect (bu)", y="Count", title="Distribution of User Effect")+
  theme_economist()

# Predict ratings with mean + bi + bu
y_hat_bu <- test_set %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE(test_set$rating, y_hat_bu)

# Update results table with movie effect RMSE

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Movie + User Effect", 
                                 RMSE = RMSE(test_set$rating, y_hat_bu)))
rmse_results

# Evaluate Model Results

# calculate biggest residuals (errors)

test_set %>% 
  left_join(bi, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%  
  head(10)

# create a database of movie titles

movie_titles <- train_set %>% 
  select(movieId, title) %>%
  distinct()

# 10 best movies according to bi 

bi %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  head(10)  %>% 
  select(title)

# 10 worst movies according to bi 

bi %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  head(10)  %>% 
  select(title)

# number of ratings for "10 best movies according to bi"

train_set %>% count(movieId) %>% 
  left_join(bi, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  head(10) %>% 
  select(n, title)

# number of ratings for "10 worst movies according to bi"

train_set %>% count(movieId) %>% 
  left_join(bi, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  head(10) %>% 
  select(n, title)

# Regularization #####

# Define a set of lambdas to tune

lambdas <- seq(0, 10, 0.25)

# Tune the lambdas using regularization function

regularization <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
 
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

# Plot - lambdas vs RMSE

qplot(lambdas, regularization)  

# Choose the lambda which produces the lowest RMSE

lambda<- lambdas[which.min(regularization)]
lambda

# Calculate the movie and user effects with the best lambda (parameter) 

mu <- mean(train_set$rating)

# Movie effect (bi)

bi_reg <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# User effect (bu)

bu_reg <- train_set %>% 
  left_join(bi_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Prediction with regularized bi and bu 

y_hat_reg <- test_set %>% 
  left_join(bi_reg, by = "movieId") %>%
  left_join(bu_reg, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Update results table with regularized movie + user effect

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Regularized Movie + User Effect", 
                                 RMSE = RMSE(test_set$rating, y_hat_reg)))

# Regularization made a small improvement on RMSE

rmse_results

# Model Evaluation

# Top 10 best movies after regularization

test_set %>% 
  left_join(bi_reg, by = "movieId") %>%
  left_join(bu_reg, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  arrange(desc(pred)) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)

# Top 10 worst movies after regularization

test_set %>% 
  left_join(bi_reg, by = "movieId") %>%
  left_join(bu_reg, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  arrange(pred) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)

# Matrix Factorization #####

# convert training set into user-movie matrix (Code Not Run as it takes up too much memory)
train_matrix <- train_set %>% 
  select(userId, movieId, rating) %>% 
  spread(movieId, rating) %>% 
  as.matrix()
head(train_matrix)

rm(train_matrix)

# use recosystem package instead

set.seed(123,sample.kind="Rounding") # randomized

# Convert the training and test sets into recosystem format

train_data <-  with(train_set, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))

test_data  <-  with(test_set,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))

# Create the model object

r <-  recosystem::Reco()

# Select the tuning parameters

opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30), 
                                       costp_l2 = c(0.01, 0.1), 
                                       costq_l2 = c(0.01, 0.1),
                                       lrate = c(0.01, 0.1),
                                       nthread = 4, 
                                       niter = 10))

# Train the algorithm  

r$train(train_data, opts = c(opts$min, nthread = 4, niter = 10))

# Calculate the predicted values  
y_hat_mf <-  r$predict(test_data, out_memory())

# Update the results table 

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Matrix Factorization", 
                                 RMSE = RMSE(test_set$rating, y_hat_mf)))
rmse_results

# Final Validation #####

# Final Validation - Matrix Factorization

set.seed(1234, sample.kind = "Rounding")

# Convert 'edx' and 'validation' sets to recosystem input format

edx_rec<- with(edx, data_memory(user_index = userId, 
                                  item_index = movieId, 
                                  rating = rating))

validation_rec<-  with(validation, data_memory(user_index = userId, 
                                               item_index = movieId, 
                                               rating = rating))

# Create the model object

r <-  recosystem::Reco()

# Tune the parameters

opts <-  r$tune(edx_rec, opts = list(dim = c(10, 20, 30), 
                                     costp_l2 = c(0.01, 0.1), 
                                     costq_l2 = c(0.01, 0.1),
                                     lrate = c(0.01, 0.1),
                                     nthread = 4, 
                                     niter = 10))

# Train the model

r$train(edx_rec, opts = c(opts$min, nthread = 4, niter = 10))

# Calculate the prediction

y_hat_mf_final <-  r$predict(validation_rec, out_memory())

# Update the results table

rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Final Validation - Matrix Factorization", 
                                 RMSE = RMSE(validation$rating, y_hat_mf_final)))
rmse_results

# top 10 best movies predicted by matrix factorization

validation %>% 
  mutate(pred = y_hat_mf_final) %>% 
  arrange(desc(pred)) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)

# top 10 worst movies predicted by matrix factorization

validation %>% 
  mutate(pred = y_hat_mf_final) %>% 
  arrange(pred) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)
