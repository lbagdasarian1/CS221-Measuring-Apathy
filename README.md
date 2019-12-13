# Measuring apathy toward mass shootings by analyzing Twitter sentiment
Stanford CS 221 final project -- Measuring apathy toward mass shootings through Twitter sentiment analysis

``getTweetDataCSV.py`` scrapes tweets from Twitter filtering by keyword (in our case the location of a given shooting) and accepting start and end timestamp parameters. Our logic interpreteing json responses received from the web in getTweets is based on Jefferson-Henrique's GetOldTweets project (https://github.com/Jefferson-Henrique/GetOldTweets-python). Tweets randomly sampled from each day tested are written to a CSV file named in the following format: [shooting name]\_[num days since first sampled]\_[num tweets collected].csv e.g. ``vegas_1_300.csv`` if we have written 300 tweets about the Las Vegas shooting sampled from the first day we examine.

``logreg.ipynb`` runs our logistic regression sentiment classification experiments. We preprocess and obtain mean GloVe embedding representations of tweets in our training dataset (Sentiment140 dataset) and train a binary classifier using logistic regression (implemented using Keras). We read data from the CSV files containing tweets created by ``getTweetDataCSV.py``, prune irrelevant topics using LDA, and plot changes in sentiment over time.

``lstm.ipynb`` runs our lstm sentiment classification experiments. We preprocess training data (again from Sentiment140 dataset) generate glove embeddings for each word, and use Keras to train an LSTM classifier using a neural network (architecture described in writeup). We prune irrelevant topics using LDA and plot changes in sentiment over time.

``graphs.ipynb`` Generates plots of graph features using matplotlib.
