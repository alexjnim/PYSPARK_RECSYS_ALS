# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# In Spark, there is one paritcular recommendation algorithm, Alternating Least Squares (ALS). This algorithm leverages collaborative filtering, which makes recommendations based only on which items users interacted with in the past. That is, it does not require or use any additional features about the users or the items.

# +
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr


# select 4 cores to process this
spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .config("spark.executor.cores", '4')\
        .getOrCreate()
# -

# # Loading data

ratings = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("../data/goodbooks-10k-master/ratings.csv")
ratings.printSchema()
ratings.createOrReplaceTempView("dfTable")

ratings.show(3)

books = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("../data/goodbooks-10k-master/books.csv")
books.printSchema()
books.createOrReplaceTempView("dfTable")

book_names = books.select("book_id", "title", "authors")
book_names.show(5)

# # Build Model and Fit

# +
from pyspark.ml.recommendation import ALS

#split data into training and test set
training, test = ratings.randomSplit([0.8, 0.2])

als = ALS()\
  .setMaxIter(5)\
  .setRegParam(0.01)\
  .setUserCol("user_id")\
  .setItemCol("book_id")\
  .setRatingCol("rating")
# print(als.explainParams())
# -

alsModel = als.fit(training)
predictions = alsModel.transform(test)

# # Evaluate

# When covering the cold-start strategy, we can set up an automatic model evaluator when working with ALS. One thing that may not be immediately obvious is that this recommendation problem is really just a kind of regression problem. Since we’re predicting values (ratings) for given users, we want to optimize for reducing the total difference between our users’ ratings and the true values. We can do this using the RegressionEvaluator.

# in Python
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator()\
  .setMetricName("rmse")\
  .setLabelCol("rating")\
  .setPredictionCol("prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = %f" % rmse)

# # Recommendation Results

# We can now output the top 𝘬 recommendations for each user or book. The model’s recommendForAllUsers method returns a DataFrame of a user_id, an array of recommendations, as well as a rating for each of those books. recommendForAllItems returns a DataFrame of a book_id, as well as the top users for that book:

# +
from pyspark.sql.functions import col

# generate top 10 book recs for each user
alsModel.recommendForAllUsers(10)\
  .selectExpr("user_id", "explode(recommendations)").show(5)

# generate top 10 user recommendations for each book 
alsModel.recommendForAllItems(10)\
  .selectExpr("book_id", "explode(recommendations)").show(5)
# -

# ### select test user

# +
test_user_id = 8

test_user = ratings.filter(ratings['user_id'] == test_user_id)
joinExpression = test_user["book_id"] == book_names['book_id']
test_user.join(book_names, joinExpression, joinType)\
 .orderBy('rating', ascending = False).show(truncate = False)
# -

# ### filter for results and join with book names

# +
userRecs = alsModel.recommendForAllUsers(10)

test_userRecs = userRecs.filter(userRecs['user_id'] == test_user_id)\
                    .selectExpr("user_id", "explode(recommendations)")

test_userRecs = test_userRecs.select("user_id", 'col.*')
# -

joinExpression = test_userRecs["book_id"] == book_names['book_id']
joinType = "inner"
test_userRecs.join(book_names, joinExpression, joinType).show(truncate = False)



# ### can also find top users for a given book

# +
test_book_id = 177

book_names.filter(book_names['book_id'] == test_book_id).show()


# +
bookRecs = alsModel.recommendForAllItems(10)\
                    .selectExpr("book_id", "explode(recommendations)")

test_bookRec = bookRecs.filter(bookRecs['book_id'] == test_book_id)\
                        .select("book_id", "col.*")

test_bookRec.show()
# -



# # Further evaluation metrics

# A RankingMetric allows us to compare our recommendations with an actual set of ratings (or preferences) expressed by a given user. RankingMetric does not focus on the value of the rank but rather whether or not our algorithm recommends an already ranked item again to a user. 
#
# First, we need to collect a set of highly ranked movies for a given user. In our case, we’re going to use a rather low threshold: movies ranked above 2.5. Tuning this value will largely be a business decision:

# in Python
from pyspark.mllib.evaluation import RankingMetrics, RegressionMetrics
from pyspark.sql.functions import col, expr
perUserActual = predictions\
  .where("rating > 2.5")\
  .groupBy("user_id")\
  .agg(expr("collect_set(book_id) as books"))

# At this point, we have a collection of users, along with a truth set of previously ranked movies for each user. Now we will get our top 10 recommendations from our algorithm on a per-user basis. We will then see if the top 10 recommendations show up in our truth set. If we have a well-trained model, it will correctly recommend the movies a user already liked. If it doesn’t, it may not have learned enough about each particular user to successfully reflect their preferences:

perUserPredictions = predictions\
  .orderBy(col("user_id"), expr("prediction DESC"))\
  .groupBy("user_id")\
  .agg(expr("collect_list(book_id) as books"))

# Now we have two DataFrames, one of predictions and another the top-ranked items for a particular user. We can pass them into the RankingMetrics object. This object accepts an RDD of these combinations, as you can see in the following join and RDD conversion:

# in Python
perUserActualvPred = perUserActual.join(perUserPredictions, ["user_id"]).rdd\
  .map(lambda row: (row[1], row[2][:15]))
ranks = RankingMetrics(perUserActualvPred)

# Now we can see the metrics from that ranking. For instance, we can see how precise our algorithm is with the mean average precision. We can also get the precision at certain ranking points, for instance, to see where the majority of the positive recommendations fall:

ranks.meanAveragePrecision
ranks.precisionAt(5)


