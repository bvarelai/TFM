import pandas as pd
from lenskit import batch, topn, datasets
from lenskit.metrics.predict import rmse
from lenskit.crossfold import partition_users, SampleN
from lenskit.algorithms import als, Recommender

def als_lenskit(df: pd.DataFrame):   

    # run the evaluation
    all_recs = []
    all_preds = []
    all_test = []
    samp = xf.SampleN(5)
    for train, test in partition_users(df.ratings, 5, samp):
    all_test.append(test)
    # configure the algorithm
    algo = als.BiasedMF(50)
    # make sure it can recommend, not just predict
    algo = Recommender.adapt(algo)
    # train the algorithm
    algo.fit(train)
    # predict ratings
    preds = batch.predict(algo, test)
    all_preds.append(preds)
    # produce recommendations
    users = test["user_id"].unique()
    recs = batch.recommend(algo, users, 20)
    all_recs.append(recs)
    # integrate test data; since splits are disjoint users
    # for a single data set, this is sufficient
    recs = pd.concat(all_recs, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    test = pd.concat(all_test, ignore_index=True)
    # compute RMSE
    print("RMSE:", rmse(preds["prediction"], preds["rating"]))
    # compute nDCG for each user
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    metrics = rla.compute(recs, test)
    # and compute overall system performance
    print("NDCG:", metrics["ndcg"].mean())