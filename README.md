
__Approach__

We have 35,644 data points, each with 18 input features, a unique ID, and a single binary label. Most of the data are unlabeled, the "test set". Of the labeled data, the "training set", 1,075 have positive labels and 2,143 have negative labels. So this is a supervised learning problem. We note that it is a fairly well-balanced training set.

The first step was to explore and summarize the data. Immediately we see that there are no missing data! This frees us from the data cleaning step. Next, we see that 9 of the features are simply binary. The other 9 are decimal, but since we do not have any context for the meaning of these features, it would be wrong to assume these are ordinal features (much less continuous). Therefore, we transform these 9 features using a one-hot (one-of-K) encoder. This expands the feature space to 53 binary features.

We chose from the start to use a Random Forest Classifier (RFC), and we committed to this for the entirety of the project (a Support Vector Classifier (SVC) would have also been appropriate: see comments below). We used the implementation provided by the Scikit-Learn library.

In order to choose the hyperparameters for the RFC estimator, we used the hyperopt library. This allowed us to specify the hyperparameter space in which to optimize the estimator. Each trial estimator was validated with the standard 10-fold cross-validation. We used the Area under the Receiver Operating Characteristic (ROCAUC) to score the estimators (the F1 score would have also been appropriate).

After optimizing over the hyperparameter space, the estimator (with optimal hyperparameters) is run through cross-validation once more in order to get the confusion matrix, which is printed. The estimator is then trained with the entire training set and finally run against the test data. The results are saved to a CSV file. The model is also recorded to a pickle file.

Git was used to version-track the entire project.

We present comparative histograms for each (non-encoded) feature.


__Hangups and Improvements__

We used the sacred library early on with the intent of carefully recording the history of all estimators and their scores. Integrating this with hyperopt was a bit messy, though, and it was quickly realized that the scope of this project does not really demand the book-keeping sacred is good for. For a larger project (like, say, one on the financial planning industry...), especially one with multiple persons involved, we strongly recommend sacred or a similar tool.

A lot of time was unfortunately wasted implementing (and debugging) a feature subset optimization. This was finally scrapped for simplicity after the endeavour proved to be unhelpful and really messy (it can be found in the Git history). It's good to acknowledge, however, that a Principal Component Analysis (PCA) would have been a better choice here, but it seems that would violate the terms of the project.

As stated, we committed to a RFC early on. A SVC would have been a fine choice as well, especially since (after encoding) the data are already scaled to [0, 1]. It would have been straightforward to include a "hyperchoice" between a RFC and a SVC with hyperopt, but time just ran out. Another excellent option would be to combine a RFC and a SVC into an ensemble of models.

Frustratingly, the ROCAUC score could not be improved beyond ~0.61. This ain't great, but without more context it's hard to judge this result qualitatively (we considered asking, but it seemed like cheating).

An interesting trick that may have been very useful is comparing the feature distributions between the training and test sets. One can resample from the training set to obtain a set that more closely resembles the test set (study the provided histograms to understand the potential usefulness of this). Time did not allow; besides, in case this seems dubious to you, we agree.

It would have been good to use the predicted probabilities and optimize our own cut-off value. This is essentially choosing the best location along the ROC. In the interest of time, we neglect this.
