# Machine Learning for the real world

* Begin a Big Data project with defining the business goals and measuring metrics to be improved, not with downloading 
some CSV data and launching a Jupiter notebook.
* Avoid disconnect between data scientists and software developers / devops.
* Deploy a solid Machine Learning infrastructure into production even before you start training your first model.
* Complement the algorithms from Tensorflow, Keras and Scikit-Learn with the solid software-engineering
practices required to deploy, operate, monitor and update ML-based products.
* Detect frequent data problems (missing data, skewed datasets, noisy features, etc)

We call this **Production First Machine Learning**

iwlearn captures all the best practices and lessons learned we had so far while creating, operating, monitoring and
updating real-world Machine Learning models on immowelt.de, a leading German real estate marketplace.

By sharing this project, we not only hope you'd avoid the mistakes we've made, but also we suggest some process 
patterns allowing to increase the ROI of Big Data projects.


# Contents

* [Process Patterns](#process-patterns)
    * [Start with Metrics](#start-with-metrics)
    * [Production first](#production-first)
    * [Rules before Models](#rules-before-models)
    * [Avoid Data Contamination](#avoid-data-contamination)
    * [Model Aging](#model-aging)
    * [Soft Model Rollout](#soft-model-rollout)
* [Feature Engineering Patterns](#feature-engineering-patterns)
    * [Simple Value](#simple-value)
    * [Clustered Mean Removal](#clustered-mean-removal)
    * [All Combination](#all-combination)
    * [Frequency Features](#frequency-features)
    * [Representing Geography](#representing-geography)
    * [One-Hot Encoding](#one-hot-encoding)
    * [Image augmentation](#image-augmentation)
    * [Eye Check](#eye-check)   
* [Developing with iwlearn](#developing-with-iwlearn)
    * [Installation and system requirements](#installation-and-system-requirements)
    * [Tutorial](#tutorial)
    * [Advanced Tutorial](#advanced-tutorial)
    * [Architectural concepts](#architectural-concepts)
    * [Reference](#reference)
 
# Process Patterns

## Start with Metrics

### Intent
* Find out whether the problem at hand needs to be solved at all and if yes, what metrics exactly capture the
desired outcome.
* Create a way to measure problem solving progress.
* Create a way to monitor whether the AI model running in production still adequately solves the problem.
 
### Problem
Most machine learning tutorials use one simple metric to measure the performance of the models (usually, it is
accuracy). In the business setting, accuracy might not represent the correct metric to be optimized or tracked. What 
metrics are the best representation of the business goal?
 
### Discussion
Consider the following example: the model is a binary classifier deciding whether to give 10% rebate to our customer,
based on information we know about him and his real estate. The purpose of the model is to increase the conversion
rate. The false positive would be then reducing the price in the situations when the user would also pay the normal 
price anyway. The false negative is not giving the price reduction for persons who have really needed it and would 
therefore bounce. 

If our goal and the top priority is increasing the conversion rate, then the false positives are nice to be avoided,
but they are not a critical problem (let's give them the weight of 0.1), while it is absolutely important to avoid
the false negatives (their weight should be 0.9). 

On the other hand, if our top priority is to increase the revenue instead, then vice versa, the false positives must 
get the weight 0.9 and the false negatives the weight 0.1.

The accuracy metric is giving to the both cases the weight 0.5, which is almost never right from the business 
perspective. 

In this example, either the recall metric should be used to maximize conversion rate (precision metric to maximize 
revenue), or we should define a custom loss function directly measuring the bounce rate / the revenue, and evaluate 
our model in terms of this loss function as compared with the bounce rate / revenue given without usage of the model.

### Solution
* Define a numeric metric adequately representing the property of the business system you want to improve.
* Measure this metric for some period of time (ideally using pre-saved historical data) to get the idea about how good
is this metric before developing the model.
* Decide whether this metric can be substantially improved with a model. For example, a bounce rate of 0.01 would be not the best
case of using AI models to be improved, because creating of AI models is a very costy process, and the outcome will
not pay off the expenses, because we can only improve the metric by 1 percent point. On the other hand, a bounce rate 
of 0.9 is too not very suitable for AI models, because it can be most probably improved by using some simple 
"quick win" solution. An AI model can help you get from 0.6 to 0.2 though.
* Use the defined metric when training and evaluating the model.
* Deploy the model for part of users in an A/B test and compare the metric changes the model creates in the production.
* After rolling out the model to all users, monitor the metrics and its changes over time to be alerted in case of
any changes in data or user behavior.

### Disadvantages
If you're a novice in machine learning and using tutorials to get started, most probably you will get some kind of 
unbiased metric by default (accuracy, RMSE, categorical cross-entropy). Usually it is not so easy to change the
metrics, because sometimes you would need to implement them yourself.


## Production first
### Intent
Avoid type errors when rolling out models into production.  

### Problem
ML models do not have any typization and would happily accept any garbage as input, silently producing random numbers
as output. Accidental changes in the data conversion or order of columns can force the model to make wrong predictions. 
This problem is especially critical for the models influencing the humans, for example for the dynamic pricing models. 

Besides, if the model uses a lot of features, a mistake in one feature implementation will just reduce the model
performance somewhat, so the bug can live in production for weeks and months before being caught.

### Discussion
Most ML tutorials start with loading a data set from CSV files or some online data source. Then the raw data is processed 
by some inline throw-away code and sent to the model for training. Sometimes, the whole is even implemented not in 
the same programming language that is used in production. 

As a result, when the model is ready, it has to be re-implemented in production, possibly using a different implementation
of the same ML algorithm or even another programming language. 

This situation is a source of all possible errors related to inconsistent implementation. The most common and simple
mistake is that the features used in production are not the same as those used for training. A real story: once we've 
removed one obsolete feature and added a new one. As a result, the new trained model has expected the same input
array shape as the previous model. We've forgot to implement the new feature in the production, so that the new model
was fed with the data of an obsolete feature.  

Some more subltle mistake could be made because of changes in the feature definitions. In another real story, our old
feature was returning the price of the real estate as is, while the new feature code returned -1 for all outlier prices.
Again, we've forgot to implement this tiny change in production.  

This can be avoided if the exact the same feature code is used both for training and production.

### Solution
* Use Python both for training/development and prediction/production phases. No R, no models reimplemented in C#
or Java.
* Inherit data sources from BaseDataSource and implement them to fetch data from the databases or other external
services and to put all the relevant data into a Python dictionary.
* Inherit features from BaseFeature and write code converting the raw data into values suitable for the model.
* Create model by instantiating one of the ready models from iwlearn, or inheriting from some base model.
* Load the training and test sets, train the model as usual
* Deploy the model along with the unchanged features and data sources into production.

### Disadvantages
* You cannot use other languages or tools for training.
* You need to write a substantial amount of code before you get to the business and can start training the model.
* Could be not suitable if the prediction phase is to be performed on the customer device (Android, iOS, Javascript).


## Rules before Models
### Intent
* Reduce development time and costs of the "smart data" solution.
* Have the decisions of the "smart data" system to be easily explained to humans.

### Problem
Developing a AI model both takes time to create useful features and/or neuronal network architectures, and requires
usage of expensive hardware for training. Also, decisions of an AI model usually cannot be explained to humans,
which is especially bad in case of the wrong decisions.
 
### Discussion
Rules can be something as simple as an IF statement following some condition. 

For example, if our goal is to determine whether some user is interested in professional relocation services, the 
condition can be the number of rooms in the apartments he is looking at, and the threshold for a positive decision 
can be "greater than 3". The condition and threshold can be first created by asking the domain experts. 

But better approach would be to support rules with data, especially when determining the optimal threshold. This 
would have the advantage that the performance of the rule can be evaluated before the rollout, and also monitored 
afterwards in production - both in terms of the goal metric we've defined in the previous step.

### Solution
* Prepare a data set containing all historical data needed to evaluate the conditions of the rule. Divide this data
set into training, test and validation sets.
* For each condition in the rule, develop an extraction logic (the "features") to convert the data from the dataset 
into values suitable for an IF statement. The preferred way is to inherit these features from the BaseFeature class
so that they can be reused "as is" for AI models in the future.
* Display the distributions and statistics of the features so you can get an idea, what features are usable for your
goal and which thresholds could be reasonable. For the binary classification tasks we display the feature distributions 
separately for the class 0 and class 1 and select the features having most unsimilar distributions.
* Write a rule with one or several IF statements using the values from the previous step and hard-coded conditions.
* Evaluate the rule using the test set and get the metric you've chosen to optimize.
* Change the rule or thresholds and repeat the steps until you've created a rule having the metrics according to the 
goal you have set.
* Evaluate the rule against the validation set to check that you haven't overfitted to the test set.
* Take the rule into production.

### Disadvantages
* Rules are generally weaker than an AI model, because the latter can support unlimited number of features and 
capture an unlimited number of interactions between them, while rules created by humans are limited by 7 +- 2 features.
* Rules require a classic feature engineering beforeahed, which is infeasible for such inputs like images, texts or
sounds.


## Avoid Data Contamination
### Intent
Ensure that the information contained in the data is the same between training and production.

### Problem
When using historical data for training, the data can be invisibly contaminated with some signal related to the label
and bearing the information from the future. If such data is used for training, the model will not perform well
in production, because it will miss the information it has used during the training.

The problem can be very tricky to spot (see example in the discussion).

### Discussion
To demonstrate the problem we will assume a fictional requirement: we would require to remove company logos from the
images used to depict real estates and would allow this only to customers that have booked this feature.

For this fictional task we would need to create a model delivering 1 for images containing any company logos and 0 
for all other images. The model is intended to be used in production just after image upload, and to send an automatic 
email to the customers with a warning. 

For training of this model, we would fetch the images stored currently (at the time point of model training) on NAS, 
label them manually, and go through some kind of CNN.

Now, let's assume, there is some human being who is checking all images manually and removing the ones with company 
logo - except of the images belonging to the customers that have booked this feature. And let's assume, currently
only the companies named "Big Agency" and "International Flats" have booked it. 

If images are not only disabled, but also removed from the NAS, only images from "Big Agency" and "International Flats" 
would contain logos in our dataset. Our model would perform very well during evaluation and suck in production, because 
it would only reliably detect logos on images belonging to "Big Agency" and "International Flats". In the worst case, 
the model will only learn the typical image size used by the "Big Agency" and therefore mark a lot of other innocent 
images accidentally having the same image size. 

In case nobody have informed us about this manual logo cleanup process, it could be very hard to detect and debug this 
situation.

We can see this kind of dataset skew as a signal leaking from the future back to the past. We mostly use historical 
data for our datasets, and they can contain some information or changes that are in the past for us, but in the future 
comparing with the time point in a life cycle of an offer, when our model usually needs to deliver a prediction. So that 
our dataset can get contaminated by the data from "the future".
 
### Solution
The sure way to avoid any possible data contamination without analyzing each and every feature is to make a snapshot 
of the data during the prediction time, exact as it is being seen by the model at that very moment of prediction.

**Therefore, models running in production should create and save sample data to be used for the future training.**

On the other hand, avoid bloat of your samples storage system and do not include immutable data into the sample, only 
a reference to it. 

### Disadvantage
* When starting a new project, there is no model running in production and saving samples for the training. Either
we would need to create an empty model that is only saving the samples, not predicting anything, and wait for some time
until enough data are stored. Or we need to prepare the first dataset from existing historical data to bootstrap 
the training process. In this case, care has to be taken to not include any contaminated data into the dataset. Usage 
of immutable historized data (like data in Clickhouse, Hadoop, some SQL Server tables known to be immutable) is strongly 
recommended.

## Model Aging
### Intent
Keep or improve the model performance over long periods of time.

### Problem
Every model released to production ages with the time. Distribution of feature values change with the time: prices rise,
web site collecting the information removes some fields or introduces new ones, image quality improves and the image
content changes (eg. there will be more fully furnished apartments than before). Therefore, the thresholds used by nodes
of the random forest or the weights used by nodes of a DNN will be more and more outdated with time. 

### Discussion
Each model released to production should be updated regularly to prevent the model aging. The update cycle can vary
from model to model. For example, for dynamic pricing we need to update it every month, while for some image 
recognition, it might be enough to update once a year.

### Solution
* Every model running in production should save the samples its getting for prediction, to build up a training set used
for the next model re-training
* Monitor performance of models in production to recognize the need of a re-training. 

### Disadvantages
* Making a model always re-trainable makes the training procedure and scripts more complicated
* A quick database is required to store samples during prediction time, so that the prediction speed is not slowed down.


## Soft Model Rollout
### Intent
Allow comparizon of several models in a multivariate live test. 

### Problem
Model evaluation with cross-validation or a set-aside validation set, often leads to overly-optimistic results. There
are various reasons for that, for example:
* generally speaking the training sets tend to be small, so that it is easy to overfit on a small test or validation 
set,
* data contamination problems are possible when bootstrapping training sets for a novel model,
* the training set could have skewed class distribution, because one class might be easier to label than another,
* any ETL steps might be re-implemented differently in production compared with the training scripts.

Therefore, the real performance of a model can often be only assessed in the first days of it running in production. In
the use case where the model is influencing humans (like dynamic pricing), it might be a critical issue,
if a model gets deployed, having a bad real performance.

### Discussion
Soft rollout can alleviate the problem. In this case, the old rule or model and the new rule or model are running in
parallel and both are making predictions. Only predictions of the old rule or model are used to perform actions. After
enough predicted samples, the new model can be evaluated and promoted to be the master model whose predictions are
used for performing some actions.

We have also conceived more complicated soft rollout scenarious, where the predictions of both (or several) models are
combined together with a high weight for the old model and low weight of the new model, and the weight of the new model
is getting higher and higher the better its performance is, while the weight of the old model gets lower correspondingly.

### Solution
* Always store the sample data and the corresponding prediction, of all models running in production, even though not
every model is allowed to take any action.
* Always ask all models and rules for their prediction - avoid the algorithm that would call the rule A first, and then
depending on its prediction either call rule B or model C. Instead, let all rules and models to make a prediction,
store it for further evaluation and monitoring, and then use the prediction of B or C depending on the value of 
prediction made by A. 

### Disadvantages
* Having a big zoo of models in production, all making the inference and storing the results in the database can eat
a lot of hardware resources. It is up to you to decide whether running some old model is still worth it.
* Old models require old features and old datasources to be working. If the underlying data table or database is going
to be eliminated due to some architectural change of the portal, the datasorce would just stop working and the model
will be forcefully removed from the run.

# Feature Engineering Patterns
Performance of many classic ML algorithms depends very strongly on how exactly the input features represent their 
values. It is naive to believe that it would be possible to just read the fields from the database, send them to some 
ML algorithm, give it a huge hardware box, and let it crunch. The resulting model performance (accuracy, recall,
precision etc) will be mostly very poor. 

Proper feature engineering helps the ML algorithms to extract most signal from the noise.

Learning, what approaches are working and what not when trying to convert the features to be more useful for some ML
algorithm, is an important part of being an experienced ML engineer. To make things worse, these approaches are also
depend on the algorithm being using (RandomForest, Bayes, Deep Learning). 
 
Below you can find some typical feature representations, which we had tried. You might want to try them out and check 
whether they are helpful in your situation.

## Simple Value
The simplest feature is to return the value of some field itself, for example the living area in square meters. Some
classifiers are sensitive to outliers, so be sure to remove them. The best way is either to remove samples with 
outliers, or to replace outliers with some value indicating "missing". We've found that using -1 for missing values 
works good enough in many cases. Do not use 0 for missing value.

Note that the Simple Value pattern works best if it represents points in some continuous euclidean space. For example,
numeric zip codes or numeric country codes do not represent points in space (difference between two zip codes
is not a distance) and are unsuitable.

Another consideration is to remove unwanted variability from the value. For example, real estate prices depend on 
many factors, but one of the strongest factors is the living area. If you suppose that the notion of "luxurious" or
"simple" apartments is a good signal for your classifier, then using the square meter price
is better than using the price itself, because the high absolute price does not indicate a luxury estate (it could
be just big one).

You should also consider temporal changes in the value. Speaking of real estate prices, we have up to 40% rise per year.
If your model is running in production for months and years, the thresholds and factors it had learned on the old prices
could be outdated quickly and the model's performance will sink. To avoid this issue, you could try to use the
Clustered Mean Removal pattern instead of the Simple Value.  

## Clustered Mean Removal
Most of ML algorithms work better if the features are normalized (i.e. their values are converted into the range of
0 to 1 or -1 to 1). Normalization will be performed for all samples of the dataset. 

But if the samples are different enough from each other, bringing them to the common range could remove some useful
signal. For example, if we have real estates from different geographical locations in the data set, and the prices
vary up to 10x depending on location, then most of the normalized prices would be situated around 0.1 and around 0.9, 
with very little samples in the middle. if the purpose of the model should not depend on location, such a feature would
send a wrong signal.

Instead, we can cluster the samples first, trying to get a possibly homogene clusters, and then calculate
cluster means and remove the mean of the corresponding cluster from each sample value.

Clustering can be both performed with ML algorithms, as well as with some simple rules (in the example above, it is 
enough to cluster real estate by zip code, estate type and living area). 

Using this pattern can also help to solve the temporal changes in the data (see example in the Simple Value pattern).

Besides, we can introduce the knowledge about average (expected) values into the model, even if this information 
cannot be inferred from the dataset itself (for example, if the dataset has too litte samples, but we have 
pre-calculated average values from some other statistical data source).

We have both tried to substract the mean value from the feature, or divide by it. If you cannot decide, whether 
substracting or diving is better, you can use the All-Combination pattern.

## All Combination
Sometimes it is hard to say, what version of the same feature would work better for your model. Should you use the
absolute price, or price relative to mean, or divide the price by living area? Should your feature count frequencies
of some events in the last day, or last 7 days, or last month?

We've found out that it could be helpful to include ALL versions of the feature into the model. After training the 
model, feature importances can be evaluated and the features that don't contribute much can be removed. But often 
it is the case that all of the versions have similar importances and they are all used equally well by the model. 
In this case, just leave them all in the model. Usually, the computational overhead is not too high to generate all 
of them.

## Frequency Features
Sometimes it makes sense to count some events, given the data stored in the sample. For example, we can count, how
often a real estate with given living area has been listed in the same zip code in the past - and return this count as
the feature value for that sample. We can use historized offers stored in the Clickhouse database to calculate that. 

Be sure to limit the time span when calculating the count, otherwise it would always grow and your model will age 
quicker than expected. 

Another thing to care about when calculating counts is considering the traffic changes with the time: if the company 
spends in one year more on advertisements than in another year, or some law changes shift the market and lead to 
increase or decrease of customer streams, the counts will change too. If the models task is not related to reacting 
on market shifts, it is better to eliminate this noise, for example by dividing the counts by the overall counts - 
in the above example, you would count apartments for rent in some zip code with the living area between 60 and 80, 
listed during the last year, and then divide it by the overall count of all apartments for rent listed in that zip 
code last year. If you also need to eliminate the geographic signal, you can alternatively use the count of all 
apartments for rent from the last year as the divisor.

Frequencies are our preferred way to make features that are very hard to represent in some other way - see 
Representing Geography for some examples.

Another typical usage of this pattern is training a sub-model making a prediction for some partial aspect of the 
problem, and return its predicted scores (which are also, in a sense, probabilities or frequences). For example, when
predicting real estate price using Random Forest, it is tempting to use information from the real estate images. 
Unfortunately, Random Forest cannot predict on images, so we've created a Tensorflow sub-model taking images and 
returning scores, and use these scores as one of the features for the final Random Forest model.

## Representing Geography
Inventing geography or space-related features is a notoriously hard task. It is not helpful to return zip codes as int
numbers, because the model would try to build some distances between the zip codes or multiply them with some factors,
which obviosly doesn't bring much signal. 

Also, using lattitude and longitude as features didn't bring the expected success, even though they are points 
in an euclidean space. We think, it didn't work, because our prediction target required splitting the space into very 
big number of small "islands" or "enclaves" and most ML algorithms are very limited in this relation (in the simplest 
case they only have one line to divide the samples into classes).

We have also tried one-hot encoding for countries, which also wasn't successful. If we would made 195 columns to 
represent all worlds countries, most of these columns was empty in the training set, bearing no signal. Because Random 
Forest implementation in sklearn does not treat the columns of the one-hot encoded features as a single feature, but 
rather as unrelated columns, it might endup choosing a lot of no-signal columns for some trees, increasing the amount 
of bad trees in the forest and therefore reducing the prediction performance. 

On the other hand, if you only take the countries that are present in the training set, how do you encode the feature, 
if some new country appears during the prediction?

Another aspect of the problem to take care of are the temporal changes. Geography changes continuously: new cities
emerge, some zip codes will be combined or splitted, some new land will be cleared for development, so that new lat/long
coordinates appear that have never been used before, etc. Encoding geography directly could therefore increase the 
aging speed of your model.

So far, the best way we've found to handle the geographic information , is to incorporate it into other features, 
i.e. using it when building the clusters for the Clustered Mean Removal and for the Frequency Features.

## One-Hot Encoding
We often use one-hot encoding for categorical features (for example, whether some EXIF tag is present in the image).
Care should be taken to handle the following cases:
* Some new category appears, or existing categories are renamed, split or combined, after model rollout into production.
* Some categories appear too rarely in the training set, so that the model fails to capture them as a tiny signal.
* If there are many categories that appear too rarely, the performance of Random Forest can diminish, or the training
of a DNN can be slowed down (as it would get batches with many empty columns). 

Usually, we group all the rare categories together into the column "Miscellaneous", and use error logging to indicate
the appearance of unexpected categories during prediction.

## Image augmentation
While some DNN tutorials mention image augmentation as something you **could** do, we **strongly** recommend to always
augment images. This is similar to the feature engineering, because it is our way to help the model to recognize,
which aspects of data should be invariant (i.e. should not influence the resulting prediction).

For example, you should randomly mirror the images for the model recognizing kitchens, because kitchens exist in any
different shape. If the model has to recognize floor plans, you should not mirror the images, because floor plans
contain text and mirrored text is invalid.

Do not just apply all possible augmentations to images. Instead, always consider what the model would learn from them. 
In one our training case, adding only two augmentations had helped to increase the F-score by 20%.

## Eye Check
Before using a new feature for training, unless you are under strong time pressure, you should always print its values 
for several thousand samples, or else print some basic statistics about it (min, max, avg, median, number of unique 
values, histogram, etc). This is also true for images: write a debugging function that would convert the already 
augmented and normalized tensor back to an image and show it.

You will be glad doing this, not only because it gives a lot of intuition about the feature performance, but
also helps to find out implementation errors, that would otherwise never be found, especially if your model has a lot
of features and one new feature more does not add lot of new signal.

# Developing with iwlearn
## Installation and system requirements
iwlearn is only tested for Python 2 (we use Python 2.7). It uses MongoDB as a persistence layer, and Scikit-Learn,
Tensorflow and Keras as ML algorithms. 

```bash
pip install iwlearn
```

To try the tutoral [https://github.com/Immowelt/iwlearn/tree/master/tutorial], you should use your local 
MongoDB instance, which can be installed like this:
```bash
mkdir /tmp/mongo
sudo docker run -d -p 27017:27017 -v /tmp/mongo:/data/db mongo
```

## Tutorial
Throughout this tutorial we will use the following example. We need to predict, whether a user is interested in 
professional relocation services, based on the real estates he has watched.

According to our process pattern, we will:
1. [Define the metric we want to improve](#define-the-metric-we-want-to-improve)
2. [Implement a rule first](#implement-a-rule-first)
    * [Define the sample class](#define-the-sample-class)
    * [Understanding sample structure](#understanding-sample-structure)
    * [Understanding datasources](#understanding-datasources)
    * [Implement sample retrieving](#implement-sample-retrieving)
    * [Implement Features](#implement-features)
    * [Implement the Rule Boilerplate](#implement-the-rule-boilerplate)
    * [Generate a dataset](#generate-a-dataset)
    * [Data-based rule definition](#data-based-rule-definition)
    * [Implement the Rule](#implement-the-rule)
    * [Evaluate Rule Performance](#evaluate-rule-performance)
    * [Bring the Rule Online](#bring-the-rule-online)
3. [Monitor Rule Performance](#monitor-rule-performance)
4. [Replace rules with a model](#replace-rules-with-a-model) 
    * [Train a model](#train-a-model)
    * [Bring the model online](#bring-the-model-online)
5. [File structure](#file-structure)

### Define the metric we want to improve
We are building a binary classifier, with the "positive" class (class 1) meaning the user who really needs a relocation
service.

Our prediction will be used to send an email to the user, suggesting him to use our professional relocation service.
Let's say, we have already sent such emails before, and measured 6% unsubscribes after the campaign: users are known to 
react allergically to unexpected emails. We compare the damage from the unsubscribed user with the possible revenue 
from using our relocation services, and decide that false positives are critical (sending email to a user
who don't need relocation service), while false negatives essentially represent the current state (no email is
sent to the users who need our help) and can be easier tolerated.

Therefore, we choose the precision of our binary classifier as our main metric to watch: we need at least the precision
of 98% and should achieve as high recall as possible. Note that the precision is 100% minus the unsubscribe rate, so 
that our controlling metric is directly related to the business goal.  

### Implement a rule first
We begin working on the problem by implementing some rule of thumb first.

Our first idea is that people who rent expensive apartments or houses might outsource the relocation instead of doing
it themselves with the help of friends, especially for larger apartments with more than 3 rooms.

But before we can start testing this idea, we must consider, how this model or rule will be used in production. There
will be a batch job running each night, retrieving all users who haven't received our email yet, and predicting for 
each of them whether they need our help. Our task is to implement a web service getting user key as input and returning
the prediction as output.

#### Define the sample class
The first step of our service would be to load all data needed for prediction given the user key. In iwlearn, it is
accomplished by defining and creating a sample. 

A sample class must inherit from BaseSample and represents all data needed for prediction, at the same time being a DTO
for storing this sample for the training set. Each sample must define entityid - the ID of the business domain entity
being processed (in our case the user key). Note that because each entity can be predicted several times (for example,
each time user visits our website a new prediction has to be made), entityid is not an unique key for samples. 

We should name the sample carefuly, because it is also the name of the collection in MongoDB, where our training set 
will be stored.

```python
class RelocationUserSample(BaseSample):
    def __init__(self, userkey):
        BaseSample.__init__(self, userkey)
```

#### Understanding sample structure
We can now add some dummy data to our sample to play with it around
```python
userkey = 12345
sample = RelocationUserSample(userkey)
sample['SomeData'] = 'data'
sample.data['OtherData'] = 5    # note that using sample or sample.data is the same. Sample.data exists because
                                # of historical reasons and its usage is not recommended
if 'SomeData' in sample:
    print(sample['SomeData'])

del sample['OtherData']
```
Sample behaves like a simple dict. Historically, we have really just used a dict instance for it. The difference of
the sample compared to a dict is not only the entityid, but also that you can compose new samples from existing 
samples and datasource instances (see below) in a structured and cached way.

When developing our sample, we need to plan its dictionary structure. Usually, it is recommended to have only a limited 
number of keys in the root level. Some of them are created automatically, like "entityid", or "created" (the timestamp
of sample construction). Every time we add some other sample to ours, its name will be added as a root-level key to
our sample, with all its values below. 

Having this clear structure and keeping it in mind helps finding out the data we need in our samples.  

#### Understanding datasources
Having the sample, we next could connect to our databases, read the data, and store it into the sample directly. If 
this is the quickest and robust way with your databases, don't hesitate doing just that. 

We rather prefer using our predined datasource instances for data retrieval. iwlearn comes with data sources for 
MongoDB, MS SQL Server, Couchbase and Clickhouse databases. A data source is a class getting connection string as
constructor parameter and providing methods like ```get_row_as_dict``` or ```get_rows_as_dict``` that you can call with
the query as parameter. The class provides thread-safe connection handling, opening and closing connection as needed,
as well as all usual data format conversion that the python database driver requires (for example, Decimals are 
converted to floats and naive datetime Python instances are converted to UTC format for the Mongo driver).

For the case that your sample is so simple that all its data can be retrieved from a single data source, iwlearn also
comes with base sample classes, one per each data source, so you can directly inherit your sample from a corresponding
base class and literally only define, what query to use to load it. 


#### Implement sample retrieving
Now we can proceed with implementing our ```RelocationUserSample.```  For our sample, we need two pieces of data:
1) The list of IDs of the apartments each user has watched on our website. Let's say we can get it from Couchbase. 
2) Attributes of each apartment given its ID (price, area, location etc). Let's say they are stored in an SQL Server.

We implement the first piece of information as a DataSource and add it to our sample:

```python
class WatchedRealEstatesDataSource(CouchBaseDataSource):
    def __init__(self):
        CouchBaseDataSource.__init__(self, COUCHBASE_CONNECTION_STRING)

    # makeimpl is the method we're always overriding when inheriting from data source or sample
    # we can define any number of parameters. In our case, we need to know, data of which user we have to retrieve
    def makeimpl(self, userkey):
        user_profile = self.get_document(userkey)
        return user_profile['VisitedRealEstates']  # list of estate ids

#...
        
class RelocationUserSample(BaseSample):
    def __init__(self, userkey):
        BaseSample.__init__(self, userkey)
        
    def makeimpl(self):    
        # The following line will make WatchedRealEstatesDataSource and add its result to the self under the key
        # with the same name as the name of the data source, without the "DataSource" suffix.
        
        # Because the data source needs to know, data of which user has to be retrieved, we can pass it here, because
        # all **kwargs parameter passed to self.add will be passed to the makeimpl method of the data source.
        self.add(WatchedRealEstatesDataSource(), userkey=self.entityid)
        
        # Always return the retrieved dict in makeimpl
        return self.data

#...

sample = RelocationUserSample(12345)
sample.make()
print (sample['WatchedRealEstates']) # will print a list of estate ids retrieved from Couchbase
        
```

The second piece of data type of data could also be implemented as a data source. But we think, it is generic enough
to be reused for other models, so it implement it as a sub-Sample:

```python
class RealEstateSample(SQLSample):
    def __init__(self, estateid):
        SQLSample.__init__(self, estateid, SQL_CONNECTION_STRING)

    # note that we don't need any parameters here (and they are not allowed), because a sample should be able
    # to completely retrieve itself only knowning its self.entityid
    def makeimpl(self):
        query = """
        select price, livingarea, rooms, zipcode, estatetype, distributiontype
        from RealEstates
        where estateid = ?
        """ 
        params = (self.entityid, )
        return self.get_row_as_dict(query, params) # dict of fields defined in the select statement
        
#...

class RelocationUserSample(BaseSample):
    def __init__(self, userkey):
        BaseSample.__init__(self, userkey)
        
    def makeimpl(self):    
        # The following line will make WatchedRealEstatesDataSource and add its result to the self under the key
        # with the same name as the name of the data source, without the "DataSource" suffix.
        
        # Because the data source needs to know, data of which user has to be retrieved, we can pass it here, because
        # all **kwargs parameter passed to self.add will be passed to the makeimpl method of the data source.
        
        # note that self.add also returns the retrieved dict(), so we can store it into estate_ids variable and use
        # later in our makeimpl code
        estate_ids = self.add(WatchedRealEstatesDataSource(), userkey=self.entityid)
        
        # now we just iterate the estateids and load the RealEstateSamples 
        self['WatchedRealEstateAttributes'] = []
        for estateid in estate_ids:
            sub_sample = RealEstateSample(estateid)
            try:
                sub_sample.make()
                self.data['WatchedRealEstateAttributes'].append(sub_sample.data)
            except:
                logging.exception('Cannot make sample %s' % estateid)
        
        # Always return the retrieved dict in makeimpl
        return self.data
                
#...         

sample = RelocationUserSample(12345)
sample.make()
print (sample['WatchedRealEstates']) # will print [list of estateids]
print (sample['WatchedRealEstateAttributes']) # will print [ {'price': .., 'livingarea': .., ..}, ...]         

```

Because the sample will be persisted and can be reused for further models having more features, we recommend to 
retrieve more information than actually needed for the current model. In this way, it will be possible to base new 
features on existing data sets without reloading the data from the original data source. It is especially important 
for the mutable data sources, because this will avoid the data contamination problem.

For example, in the ```RealEstateSample``` shown above, we would really prefer ```select *``` rather than selecting a 
predefined set of fields.


#### Implement Features
Next step is to define features we need for our rule. The features use the data stored in the sample and calculate
some value that can be used as input for prediction. The output_shape of the features is not limited so that it is 
possible to define features returning several values, or multidimensional values (images, sound, video). 
 
To test our idea, we will calculate medians of living area and rooms and the percentage of houses for rent, across 
all real estates the user has watched. Detecting expensive real estates is a little trickier, because we must compare 
with the average price in the same zip code.

When creating a feature, you need to implement the get method. You can add any top-level keys of your sample as
parameters of the method (lowercasing the first letter), and you will get the corresponding value passed to the method.

For example, if you have this kind of sample:
```python
sample = MySample()
sample['SomeData'] = 123
sample['OtherData'] = {'some property': 'some value'}
sample['SomeDataForOtherFeatures'] = ...
```

then you can develop a feature like this

```python
class MyFeature(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = ()

    def get(self, someData, otherData):
        print (someData)  # 123
        print (otherData) # dict {'some property': 'some value'}
        
        if otherData['some property']:
            return someData * 100
        else:
            return BaseFeature.MISSING_VALUE
```

Now we are ready to implement our features: 

```python
import numpy as np

from iwlearn import BaseFeature

from pricingengine import get_price

class LivingAreaMedian(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)

    def get(self, watchedRealEstateAttributes):
        if len(watchedRealEstateAttributes) == 0:
            return BaseFeature.MissingValue
        return np.median([estate['livingarea'] for estate in watchedRealEstateAttributes])

class RoomMedian(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)

    def get(self, watchedRealEstateAttributes):
        if len(watchedRealEstateAttributes) == 0:
            return BaseFeature.MissingValue
        return np.median([estate['rooms'] for estate in watchedRealEstateAttributes])

class PercentageHousesForRent(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)

    def get(self, watchedRealEstateAttributes):
        if len(watchedRealEstateAttributes) == 0:
            return BaseFeature.MissingValue
        return np.sum([1 for estate in watchedRealEstateAttributes
                       if estate['estatetype'] == 'HOUSE' and estate['distributiontype'] == 'RENT']) / \
               len(watchedRealEstateAttributes)

class ExpensivePrice(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)

    def get(self, watchedRealEstateAttributes):
        prices = []
        for estate in watchedRealEstateAttributes:
            if estate['distributiontype'] != 'RENT':
                continue
            hisprice = estate['price']
            meanprice = get_price(estate['zipcode'], estate['livingarea'], estate['estatetype'],
                                  estate['distributiontype'])
            prices.append((hisprice - meanprice) / meanprice)
        if len(prices) == 0:
            return BaseFeature.MissingValue

        return np.median(prices)
```

#### Implement the Rule Boilerplate 
Before we can start defining a rule in a data-based way, we need to prepare some boilerplate code. The rule must be
implemented as a class inheriting from BaseRule. It must define a list of features it is using in its constructor, and 
implement the method _implement_rule, which recieves the values of the features as its parameters (similarly like a 
feature is receiving the data from sample in its parameters). This method must return None, if the rule cannot make
any prediction, or a some prediction, which is an instance of RulePrediction class, or our own class inheriting from it.

```python
from iwlearn import BaseRule
from iwlearn.rules import RulePrediction

import features as ff

class RelocationRule(BaseRule):
    def __init__(self):
        BaseRule.__init__(self,
                          [
                              ff.LivingAreaMedian(),
                              ff.RoomMedian(),
                              ff.PercentageHousesForRent(),
                              ff.ExpensivePrice()])
        self.sampletype = RelocationUserSample                             
                              
    def _implement_rule(self, livingAreaMedian, roomMedian, percentageHousesForRent, expensivePrice):
        return None
```

#### Generate a dataset
We are now all set and done to see how the values of our features are distributed in the real historical data. To do
this, we will generate a data set. 

Usually, our data sets are generated automatically, because each prediction would add a sample in our samples database. 
But in the beginning, we don't have anything making predictions, so either we would create a dummy code that would just 
make and store a sample, put it in production and wait for some time until the dataset is created in a natural way, or 
we can retrieve some data from other historical data sources and fake the pre-existing data set. We call this faking 
process bootstrapping.  

To bootstrap our dataset, we need to create samples and store them in our samples database (MongoDb). The clean way
to do it looks like this:

```python
import iwlearn.mongo as mongo

from tutorial.common.samples import RelocationUserSample

mongo.setmongouri('mongodb://localhost:27017/')   # for this tutorial, we use a DUMMY local Mongo instance

# For this tutorial, we fake our data. In a real project,
# load here your real keys and labels directly from the databases
DUMMY_userkeys = ['user' + str(x) for x in range(0, 10000)]
DUMMY_userlabels = [1 if x < MAGIC else 0 for x in range(0, 10000)]

for userkey, label in zip(DUMMY_userkeys, DUMMY_userlabels):
    sample = RelocationUserSample(userkey=userkey)
    sample['RelocationLabel'] = label
    sample.make()
    mongo.insert_sample(sample)
```
Note that in the code above, each sample will be made independently, so in total 10000 separate calls to Couchbase and 
10000 separate calls to SQL Server will be made. Usually, this would be prohibitevily slow. We can hack it by using
multi_get requests for Couchbase and a better select statement for the SQL Server, for example like this:

```python
import iwlearn.mongo as mongo

from tutorial.common.samples import RelocationUserSample

mongo.setmongouri('mongodb://localhost:27017/')   # for this tutorial, we use a DUMMY local Mongo instance

recentestates = sqlconnection.get_rows_as_dict("""
        select estateid, price, livingarea, rooms, zipcode, estatetype, distributiontype
        from RealEstates
        where created > getdate() - 30
        """)
estatedata = dict()
for row in recentestates:
    estatedata[row.estateid] = row
    
userkey = load_user_keys() # for example similarly as above
userlabels = load_user_labels() # for example similarly as above

couchbasedocs = couchbaseconnection.multi_get(userkey)
couchbasedata = dict()
for doc in couchbasedocs:
    couchbasedata[doc.userkey] = doc

for userkey, label in zip(DUMMY_userkeys, DUMMY_userlabels):
    sample = RelocationUserSample(userkey=userkey)
    sample['RelocationLabel'] = label
    sample['WatchedRealEstates'] = couchbasedata[userkey]
    sample['WatchedRealEstatesAttributes'] = [estatedata[estateid] for estateid in couchbasedata[userkey]]
    mongo.insert_sample(sample)
```

This would run quicker, but actually it is a hack, because our sample data is generated not by the exactly the same
code as it will be in production. If our sql or couchbase commands in the script are different with those we have 
implemented in the data source, the samples made in production will have a different data structure than samples made 
during the bootstrapping - in the worst case we wouldn't even detect that. So use this approach very carefully and 
always prefer making samples individually, better even in production, if you really can afford it.

Having the samples now, we can't finally generate our dataset:
```python
    mongo.setmongouri('mongodb://localhost:27017/')
    DataSet.generate('train', RelocationRule(), filter={'entityid': {'$regex': r'^user[0-9]*?[0-7]$'}})
    DataSet.generate('test', RelocationRule(), filter={'entityid': {'$regex': r'^user[0-9]*?[8-9]$'}})
```
This will produce two subdirectories input/train and input/test, containing local files with the data from our samples. 
We have splited our sampels already into the train and test sets, so that we can avoid overfitting when defining our 
rule. In the filter parameter of generate we can pass any statement that pymongo.find accepts. The samples loaded from 
MongoDb will be then passed to the features defined in the RelocationRule, and the feature values will be persisted into
local files. 

This is to support model hyperparameter experimenting. If you change anything in your rule or model algorithm or its
hyperparameters, you can re-train the models or re-check the rule, without needing to retrieve and generate the dataset
again. Only when you need to change or extent features, the dataset will need to be regenerated. 

#### Data-based rule definition 
We can now load the dataset and plot the histograms:
```python
    train = DataSet('train')
    print ('Number of samples %d' % len(train))

    train.plot_data()   
```
This will display 4 histograms, one for each rule. Every histogram will show two distributions of the corresponding
value - one for the class 0 and another for the class 1. After looking at the distrubutions, it will be possible
to determine the thresholds for our rule visually: just draw a vertical line separating the distributions of the
two classes in an optimal way. So now we can finish implementing the rule: 

```python
    def _implement_rule(self, livingAreaMedian, roomMedian, percentageHousesForRent, expensivePrice):
        if livingAreaMedian >= 75 and roomMedian >= 3 and \
            percentageHousesForRent > 0.3 and expensivePrice > 1.0:
            return RulePrediction(1)    # class 1 means "positive"
        return RulePrediction(0)
```

#### Evaluate Rule Performance
We can evaluate rule performance before going live by using our test set. 

```python
    test = DataSet('test')
    rule = RelocationRule()
    rule.evaluate_and_print(test)
```
This will print something like
```
Evaluation of RelocationRule(RelocationRule)
--------------------------------------------------------------------------------
Number of test samples	2000
confusion_matrix	[[1800    0]
 [  86  114]]
f1_score	0.7261146496815287
precision_score	1.0
recall_score	0.57
```

#### Bring the Rule Online
Recall of 57% with Precision 100% satisfies our needs, so we can now take this rule into production:

```python
import cherrypy

import iwlearn.mongo as mongo

from tutorial.common.rules import RelocationRule
from tutorial.common.samples import RelocationUserSample

class RelocationService():
    def __init__(self):
        self.rule = RelocationRule()
        self.collection = mongo.mongoclient()['Tutorial']['Predictions']

    @cherrypy.expose
    def predict(self, userkey):
        sample = RelocationUserSample(userkey=userkey)

        if sample.make() is not None:
            mongo.insert_sample(sample)

            prediction = self.rule.predict(sample)
            mongo.insert_check(prediction, sample)     # save prediction for rule monitoring and evaluation
            pre_dict = prediction.create_dictionary()  # this will serialize the prediction to a way suitable
                                                       # for storing in MongoDB or transmitting over the wire
            return pre_dict
        return 'cannot make sample'
```

### Monitor Rule Performance
The predictions of the rule will be saved into the Predictions collection. Each prediction has a reference to the 
sample document stored in the collection RelocationUserSamples. If we now implement an additional code that would track
which users have reacted on our email campaign, find out the corresponding samples and store the label there, we would
have the following advantages: 
* Our labeled data set will grow in size allowing for better training and evaluation.
* We can re-tune the rule thresoholds after some time to prevent its aging.
* Last but not least, we can evaluate the real performance of our rule in production, and eventually decide to replace
the rule with a model.

The future versions of iwlearn will introduce some functionality helping to label samples, and to monitor and 
to visualize real performance.

### Replace rules with a model
After having run one or several rules in production, sometimes there is an intuition that a ML model would achieve a
better performance, for example due to much more features or combination of unrelated features.

Another situation when you want to train the model is when a rule was infeasible from the beginning on (for example
image classification).
 
#### Train a model
If you jumped to this chapter directly, please read all the chapters from 
[Define the sample class](#define-the-sample-class) to [Generate data set](#generate-data-set), because you will also
need them for the model. The chapters will left you with the Sample and Features defined and the test and train datasets
generated. The next step is to define and train the model. 
iwlearn comes with two base models, using Scikit-Learn and Keras respectively as their engines. The best practice is
to inherit your model from one of them, or directly from the BaseModel:

```python
import tutorial.common.features as ff
from tutorial.common.samples import RelocationUserSample

class RelocationModel(ScikitLearnModel):
    def __init__(self):
        ScikitLearnModel.__init__(self,
                                  name = 'RelocationModel',
                                  features = [
                                    ff.LivingAreaMedian(),
                                    ff.RoomMedian(),
                                    ff.PercentageHousesForRent(),
                                    ff.ExpensivePrice()],
                                  sampletype=RelocationUserSample,
                                  labelkey='RelocationLabel')
```
Even though such inheritance seems to be redundant in this particular, it makes possible to override methods for 
training, creating labels, dealing with scores, defining neuronal network architecture for Keras-based models, etc 
so that in our experience it is always almost needed for one reason or another. If you don't need it in your project,
you can instantiate ScikitLearnModel directly.

Having the model, we can train it with
```python
train = DataSet('train')
test = DataSet('test')
print ('Samples in train %d, in test %d' % (len(train), len(test)))

model = RelocationModel()
model.train(train)
model.evaluate_and_print(test)
```

This will evaluate to something like
```
Evaluation of RelocationModel(RelocationModel)
--------------------------------------------------------------------------------
01. RoomMedian Importance	0.5087875046910646
02. LivingAreaMedian Importance	0.22190641952636728
03. ExpensivePrice Importance	0.2133446725970341
04. PercentageHousesForRent Importance	0.05596140318553421
Number of test samples	2000
f1_score	0.9558441558441559
precision_score	0.9945945945945946
recall_score	0.92
``` 
Just like we have expected, an ML model performs better: with the similar precision it has a much higher recall. So we
take it online.

#### Bring the model online
Currently we would pickle the model, save it to disk and then load and use it in production. We're considering options 
to integrate a model version control system.

To use the model instead of the rule in our service, we just replace the rule with the model:
```python
import cherrypy

import iwlearn.mongo as mongo

from tutorial.common.rules import RelocationRule
from tutorial.common.samples import RelocationUserSample
from tutorial.common.models import RelocationModel

class RelocationService():
    def __init__(self):
        self.model = RelocationModel.load('/path_to_model')
        self.collection = mongo.mongoclient()['Tutorial']['Predictions']

    @cherrypy.expose
    def predict(self, userkey):
        sample = RelocationUserSample(userkey=userkey)

        if sample.make() is not None:
            mongo.insert_sample(sample)

            prediction = self.model.predict(sample)
            mongo.insert_check(prediction, sample)     # save prediction for rule monitoring and evaluation
            pre_dict = prediction.create_dictionary()  # this will serialize the prediction to a way suitable
                                                       # for storing in MongoDB or transmitting over the wire
            return pre_dict
        return 'cannot make sample'
```


### File structure
We recommend to use similar project files structure like in the tutorial:
```
src \
    common \                # Content of the files corresponds to the name 
        datasources.py  
        samples.py
        features.py
        models.py
        rules.py
    serving \
        # All components related to serving predictions in production
    training \
        template \
            10_xxx.py - 19_xxx.py # Scripts related to bootstrapping samples, storing them in Mongo, updating them with 
                                  # additional information, generating datasets, etc
            20_xxx.py - 29_xxx.py # Scripts related to visualizing data, understanding its distribution, finding outlier, etc
            30_xxx.py - 30_xxx.py # Scripts related to evaluating and trying out different rules
            40_xxx.py - 40_xxx.py # Scripts related to trying out and evaluating ML models.
        experiment_one \
            # You copy the template folder every time you want to add features to the model, create new model etc, and
            # change the scripts directly here in place.
        experiment_two \
        experiment_three \
```

## Advanced Tutorial
The advanced tutorial builds up on the [Tutorial](#tutorial) and demonstrates the following topics:
1. [Using Keras-based Models](#using-keras-based-models)
2. [Implementing Regressor Models](#implementing-regressor-models)
3. [Advanced ScikitLearnModels](#advanced-scikitlearnmodels)
    - [Feature selection](#feature-selection)
    - [Learning curve](#learning-curve)
    - [Validation curves](#validation-curves)
    - [Hyperparameter selection](#hyperparameter-selection)   

You can find the source code here
[https://github.com/Immowelt/iwlearn/tree/master/tutorial]

### Using Keras-based Models
Keras-Tutorial is being developed. Inherit your models from iwlearn.models.BaseKerasClassifierModel or 
BaseKerasRegressorModel and define the method _createkerasmodel(), where you need to define the keras model 
architecture, compile the model and set it to self.kerasmodel. 

### Implementing Regressor Models
Tutorial is being developed.

### Advanced ScikitLearnModels   
While model training shown in the basic tutorial could be enough for some not very challenging tasks, usually it is
not so easy to train models. In a more realistic, professional setting, we might want to start with the feature
selection, plotting learning and validation curves to determine required dataset size and model architecture, as 
well as to fine-tune hyper-parameter to achieve a little better model performance.

#### Feature selection
To show the feature selection in the tutorial, we define two more features that are intentionally less useful than the
rest of the features:

```python
class RoomMedianNoisy(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = ()

    def get(self, watchedRealEstateAttributes):
        return np.median([estate['rooms'] for estate in watchedRealEstateAttributes]) + 100 * random.random()

class PureNoise(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = ()

    def get(self, watchedRealEstateAttributes):
        return random.random()
```

we now define a new model using these features and generate new train and test datasets. Then, we can perform the
automated feature selection procedure: 

```python
train = DataSet('train')
test = DataSet('test')
print ('Samples in train %d, in test %d' % (len(train), len(test)))

model = RelocationModel()
model.train(train)
model.evaluate_and_print(test)

scored_features = model.feature_selection(test, step=1, n_splits=4)

selected_features = []
for i, (feature, score) in enumerate(zip(model.features, scored_features)):
    logging.info('%s %s %s ' % (i, feature.name, score))
    if score <= 1:
        selected_features.append(feature)
```

This will output something like
```
Evaluation of RelocationModel(RelocationModelPro)
--------------------------------------------------------------------------------
01. LivingAreaMedian Importance	0.3737243851030765
02. ExpensivePrice Importance	0.3511122606354373
03. PercentageHousesForRent Importance	0.14942548830713584
04. RoomMedianNoisy Importance	0.06586973954153408
05. PureNoise Importance	0.059868126412816185
Number of test samples	2000
f1_score	0.8518518518518519
precision_score	0.9044943820224719
recall_score	0.805
INFO:root:Recursive Feature Elimination CV
INFO:root:Optimale Anzahl an Features: 3
INFO:root:Optimale Feature: [1 2 3 1 1]
INFO:root:0 LivingAreaMedian 1 
INFO:root:1 RoomMedianNoisy 2
INFO:root:2 PureNoise 3
INFO:root:3 PercentageHousesForRent 1 
INFO:root:4 ExpensivePrice 1 
```
Note that our noisy features have got the worst importances, and the feature selection has ranked all features. The
rank of 1 is the best, the worse the feature is, the higher its rank is.

#### Learning curve
Learning curve helps detecting under- and overfitting by calculating model performance for differeren dataset sizes.
(http://mlwiki.org/index.php/Learning_Curves). We can plot it for our model:
```python
    train = DataSet('train')
    test = DataSet('test')
    print ('Samples in train %d, in test %d' % (len(train), len(test)))

    model = RelocationModel()
    model.train(train)

    plt = model.plot_learning_curve(train)
    plt.show()
```
The learning curve uses the accuracy by default to score the performance of the model. You can pass another score 
using the scoring parameter. 

We use the learning curve to understand whether we could benefit from adding more samples into the dataset, as well
as to determine the minimal data set size used for crossvalidation.

#### Validation curves
Validation curves use the same idea as the learning curve, but they variate other hyperparameters instead of the 
dataset size, for example, if we use RandomForestClassifier, we can change number of estimators.

```python
train = DataSet('train')
test = DataSet('test')
print ('Samples in train %d, in test %d' % (len(train), len(test)))

model = RelocationModel()
model.train(train)

logging.info('Validation Curve n_estimators')
n_estimators_range = np.linspace(start=30, stop=110, num=int((110 - 30) / 3), dtype=int)
logging.info(len(n_estimators_range))
plt = model.plot_validation_curve(dataset=train, param_name="n_estimators", param_range=n_estimators_range)
plt.show()

logging.info('Validation Curve max_features')
max_features_range = np.linspace(start=1, stop=4, dtype=int)
logging.info(len(max_features_range))
plt = model.plot_validation_curve(dataset=train, param_name="max_features", param_range=max_features_range)
plt.show()

logging.info('Validation Curve min_samples_leaf')
min_samples_leaf_range = np.linspace(start=1, stop=6, num=6, dtype=int)
logging.info(len(min_samples_leaf_range))
plt = model.plot_validation_curve(dataset=train, param_name="min_samples_leaf", param_range=min_samples_leaf_range)
plt.show()
```

#### Hyperparameter selection   
ScikitLearnModels provide the method train_with_hyperparameters_optimization performing grid search of hyperparameters
and ensuring the most optimal model training compared with the simple train method using default hyperparameters. To
demonstrate the hyperparameter optimization, we define an additional feature pumping a lot of noise, so that more 
than default number of estimators is required to get good performance.

```python
train = DataSet('train-hyper')
test = DataSet('test-hyper')
print ('Samples in train %d, in test %d' % (len(train), len(test)))

model = RelocationModelHyper()

# Train the model in a simple way to provide a baseline of the model performance
model.train(train)
model.evaluate_and_print(test)

# Now perform training with the hyperparameter optimization
n_estimators_range = np.linspace(start=100, stop=600, num=5, dtype=int)
max_features_range = np.linspace(start=1, stop=7, num=3, dtype=int)
min_samples_leaf_range = np.linspace(start=1, stop=4, num=3, dtype=int)

folds = 5 # use TrainingsSizeCrossValidation.xlsx and learningCurve to number of folds right
param_distributions = {"n_estimators": n_estimators_range, "max_features": max_features_range, "min_samples_leaf": min_samples_leaf_range}
configuration = {'n_splits': folds, 'param_distributions': param_distributions, 'params_distributions_test_proportion': 0.2, 'test_size': 0.25}
model = RelocationModelHyper()
model.train_with_hyperparameters_optimization(dataset=train, **configuration)
model.evaluate_and_print(test)
```
This will output something like
```
f1_score	0.7988505747126436
max_features	auto
min_samples_leaf	1
n_estimators	100
precision_score	0.9391891891891891
recall_score	0.695
...
f1_score	0.8081395348837209
max_features	10
min_samples_leaf	1
n_estimators	600
precision_score	0.9652777777777778
recall_score	0.695
```


## Architectural concepts
TBD

## Reference
Read the source code: [https://github.com/Immowelt/iwlearn]
