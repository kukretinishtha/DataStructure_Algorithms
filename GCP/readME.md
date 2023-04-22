Which metric is:

Did the boy miss any wolves?
recall
Which metric is:

Did the boy cry wolf too often?
precision
Relu
-example of an activation function
-used between hidden layers
-allows you to cap negative x-coordinates to be 0 rather than negative y-values
3 ways to fix class imbalance
-upsampling (SMOTE)
-downsampling
-weighted classes (give more attention to minority class)
When to use TPUs?
-large batches
-sharded data
-large models

use tf.data

DNN on tf.keras
When to use GPUs?
-need lots of parallelization
-lots of math ops
when to use tf.data?
-if dataset can't fit in memory
-if you need preprocessing
-need access to different hardware/batches
what is prefetching?
allows for more efficient use of CPUs w/ an accelerator.

specifically, if the CPU is preparing batch 2 of data, lets GPU/TPU simultaneously train batch 1 such that it is not idle time while waiting for batches
Embeddings
-allow for lower dimensionality representation of feature crosses to help w/ sparsity
Clipping
-fix for outliers
-handles extreme outliers by setting outlier to be = max value

ex: if housing data shows a house w/ 500 rooms, instead adjust the 500 = max of dataset (such as 10)
When to use tf.transform
during preprocessing
phases of tf.transform
1. analysis
-doing during training
-for numeric might be searching for min/max over whole dataset
-for categorical might be searching for all unique values
-uses beam

2. transform
-done during prediction
-scales individual input by min/max for numeric; might be changing to one-hot encoding for categorical
-uses tensorflow
what happens if you do feature engineering during "feature creation" phase with beam?
-training/serving skew
-you will need to run the same pipeline at prediction time to compute the same aggregations
what happens if you do feature engineering during "feature creation" with BQML?
-training/serving skew

-you can fix this by adding TRANSFORM clause and putting all SELECT logic inside of it
-this bakes it into the prediction graph
why would you use tf.keras.layers.lambda?
to help with training/serving skew
what happens if you do feature engineering during "train model"?
-this would mean using tf
-this is not helpful if you need to compute averages or any aggregations over multiple inputs
ai platform training local CLI
gcloud ai-platform local train --module-name /
trainer.task --package-path trainer / 
--train-files $TRAIN_DATA --eval-files $EVAL_DATA /
--job-dir $OUTPUT_DIR
ai platform training steps
1. train locally
2. upload to gcs
3. submit to ai platform training to run on cloud
what happens under-the-hood for ai platform training
bayesian optimization for hyperparameter tuning
if you want to do your own hyperparameter tuning in ai platform training how do you do it?
include --config flag & a config.yaml. use trainingInput in yaml and then specify maxTrials, enableEarlyStopping, metric, etc.
ai platform training - cloud job submit CLI
gcloud ai-platform jobs submit training $JOB_NAME /
--job-dir $OUTPUT_PATH --runtime-version 1.13 /
--module-name trainer.task --package-path trainer /
--region $REGION --train-files $TRAIN_DATA /
--eval-files $EVAL_DATA --num-epochs 1000 /
--learning-rate 0.01
ai platform training CLI flag for standard distributed training
--scale-tier

BASIC_GPU, BASIC_TPU
ai platform training CLI flag for custom machine types?
--scale-tier CUSTOM

then add in parameters as flags (--master-machine-type n1-highcpu-16, etc.)

or config.yaml
ai platform training default worker configuration
--scale-tier BASIC

single worker node
if you don't know how to code and want to submit ai platform training job, what should you do?
use UI and choose "prebuilt algorithms"

ex: linear linear, xgboost, wide and deep, object detection, image classificaiton, etc.
ai platform prediction - batch - frameworks available?
only tf
ai platform prediction - online - which frameworks available?
tf, xgboost, scikit-learn, etc.
how to send prediction input to ai platform prediction?
gcloud ai-platform predict --model $NAME --version \
$VERSION --json-instances='data.txt'

where data.txt is newline-delineated JSON
which frameworks use explainability in ai platform prediction?
tf
CLI to use explainability in prediction call?
gcloud beta ai-platform versions create $VERSION \
--model $NAME --explanation-method \
'integrated-gradients'

gcloud beta ai-platform explain --model $NAME \
--version $VERSION --json-instances='data.txt'
integrated gradients: type of data
tabular, low-resolution images (such as x-rays), text
sampled shapley: type of data
tabular
xrai: type of data
images
differential models: type of model & explainability framework
neural nets

can use integrated gradients or xrai
non-differential models: type of model & explainability framework
xgboost, decision trees

sampled shapley
steps to train a custom model
1. develop tf model/code
2. create dockerfile with model code
3. build the image
4. upload the image to GCR
5. start training job
ai platform CLI for creating job with custom model
gcloud ai-platform jobs submit training my-job \
--region $REGION --master-image-uri \
gcr.io/my-project/my-repo:my-image --lr=0.01
parameterserver worker strategy
-asynchronous distributed training

-some machines are workers and some are parameter servers
-workers calculate gradients; parameter server updates the weights and passes that to the workers
when to use parameterserver worker strategy? (3 reasons)
-low latency
-want to continue if a machine crashes (such as using preemptible machines)
-machines all have different performance
central storage distributed training & when to use?
-synchronous

-1 machine/worker that is attached to multiple GPUs
-each GPU calculates gradients and sends to CPU of machine. CPU updates weights and sends to GPUs to calculate gradients

-good for large embeddings that don't fit on single GPU
mirror strategy
-synchronous

-one machine attached to multiple GPUs/TPUs
-each GPU/TPU has a copy of the model
-each machine shares its weights with the other machines
-all weights are then aggregated together (usually mean)
-requires good connection between GPU/TPUs
multi-worker mirror strategy
-synchronous

-multiple machines each with multiple GPUs
synchronous distributed training
all workers train over different slices of input data in sync, and aggregating gradients at each step
asynchronous distributed training
all workers are independently training over the input data and updating variables asynchronously
main difference between automated ML pipeline and full CI/CD pipelining
automatically deploying the model via Cloud Build triggers vs. manually deploying new version
did the system miss anything?
recall
of the things that the system predicted, how correct was it?
precision
in continuous training, if you find you have data drift, what should be done?
retrain model only
in continuous training, if you have model drift, what should be done?
-need to retrain model, redeploy new model
-retrigger whole CI/CD pipeline
TFDV: purpose and parts
-analyze data/validate it is correct

statisticsgen, schema gen, example validator
statisticsgen
-part of TFDV
-visual report/graphical distribution of data

-can detect outliers, anamolies, skews, missing data
what is tf.data?
-library for reading TFRecords as datasets

-lets you significantly reduce latency by enabling prefetching for letting training happen on the accelerator while CPU does transformations (reduces CPU idle time)
TFRecords
-stores data as protobuf instead of bytes
-improves readability
SavedModel
-serialized model artifact
-allows for model-agnostic deployment (CPU/TPU/GPU)
how to optimize offline prediction?
add more machines
how to optimize online prediction?
-scale out with GKE, GAE, CAIP prediction
-make each type of prediction microservice
which performance metric to use if class is balanced and each class is equally important?
accuracy
AUC ROC
-plots TPR vs. FPR
-tells you the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example
-good default
AUC PR
-plots precision vs. recall
-use this if you care more about positive than negative class / dataset is imbalanced
example: need to detect fraud
if you increase the classification threshold, what will happen to precision?
it will increase b/c false positive rate will decrease
if you increase the classification threshold, what will happen to recall?
it will stay the same or decrease b/c true positives will increase or stay the same
Recommendations AI
-configure to set up A/B testing
-integrate with Google Tag Manager to record events (like clicks, etc.)
-integrate with Merchant Center to upload product catalog
what metric do we want to optimize for spam detection?
minimize FP; optimize precision
storage transfer service
-moves data to GCS (from S3, URL, or other GCS bucket)
- data > 1 TB
BQ DTS
used for ingesting Google Ad Data to BQ
Transfer Appliance
physical device
connect from your network to upload to GCS
what is smote?
oversampling the minority group to make classes more balanced
what is the data labeling service?
-provide dataset, instructions, list of labels 
-assigns humans to give labels to data
-part of continuous evaluation strategy 
-assigns ground truth to data
how to enable continuous evaluation w/ ai platform prediction?
-establish ground truth as either yourself or use data labeling service
-must already have a model version deployed
-then you can run a daily evaluation job. this job will compare online prediction results by storing them in BQ and comparing to existing ground truth
-you can then analyze evaluation metrics in console
how does evaluation job work if using data labeling service?
1. data labeling service creates an ai platform dataset w/ all of the new rows in BQ since the last run --> both input/output of model
2. data labeling service sends labeling request on this new data to generate groundtruth
3. data labeling service will calculate evaluation metrics for the day before it ran (so parallel evaluation jobs will always sample day before's data to ensure different samples)
how does evaluation job work if using own groundtruth?
1. data labeling service creates an ai platform dataset w/ all of the new rows in BQ since the last run 
2. you must have already added in groundtruth labels in the column BEFORE the evaluation job runs (evaluation job will skip any rows w/o groundtruth label)
3. data labeling service will then calculate evaluation metrics
recommendations AI - what is rejoining?
-best practice: ensure product catalog is up-to-date and if you are importing catalog while recording events, you will need to rejoin on product ID
-events that can't be associated w/ an ID are not used during training
3 ways to record events in Recommendations AI
-javascript pixel
-API: eventStores.userEvents.write
-google tag manager (creates a trigger that will fire whenever the event occurs)
Recommendations AI - model types
1. others you may like
2. frequently bought together
3. recommended for you
4. recently viewed

each have a default placement & optimization (such as CTR, revenue per order, conversion)
what is "placement" for recommendations AI?
-area on website where to locate the recommendation
what is schemagen?
-part of TFDV

-takes raw data and infers schema
-this is stored as metadata and used later in pipeline to ensure consistency (such as during tf transform)
what is examplevalidator?
-part of TFDV

-validates data/schema to make sure data conforms (such as making sure it is an int, etc.)
-also used by tf.transform to look for training/serving skew since it knows previous shape of data
what happens during transformation phase in ci/cd?
-transform data (such as string -> int, bucketizing, etc.) in dataflow
-important to use tf.transform to reduce training/serving skew
what happens during "trainer" phase in ci/cd?
-produces serialized "SavedModel" to be stored in GCS
-keras/estimator phase
what happens during model evaluation phase in ci/cd?
-use TFMA in dataflow
-can compare two models and see how performance differs
-can also slice data by certain metrics (such as comparing dates or features)
if customer wants model ASAP/cheapest, which should they choose?
BQML
if customer can't move data outside of EDW for compliance what should they choose?
BQML
which products have explainability built-in?
AutoML & AI Platform

*look for keyword "trust"
which distributed training service to use to optimize wall-time?
centralstorage -> each GPU will compute weights w/o waiting for others
AUTO_CLASS_WEIGHTS
-BQML parameter
-used if need to balance the classes
rolling average
-dataprep preproccesing function
-smooths out noise
-preferred over daily min/max
when to use normalization?
if the range of values is really large (such as age, income, city population, etc.)
when to use scaling for normalization?
when range is evenly distributed & you know lower/upper bound
i.e. age
NOT income
when to use clipping for normalization?
if your dataset has extreme outliers
what is principal component analysis?
-used as a way to mask sensitive data
-allows you to create a formula to multiply weights against column values, making it hard to determine the individual
such as 1.5age+0.8smoker
ways to mask data on GCP?
-if text fields, can use DLP
-if usage/video, might be able to use Cloud Vision or Video API

hash/salt anonymizes data
coarse --> bucketize data such as just using first 3 digits of zip code
what-if tool
-use to evaluate fairness 
-simulates changes to examples to see how predictions would alter
-shows confusion matrix
kinds automl edge model
-available for offline prediction on tf lite 

1. high accuracy
2. low latency
3. general purpose
CNN vs. RNN
-CNN is best for non-sequential data, like images

-RNN is best for sequential data like video (sequence of images), and text (for understanding context of words together). used for autocorrect, captions, etc.
data fusion vs. data prep
-data fusion is for creating data pipelines (encrypt/decrypt, ingestion)
-runs on dataproc

-dataprep is for cleaning data (format, filter, new column creation)
-runs on dataflow
when do you specify a machine type in ai platform prediction?
when you create a version
in kfp, what's the best way to execute BQ queries / GCS copy ops?
built-in KFP components
purpose of interleave in tfrecord?
process files concurrently
ai platform prediction JSON format
{"instances": 
[
{"values": [1, 2, 3, 4], "key": 1},
{"values": [5, 6, 7, 8], "key": 2}
]}
scale __ before __. 

what is the best practice?
-up before out
i.e. increase memory to larger GPU before adding more GPUs due to network latency
tpu supported models
tf and pytorch
how to split training/test/validation for timeseries data?
do based on date range, not random splits. 

in most other cases, you can do 80/20 split
keras API vs. estimator API
keras has better support for distributed training
do ai platform built-in algos support distributed training?
no
Quantization
-process to make weights less accurate (cast float to int) to reduce storage
Imagine a linear model with 100 input features, all having values between -1 and 1:
10 are highly informative.
90 are non-informative.
Which type of regularization will produce the smaller model?
L1
transfer learning
Take a general pretrained model on some data and use it own your own model with new data

can be stored on AI Hub
____ the learning rate & ____ number of epochs
decrease learning rate
increase epochs
elements of a good feature
1) related to the objective; 
2) known at prediction time; 
3) definition won't change over time; 
4) numeric with meaningful magnitude (not ordinal but cardinal); 
5) has enough examples; 
6) brings human insights to the problem.
how to handle missing data
provide an additional column that says whether the data is missing in the original column, and only then replace the missing values with the mean/mode.
which model type in recommendations AI optimizes for revenue per order?
frequently bought together
which model type in recommendations AI optimizes for CTR?
-recommended for you
-others you may like
BQ Streaming Row Limit
1MB
BQ Streaming Throughput Limit
100k requests/second/project with insert ID
Dataflow ParDo
take method and run in parallel
best practice for training/test/validation split
-use hash & modulo not RAND

-for timeseries, need to order
-for others, 80/20 split
offline or online?
training or serving?

production recommendation
offline serving
offline or online?
training or serving?

text-to-speech
offline training
offline or online?

spam detection
online training
3 solutions to I/O limits on training?
-TFRecords
-parallelize reads
-reduce batch size
3 solutions to gradient speed on training?
-use accelerators
-upgrade proccessor
-reduce model complexity
3 solutions to OOM?
-add more memory
-reduce model complexity
-reduce batch size
what is data leakage?

2 examples
leaking data from test set into training/validation

such as if duplicate data appears in both training/test or timeseries where future data appears in training set
symptoms of overfitting?
-low training loss, high validation loss
symptoms of underfitting?
high training loss
when to use AUC PR over AUC ROC?
if positive predictions are more important
when to use F1 vs. AUC ROC?
if positive predictions are more important
what is downsampling?
disregard random sample of majority class
which distributed training strategies use 1 machine?
-central storage
-mirrored strategy
which distributed training strategies use > 1 machine?
-parameter worker server
-multiworker
maxTrials
maximum # of models that bayesian optimization should use for computing best hyperparameters
hyperparameterMetricTag
-metric to use when ai platform is picking which model is the best
gcloud command for hyperparameter tuning in ai platform training?
use trainingInput
hyperParameterSpec -- specify metricTag, maxTrials, enableEarlyStopping

then under --parameters include which parameters you want and specify type of variable (numeric, categorical, etc.)
two purposes of explainable AI
1. debugging (can show radiologist penmark as a feature for x-ray)

2. optimizing (figuring out which features are most important)
which explainability methods for non-differential models?
shampled shapley
which explainability method for differential models?
integrated gradients, xrai
if you have 1 machine & 4 GPUs, which strategy?
mirror strategy
if you have 10 machines & 4 GPUs, which strategy?
multiworker
if you care more about minority class / data is imbalanced, which metric should you optimize for?
AUC PR
if you want to use pytorch or caffe, how should you create a training job?
create a custom container in ai platform
when using tf.data, when do you want to prefetch?
after batching to ensure it's not just doing element-by-element
when to use hashing?
-if you have infinite categories (like cats)

need to be sure # of hash buckets > # of inputs (to avoid collisions)
common tf.transform use cases?
-buckets
-quantiles
-ngrams
-z score
-string split/join
when to use softmax?
if you are doing classification with N classes
what is the pearson correlation?
numeric input + numeric output

linear
what is anova?
numeric input + categorical output
what is chi squared?
categorial input + categorial output
feature crosses are usually combined ___ to help with memorization?
embedding layer
type of data to use w/ boosted trees?
structured data
type of data to use with LSTM?
timeseries
ML.WEIGHTS
-allows you to see the weights used by BQML model
if you wanted to look at trained models and find the best one in kfp, what kind of component to use?
lightweight python
how to monitor for data drift?
TFDV
which metric for classification evaluates the percentage of real spam that was recognized correctly?
recall
how many true positives were found?
recall
what to do if gradients vanish?
-use relu
-adam optimizer
what to do if gradients explode? (loss sharply increases)
- batch normalization, grading, clipping
what to do if relu layers die?
decrease learning rate
what loss function to use if multi-class, multilabel?
sigmoid cross entropy
what loss function to use if labels can only belong to 1 class?
softmax cross entropy
what looks at feature's distribution in the training data?
skew
what looks at feature's distribution in the production data?
drift