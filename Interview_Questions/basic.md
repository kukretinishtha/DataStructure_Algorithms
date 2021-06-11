1. What are the types of Artificial Intelligence Learning Models
    Knowledge-Based Classification
    Inductive Learning: This type of AI learning model is based on inferring a general rule from datasets of input-output pairs.
    Deductive Learning: This type of AI learning technique starts with a series of rules and infers new rules that are more efficient in the context of a specific AI algorithm.

    Feedback-Based Classification
    Based on the feedback characteristics, AI learning models can be classified as supervised, unsupervised, semi-supervised or reinforced.

    —  Unsupervised Learning: Unsupervised models focus on learning a pattern in the input data without any external feedback. Clustering is a classic example of unsupervised learning models.

    —  Supervised Learning: Supervised learning models use external feedback to learning functions that map inputs to output observations. In those models the external environment acts as a “teacher” of the AI algorithms.

    —  Semi-supervised Learning: Semi-supervised learning is a class of supervised learning tasks and techniques that also make use of unlabeled data for training – typically a small amount of labeled data with a large amount of unlabeled data. The goal of a semi-supervised model is to classify some of the unlabeled data using the labeled information set.

    —  Reinforcement Learning: Reinforcement learning models use opposite dynamics such as rewards and punishment to “reinforce” different types of knowledge. This type of learning technique is becoming really popular in modern AI solutions.
    Data Mining Vs Machine Learning




2. Difference between Machine Learning and Data Mining?
    Machine learning focuses on prediction, based on known properties learned from the training data.
    Data mining focuses on the discovery of (previously) unknown properties in the data. This is the analysis step of Knowledge Discovery in Databases.




3. How to tackle Machine Learning Project?
    1. Specify business objective
    2. Define problem.
    3. Create a common sense baseline. But before you resort to ML, set up a baseline to solve the problem as if you know zero data science. You may be amazed at how effective this baseline is. It can be as simple as recommending the top N popular items or other rule-based logic. This baseline can also server as a good benchmark for ML algorithms.
    4. Review ML literatures
    Set up a single-number metric. The metric has to align with high-level goals, 
    5. exploratory data analysis (EDA). 
    6. Preprocess. This would include data integration, cleaning, transformation, reduction, discretization and more.
    7. Feature Engineering
    8. Ensemble
    9. Deploy models into production for inference.
    10. Monitor model performance, and collect feedbacks.
    11. Iterate the previous steps. 




4. Parametric vs Nonparametric ?
    A learning model that summarizes data with a set of parameters of fixed size (independent of the number of training examples) is called a parametric model.
    A learning model where the number of parameters is not determined prior to training. On the contrary, nonparametric models (can) become more and more complex with an increasing amount of data.




5. Discriminative vs Generative Learning Algorithm ?
    Discriminative algorithms model p(y|x; w), that is, given the dataset and learned parameter, what is the probability of y belonging to a specific class. A discriminative algorithm doesn't care about how the data was generated, it simply categorizes a given example
    Ex: Linear Regression, Logistic Regression, Support Vector Machines etc.

    Generative algorithms model p(x|y), that is, the distribution of features given that it belongs to a certain class. A generative algorithm models how the data was generated.
    Ex: Naive Bayes, Hidden Markov Models etc.

    Given a training set, an algorithm like logistic regression or the perceptron algorithm (basically) tries to find a straight line—that is, a decision boundary—that separates the elephants and dogs. Then, to classify a new animal as either an elephant or a dog, it checks on which side of the decision boundary it falls, and makes its prediction accordingly.

    First, looking at elephants, we can build a model of what elephants look like. Then, looking at dogs, we can build a separate model of what dogs look like. Finally, to classify a new animal, we can match the new animal against the elephant model, and match it against the dog model, to see whether the new animal looks more like the elephants or more like the dogs we had seen in the training set.





6. What is cross validation ?
    Cross Validation is a technique to evaluate predictive models by partitioning the original sample into a training set to train the model, and a validation set to evaluate it. For ex: K fold CV divides the data into k folds, train on each k-1 folds and evaluate it on remaining 1 fold. The result of k models can be averaged to get a overall model performance.

    Time - Series Cross Validation : Use forward chaining strategy





7. What is overfitting?
    Overfitting or High Variance is a modeling error which is caused by a hypothesis function that fits the training data too close but does not generalise well to predict new data.
    




8. What is regularization?
    Regulariztion is a technique to prevent overfitting by penalizing the coefficients of the cost function.
    




9. Ridge Regression
    It performs ‘L2 regularization’, i.e. adds penalty equivalent to square of the magnitude of coefficients. Thus, it optimises the following:

    Objective = RSS + α * (sum of square of coefficients)





10. Lasso Regression
    LASSO stands for Least Absolute Shrinkage and Selection Operator.Lasso regression performs L1 regularization, i.e. it adds a factor of sum of absolute value of coefficients in the optimisation objective.

    Objective = RSS + α * (sum abs (coefficients))

11. Elastic nets
    A technique known as Elastic Nets, which is a combination of Lasso and Ridge regression is used to tackle the limitations of both Ridge and Lasso Regression.





12. Loss Functions for Regression and Classification?
    Regression Loss Function
        Square or l2 loss (not robust)
        Absolute or Laplace loss (not differentiable)
        Huber Loss (robust and differentiable)
    Classification Loss Function
        SVM/Hinge loss
        log loss





13. How do you handle missing or corrupted data in a dataset?

    Reason for why data goes missing.
    Missing at Random (MAR): Missing at random means that the propensity for a data point to be missing is not related to the missing data, but it is related to some of the observed data.

    Missing Completely at Random (MCAR): The fact that a certain value is missing has nothing to do with its hypothetical value and with the values of other variables.

    Missing not at Random (MNAR): Two possible reasons are that the missing value depends on the hypothetical value (e.g. People with high salaries generally do not want to reveal their incomes in surveys) or missing value is dependent on some other variable’s value (e.g. Let’s assume that females generally don’t want to reveal their ages! Here the missing value in age variable is impacted by gender variable).

    We can hando the following:
    * Mean, Median and Mode Imputation
    * Multiple Imputation
    * KNN (K Nearest Neighbors)





14. How would you handle an imbalanced dataset?
    1. Using a better metrics like AUROC, Precision, Recall etc.
    2. Cost-sensitive Learning
    3. Over sampling of the minority class or Under sampling of the majority class.
    4. SMOTE (Synthetic Minority Over-sampling Technique.)
    5. Anomaly Detection





15. how do you detect outliers?
    Outliers are extreme values that deviate from other observations on data , they may indicate a variability in a measurement, experimental errors or a novelty.

    Causes of outliers on a data set:
    1. Data entry errors (human errors)
    2. Measurement errors (instrument errors)
    3. Experimental errors (data extraction or experiment planning/executing errors)
    4. Intentional (dummy outliers made to test detection methods)
    5. Data processing errors (data manipulation or data set unintended mutations)
    6. Sampling errors (extracting or mixing data from wrong or various sources)

    Some of the most popular methods for outlier detection are:

    1. Extreme Value Analysis: 
    Determine the statistical tails of the underlying distribution of the data. For example, statistical methods like the z-scores on univariate data.

    2. Probabilistic and Statistical Models: 
    Determine unlikely instances from a probabilistic model of the data. For example, gaussian mixture models optimized using expectation-maximization.

    3. Linear Models: Projection methods that model the data into lower dimensions using linear correlations. For example, principle component analysis and data with large residual errors may be outliers.

    4. Proximity-based Models: Data instances that are isolated from the mass of the data as determined by cluster, density or nearest neighbor analysis.

    5. Information Theoretic Models: Outliers are detected as data instances that increase the complexity (minimum code length) of the dataset.

    6. High-Dimensional Outlier Detection: Methods that search subspaces for outliers give the breakdown of distance based measures in higher dimensions (curse of dimensionality).

