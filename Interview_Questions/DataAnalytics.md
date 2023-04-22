1. What are the steps in Data Analysis?
    * Data Collection
    * Data Storage
    * Data Cleaning
    * Data Analysis
    * Visualization
    * Decision

2. What are diffrent types of anlytics?
    * Descriptive Analysis
        It describes what has happened in the past.It describes the basic features of data in-study.<br/>
        ex-How many simcard were sold in the past.<br/>
    
    * Prescriptive Analysis
        It uses optimization to identify the best alternatives to minimize or maximize some objective.<br/>
        ex-What should be pricing or advertising to maximize the simcard revenue.<br/>
    
    * Predictive Analysis
        It describes what will happen in the future.<br/>
        ex-Which user will switch the sim. <br/>

3. Why to visualize the data?
    * Comprehend Information quickly<br/>
    * Identify relationship and updates<br/>
    * Pinpoint emeging trends<br/>
    * Communicate the story to Otherwise<br/>

4. What is exploratory data anlysis?
    It is the process of performing initial investigations on data. <br/>
    It is required for :<br/>
    * To discover patterns
    * To spot anomalies patterns
    * To check assumptions about

5. What are the steps for data preparationa nd explorations?
    * Variable Identifications
    * Univariate analysis
    * Bi-Variate Analysis
    * Missing Values Treatment
    * Outlier Treatment
    * Variable Transormation
    * Variable Creation


6. Descriptive Statistical Measurement in Detail
    #### Distribution 
        It is the summary of individual variable frequency <br/>

    #### The Central Tendancy
        It tells about the value w.r.t centre of data <br/>
    ##### Mean, Medain, Mode
        ** When to use mean Imputation -  When the data is Uniformly distributed and is numerical <br/>
        ** When to use Median Imputation - When the data is skewed and does not follow uniform distribution and is numerical. Median is less sensitive to ouliers. <br/>
        ** When to use Mode - When the data is categorical/ Numerical with small number of unique values <br/>

    #### Dispersion
        It tells about the degree of variation in the data <br/>
    ##### Range, Variance, Interquantile range, Standard derviation
        ** Range - Tells you the differnece between max and min value in the data
        ** Variance - Tells you how much data is varying from the mean
        ** Interquantile Range - Tells you about the mid spread (only 50% of the midle data), difference between Q1 and Q3
        ** Standard Deviation - how the data is spread across the mean (ideally +-2SD)
    ##### Measures, Shape and Association
        ** Skewness - Measures the degree of symmetry
        ** Covariance - How the two variables are differnt from each other. range(-inf, +inf)
        ** Correlation - How the two variables are related to each other. range(-1 to +1)

8. Different types of Distribution in detail
    #### Bernoulli Distribution
        * It tells there will be only two events Pass/Failure in a single trial
        * Probability mass function = p^x * (1-p)^(1-x) where x -> (0,1)

    #### Uniform Distribution
        * It tells there will be n number of outcomes and all have equal probability of occurance.
        * Density Function = 1/(b-a)

    #### Binomial Distribution
        * It tells there will be only two events Pass/Failure in a n number of trial.
        * Function = (nCx)(p^x)(q^(n-x))
    
    #### Normal Distribution
        * The mean, median and mode of the distribution coincide
        * The curve of the distribution is bell shaped
        * The total area under the curve is 1
    
    #### Poission Distribution
        * Events occurs at random interval of time and space

    #### Exponential Distribution
        * it is widely used for survival analysis

9. Different types of sampling
    * Random - Rndomly select from population
    * Stratified - Samples are collected from each suset of population
    * Cluster - Random sampling of clusters after analysing
    * Multistage - Mutistage cluster sampling
    * Systematic - Every 10th example is selected from population

10. Differnt types of sampling errors
    * Poulation specific error - Reseachers does not identify the correct population to survey on
    * Selection error - Self select their ineterst of participation
    * Sample frame error - Wrong population is used

11. Hypothesis Testing
    #### One Sample t-test
        * Checks whether the sample means differ from population means
        ** Null Hypothesis - Mean of sample are same as that of population
        ** Alternate Hypothesis - Mean of sample differ from population
        When to reject - if the t-statistic tile differ from population mean

    #### Two Sample t-test
        * Check whether the two samples mean differ from each other
        ** Null Hypothesis - Both the groups are same
        ** Alternate Hypothesis - The mean of two independent data sets are differnt

    #### Paired t-test
        * Check the difference between two independent samples from different times
        ** Null hypothesis - Both groups are same
        ** Alternate Hypothesis - The mean of two independent data sets are differnt

12. Type I error <br/>
    - False positive <br/>

13. Type II eeror <br/>
    - False Negative <br/>

14. Chi-Squared test for independence
    - used to find relationship between two items or idependence of two features <br/>
    Null Hypothesis - There is a relationship between variables <br/>
    Alternative Hypothesis - There is no relationship between variables <br/>

15. Chi-Squared test for goodness of Fit
    - Used with categorical data to check whether the samples belongs to same distribution <br/>
    Null Hypothesis - The two dustibution are same <br/>
    Alternative Hypothesis - The two distribution are different <br/>

16. ANOVA - Analysis of variance test
    - Used to test multiple groups at the same time as conducting multiple t-test on each faeture group may results in false positives <br/>
    - It uses f-distribution <br/>
    Null Hypothesis - The distribution are same <br/>
    Alternative Hypothesis - The two distribution are different <br/>

17. Z-Score
    It tells how much data point is close to mean <br/>
    Its works very well with the 1D data <br/>
    Does not work with-  <br/>
        * N-D Data
        * Multimodel Data
        * Non Gaussian Data

18. SMOTE - Synthetic Minority Oversampling Techniue
    Creates data in the same dimension direction randomly but with different point <br/>

19. Rejection Sampling
    - If the datapoint does not follow any distribution <br/>
    - Sample point from proposal distribution and then accept and reject accordingly <br/>

20. Autoregressive Model
    -It assumes that the future values are dependent on the past and present values <br/>
    ex- A financial stock investor using an autoregressive model at that time would have had good reason to believe that prices in that sector would stay stable or rise for the predictable future. <br/>
    - Adavntages: <br/>
        * Autocorrelation can tell about the randomness in the data
        * Capable of forecasting for recurring patterns
        * Possible to predict outcomes with less data
    - Disadvanatages: <br/>
        * Autocorrelation coefficient >= 0.5 for better prediction
        * If it is significantly affected by soxial factors then don't use
    
    #### Python Code
    from statsmodels.tsa.ar_model import AutoReg <br/>

21. Moving Average
    - Models the next step in the sequence as a liniear function of the residual errors. <br/>
    #### Python Code
    df.to_frame().rolling(window_size).mean() <br/>
    ##### OR
    from statsmodels.tsa.arima.model import ARIMA <br/>

22. ARIMA  - Auto Regressive Integrated Moving Average
    - Models the next step in the sequence as a linear function of the differenced observations and residual errors at prior time steps. <br/>
    #### Python Code
    from statsmodels.tsa.arima.model import ARIMA <br/>

23. SARIMA - Seasonal Auto Regressive Regressive Integrated  moving Averge
    - models the next step in the sequence as a linear function of the differenced observations, errors, differenced seasonal observations, and seasonal errors at prior time steps. <br/>
    #### Python Code
    from statsmodels.tsa.statespace.sarimax import SARIMAX <br/>

24. VAR - Vector Auto Regression
    - Applying time series to multiple parallel time series <br/>
    - The method is suitable for multivariate time series without trend and seasonal components <br/>
    #### Python Code
    from statsmodels.tsa.vector_ar.var_model import VAR <br/>

25. VARMA - Vector Auto Regression Moving Average
    - ARMA to multiple parallel time series, e.g. multivariate time series <br/>
    #### Python Code
    from statsmodels.tsa.statespace.varmax import VARMAX <br/>

26. VARMAX - Vector Autoregression Moving-Average with Exogenous Regressors 
    - VARMA + the modeling of exogenous variables <br/>
    #### Python Code
    from statsmodels.tsa.statespace.varmax import VARMAX <br/>

27. SES - Simple Exponential Smoothing
    - models the next time step as an exponentially weighted linear function of observations at prior time steps. <br/>
    #### Python Code
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing <br/>

28. HWES - Holt Winter’s Exponential Smoothing
    - Triple Exponential Smoothing method models the next time step as an exponentially weighted linear function of observations at prior time steps, taking trends and seasonality into account. <br/>
    #### Python Code
    from statsmodels.tsa.holtwinters import ExponentialSmoothing <br/>