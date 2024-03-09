# Heart Disease Prediction | COGS 118B Project

Samantha Prestrelski, Denny Yoo, Jeffrey Yang, Yash Pakti, Fayaz Shaik


## Abstract 
We aim to predict which people do and do not have heart disease based on 18 factors. 

The data comes from annual telephone surveys conducted by the CDC as part of the Behavioral Risk Factor Surveillance System (BRFSS). Each row represents a different person. Each of the 19 columns represents a single health factor that may play a part an individual having heart disease. These columns are HeartDisease, BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer. Some of these columns can be converted to binary representation, the others will need to be one-hot encoded. 

We will be creating unsupervised machine learning models that use the data in order to predict which people do and don't have heart disease. We will be using models such as K-Means and GMM.

We will measure success by using the columns HadHeartAttack, HadAngina, and HadStroke as true indicators for having heart disease. 


## Background

Heart disease affects millions of people each year and is a leading cause of mortality worldwide. The early prediction and diagnosis of heart disease is crucial for early intervention, improved treatment, and better patient outcomes. It is thus critical that medical practitioners have access to tools that would grant them the ability to make such early detections of heart disease in patients, a challenge that our project seeks to address.

As our model utilizes various health factors in its prediction of heart disease, it is crucial to understand how these factors influence the probability of an individual having heart disease. Extensive prior research exists which meticulously analyzes how factors such as BMI and age contribute to the likelihood of heart disease, illuminating the complex relationships that our model seeks to capture. A 2021 study by the American Heart Association found that obesity, as quantified by BMI, is directly responsible for a range of cardiovascular risk factors such as diabetes and hypertension which heavily contribute to an increased likelihood of heart disease <a name="cite_ref-1"></a>[<sup>1</sup>](#cite_note-1). A WebMD article corroborates that individuals past the age of 65 are drastically more susceptible to heart failure and other conditions that are linked to heart disease <a name="cite_ref-2"></a>[<sup>2</sup>](#cite_note-2). Due to their high impact nature, it is clearly imperative that BMI and age, along with the other 15 health factors, be integrated into our model should we want to comprehensively predict heart disease in patients.

Past research has also shown machine learning models to be excellent predictors of medical conditions like heart disease. A 2023 study found various neural network models to be capable of achieving up to 94.78% accuracy in heart disease prediction <a name="cite_ref-3"></a>[<sup>3</sup>](#cite_note-3). Due to the past success of these models, it becomes increasingly clear that machine learning algorithms are best fitted towards solving our problem of predicting heart disease. 

1. <a name="cite_note-1"></a>[](#cite_ref-1) Powell-Wiley TM, Poirier P, Burke LE, Després J-P, Gordon-Larsen P, Lavie CJ, Lear SA, Ndumele CE, Neeland IJ, Sanders P, St-Onge M-P; on behalf of the American Heart Association Council on Lifestyle and Cardiometabolic Health; Council on Cardiovascular and Stroke Nursing; Council on Clinical Cardiology; Council on Epidemiology and Prevention; and Stroke Council. Obesity and cardiovascular disease: a scientific statement from the American Heart Association. Circulation. 2021;143:e984–e1010. doi: 10.1161/CIR.0000000000000973
2. <a name="cite_note-2"></a>[](#cite_ref-2) “What to Know about Your Heart as You Age.” WebMD, WebMD, www.webmd.com/healthy-aging/what-happens-to-your-heart-as-you-age. Accessed 16 Feb. 2024. 
3. <a name="cite_note-3"></a>[](#cite_ref-3) Srinivasan, S., Gunasekaran, S., Mathivanan, S.K. et al. An active learning machine technique based prediction of cardiovascular heart disease from UCI-repository database. Sci Rep 13, 13588 (2023). https://doi.org/10.1038/s41598-023-40717-1

## Problem Statement

As described earlier, heart disease is a huge problem, and according to our dataset, affects a huge population. Fortunately, heart disease is treatable and chances of it being treated are far better if prevented early on. Our problem we hope to solve is that, given our dataset filled with various factors relating to a patient, can we build a ML model that can accurately predict a person's chance of getting heart disease? Our dataset features a variety of patients' information including their BMI, gender, and medical history, which we believe can be used to generate patterns in the form of quantitative metrics such as "there are X patients who have heart disease" and "X% of patients who are between the ages of 50-60 have heart disease."

The problem is quantifiable as it expresses the risk of heart disease as a percentage or a singular yes/no to express whether that person should be worries about heart disease if they continue their current lifestyle. This model's performance can be measured by how accurately it tracks a person's actual risk of heart disease. This can be achieved by splitting the dataset into three: training, validation, and testing, with the training dataset larger than the validation and testing. Finally, this problem is replicable as our dataset and other ML training methods can be shared and used by everyone / other datasets to reproduce results. Our models can also be retrained later with updated data.

We can use ML training methods such as logistic regression to train a predictive model on our dataset, or one half of the dataset coined as the training dataset, to do testing on the other half of the dataset for validation purposes. We could also run several feature detection tasks to ensure that the most predominant features are used to predict heart disease.

## Data

We will use 17 variables to predict Heart Disease and a binary value for whether or not the person has heart disease.  

Link to data: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease  
- 18 variables
    - HeartDisease, BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer.
- 320,000 observations
    - Each observation is a U.S. resident that provided their health status as part of the Behavioral Risk Factor Surveillance System (BRFSS)'s telephone surveys.

According to the [CDC](https://www.cdc.gov/heartdisease/risk_factors.htm), 
> About half of all Americans (47%) have at least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking.

Although high blood pressure and high cholesterol are hard to measure, diabetes and obesity are indicators of high blood pressure, making `BMI`, `Diabetic`, and `PhysicalActivity` critical variables, as well as `Smoking`. BMI is numerical, Diabetic is Yes/No/Other, PhysicalActivity is Yes/No, and Smoking is Yes/No. PhysicalActivity is not very well described, so we will need to check how this dataset was [created](https://github.com/kamilpytlak/data-science-projects/blob/main/heart-disease-prediction/2022/documentation/vars_list_with_descriptions.txt) and match up the raw data with the processed data to figure out how to interpret some of the variables. 

We will need to convert binary columns that are listed as Yes/No to 0s and 1s, and category columns like `General Health` to some sort of numerical in order to vectorize all of the data. 

## Proposed Solution

We will compare the performance of different techniques of unsupervised learning methods learned in class. This includes methods such as K-means, Gaussian Mixture Models, Hierarchical clustering, and spectral clustering. Since there are 17 predicting variables for a binary value, we'll likely have to do some sort of dimensionality reduction such as Principal Component Analysis to make the model more manageable.  

Although traits like high blood pressure, high cholesterol, and smoking are known to be risk factors for heart disease, we are unsure of their predictive value compared to other traits in the dataset, since there is no simple equation for heart disease. Thus, unsupervised learning methods can help us see if there's any trends in the data of people with similar traits.  

We will split our data into a training, validation, and test set to evaluate the performance of our models. Each model will be trained on the training set, then evaluated on the validation set for initial results. Once we've tuned the hyperparameters for each of the individual models, we will do a final evaluation against the test set. We will compare our methods to a baseline [logistic regression model](https://share.streamlit.io/kamilpytlak/heart-condition-checker/main/app.py) developed by the creator of the dataset.  

We will do all work in Jupyter notebooks using publicly available models and datasets, as well as provide links to any resources used so that our work is reproducible. 

## Evaluation Metrics

One evaluation metric that can be used is distortion score. Distortion score takes into account only how tight a cluster is by calculating the average of the squared distances between each point in a cluster and the cluster center. It is represented as such: 
$$
J = \sum_{n=1}^N\sum_{k=1}^Kr_{nk} ||x_n-\mu_k||^2
$$

Another possible evaluation metric to be used is silhouette score which is similar to distortion score except it also takes into account the distances between the points of one cluster and the nearest cluster center. The function for silhouette score can be generalized as such:

$$
\frac{separation - cohesion}{max(separation, cohesion)}
$$

These metrics can be used to determine the optimal number of clusters for our data as certain factors could lead to specific kinds of heart disease, or we can see if there is great overlap between them with a smaller number of clusters. Once we decide on the specific models to use, we will look more deeply into other evaluation metrics.

## Ethics & Privacy
Using the provided data science ethics checklist from https://deon.drivendata.org, we discuss the following potential concerns with ethics and data privacy:
Data Collection
- **Informed Consent**: The human subjects opted in, as they could have refused the telemarketing survey, hung up the phone at any time, or refused to answer questions. 
- **Collection Bias/Bias Mitigation**: Some bias is towards people that are willing to give their information, which might be affected by age or location. It also restricts the survey to those that have access to phone services. This dataset might also be affected by access to healthcare: certain diagnoses like diabetes, heart disease, and kidney disease might be missed for lower-income people that don't have the resources to get diagnosed. While we cannot fix the collection process, we will need to do exploratory data analysis to see what demographics are represented and if there are any specific groups are over- or underrepresented. 
- **Limit PII exposure**: Health information is inherently personally identifiable. However, the dataset has been cleaned to only include Sex, Age, and Race as the most PII. We can do research into whether these factors are very important in predicting heart disease, or if there's negligible difference. If there are not significant differences, we can anonymize this dataset further. 

Data Storage
- **Data retention plan**: This dataset is public and managed by someone else. However, in the testing phase, we should not store any results of people who test our model if they input their own information. 

Deployment
- **Monitoring and Evaluation**: If this ML project were to go into production, we would not collect user data. Any computations would be done on the client side, meaning we have no access to any of the user inputs and thus cannot store them.
- **Redress**: To prevent unintentional user harm, we would also put a warning that this model is not a recommendation from medical professionals and is purely based on data. If this were to be in production for a while, we could update our model based on yearly new releases from the CDC's BRFSS. We can also provide a feedback form for any user complaints. 

As we continue to work with the data and develop our model/metrics, we will revisit the data ethics checklist to ensure we address potential ethical concerns.
