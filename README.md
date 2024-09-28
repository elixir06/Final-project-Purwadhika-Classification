# BarracudaGroup_DTI_01_FinalProject 
This folder contains the Final Project for Purwadhika DTI-DS Progran related to the Bank Marketing Campaign


By team Barracuda:
1. Muhamad Farikhin
2. Wahyu Bornok Augus Sinurat
3. Christian Manan


## **1. Business Problem Understanding**

### **Context** 

Given the data obtained from 2008 to 2010, which coincides with the European Financial Crisis, several Eurozone countries experienced significant economic impacts. This crisis, following the global financial crisis of 2007-2008, led to severe difficulties in countries such as Greece, Portugal, Spain, Ireland, and Italy.

As a result of the crisis, banks became more cautious about lending, leading to credit restrictions. In the most affected countries, banks faced increased competition for client deposits as a stable source of funding. To address these challenges, banks implemented various marketing strategies, including direct marketing campaigns. These campaigns involve contacting selected customers through channels such as personal contact, mobile phone, mail, or email to promote new products or services, particularly deposit products.

However, not all customers are interested in making deposits, potentially leading to wasted marketing resources. This scenario presents an opportunity for machine learning applications. Banks can develop predictive models to identify which customers are most likely to respond positively to deposit offers, thereby optimizing their marketing efforts and reducing unnecessary expenditure.

### **Problem Statements**

Direct marketing, while effective, comes with inherent drawbacks—chief among them the potential to trigger negative attitudes towards banks due to perceived intrusions on privacy (Page and Luding, 2003).

Banks face a dual challenge: on one hand, they seek to maximize deposits; on the other hand, making unsolicited calls to clients without understanding their interest levels can lead to annoyance. This not only risks alienating clients but also undermines the bank's objective of maintaining strong, long-term relationships with them.

To address this, the marketing team aims to enhance the effectiveness of their direct marketing efforts. Their goal is to identify and target clients who are more likely to be interested in making deposits and are willing to invest more for the opportunity. By focusing on these high-potential clients, the team hopes to improve the quality of their outreach and strengthen client relationships, all while achieving the bank's financial objectives.

### **Objectives**

Here, we position ourselves as data scientists at a consulting company that helps our stakeholder, the Marketing Manager, solve the aforementioned issues.

Given the negative impact of making ineffective calls, our goal is to develop a machine learning model that predicts the likelihood of a client making a deposit. This model will assist the marketing manager in making informed decisions about which clients to target. Additionally, the model will identify the key features that influence a client’s interest in making a deposit.

With this knowledge, the marketing team will be empowered to conduct more effective and efficient campaigns—maximizing the number of calls that result in deposits while minimizing those that do not. This approach will ultimately lead to increased revenue and a stronger relationship between the bank and its clients.

### **Analytic Approach** 
We will analyze deposit patterns across various features using previously recorded data, which includes labels indicating whether a client made a deposit. This analysis will help us understand the factors influencing deposit behavior.

Based on our findings, we will build a model that accurately classifies clients according to their likelihood of making a deposit. This model will assist in predicting and targeting potential depositors more effectively.

### **Evaluation Metric** 

Since our main focus is on customers who make deposits, we set our target as follows:

**Target:**
- 0: no (does not make a deposit)
- 1: yes (makes a deposit)

**Type 1 error**: False Positive (customers who actually don't deposit but are predicted to deposit)
Consequence: Loss from telemarketing costs, customer inconvenience

**Type 2 error**: False Negative (customers who actually deposit but are predicted not to deposit)
Consequence: Loss of potential revenue from potential customers


In this project, we will use ROC AUC (Receiver Operating Characteristic Area Under the Curve) as the primary metric for evaluation. We chose ROC AUC over other metrics due to its versatility and robustness. It provides a comprehensive outlook of the model's performance by measuring its ability to separate positive and negative instances, thus giving a summary of overall model effectiveness.

ROC AUC is particularly well-suited for our case because it is robust when dealing with imbalanced datasets. As we will see later, our dataset has a significant imbalance between positive and negative instances. Additionally, ROC AUC evaluates the model's performance across all possible classification thresholds, which is useful when the optimal threshold is not known in advance.

The interpretability of ROC AUC and its ability to allow fair comparisons between different models further justify our choice. The AUC value provides an easily interpretable measure of discriminative power, where 0.5 indicates random guessing and 1.0 indicates perfect separation.

Using ROC AUC as our primary metric, our model evaluation will consist of two phases:

1. **Model Selection**: We will determine the best model by selecting the one with the highest ROC AUC score. This approach ensures we choose the model with the best overall discriminative power across all possible thresholds.

2. **Threshold Optimization**: Once the best model is selected, we will determine the optimal classification threshold to maximize performance in terms of revenue. This step involves finding the balance between identifying potential depositors (True Positives) and minimizing wasted marketing efforts (False Positives).

To optimize our model's performance in terms of revenue, we'll use the following assumed business variables for our cost analysis. Note that these are hypothetical figures used to demonstrate our methodology:

#### **Cost Analysis**:

**Campaign Cost per call**:
- Call Cost: 2 Euro
- Salary & Infrastructure Cost: 5 Euro

**True Positive Income / False Negative Cost**:
- Average deposit term: 2000 Euro
- Deposit revenue: 50% of average deposit term = 1000 Euro 
(Assumed figure of banks generate revenue by issuing loans and charging interest)
- Income = Deposit revenue = 1000 

TP Income / False Negative Cost = Income - Campaign Cost = 1000 - 7 = 993

**False Positive Loss**:
- Customer churn probability due to annoyance: 1.2%
- Customer Lifetime Value: 10,000 Euro
- Customer Annoyance cost: 1.2% * 10,000 = 120 Euro
- FP loss = Customer Annoyance Cost = 120 Euro
- FP Cost = FP loss + Campaign Cost = 120 + 7 = 127 Euro

**Revenue**
- Visible Revenue = TP Income = Income - Campaign Cost = 1000 - 7 = 993 Euro
- Actual Revenue = TP Income - FP Cost = 993 - 127 = 866 Euro

We distinguish between Visible Revenue (considering only campaign costs) and Actual Revenue (including potential customer loss due to marketing annoyance). Visible Revenue can be misleading as it overlooks hidden costs like customer churn. We prefer Actual Revenue for its more comprehensive view of business impact.

Our calculations use approximations, given the difficulty in obtaining exact figures for some variables. To address this, we've created flexible metrics adaptable to different values. This approach allows for solutions that can be tailored to specific business contexts.

While based on estimates, our methodology remains valuable and can be adjusted with more precise data, ensuring relevance in real-world scenarios.


## **Data Undestanding**

The dataset originates from https://archive.ics.uci.edu/ml/datasets/bank+marketing and represents the results of a marketing campaign conducted by a bank in Portugal over the period of May 2008 to November 2010. The campaign involved direct contact via telephone to offer customers a term deposit product.

Each row in the dataset corresponds to information about an individual customer, including their socio-economic status from the previous marketing efforts. It is important to note that the dataset contains missing values, which are labeled as 'unknown'.

The purpose of this dataset is to provide insights into customer behavior and preferences when it comes to banking products like term deposits. This information can help the bank's marketing team better target and optimize their outreach strategies to maximize conversions and minimize unnecessary costs.


### **Attribute Information** 
**Demographic attributes of bank clients:**

| # | Attribute | Data Type | Description |
| --- | --- | --- | --- | 
| 1 | age | Integer | age of client |
| 2 | job | Text | type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown") |
| 3 | marital | Text | marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed) |
| 4 | education | Text | level of education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown") |
| 5 | default |Text | has credit in default? (categorical: "no","yes","unknown") |
| 6 | housing |Text | has housing loan? (categorical: "no","yes","unknown") |
| 7 | loan |Text | has personal loan? (categorical: "no","yes","unknown") |

**Information related with the last contact of the current campaign:**

| # | Attribute | Data Type | Description |
| --- | --- | --- | --- | 
| 8 | contact | Text | contact communication type (categorical: "cellular","telephone") |
| 9 | month | Text | last contact month of year (categorical: "jan", "feb", "mar", …, "nov", "dec") |
| 10 | day_of_week | Text | last contact day of the week (categorical: "mon","tue","wed","thu","fri") |
| 11 | duration | Integer | last contact duration, in seconds. Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no") |

Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

**Other attributes:**
| # | Attribute | Data Type | Description |
| --- | --- | --- | --- | 
| 12 | campaign | Integer | number of contacts performed during this campaign and for this client (numeric, includes last contact) |
| 13 | pdays | Integer | pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted) |
| 14 | previous | Integer | previous: number of contacts performed before this campaign and for this client |
| 15 | poutcome | Text | poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success") 

**Social and economic context attributes:**
| # | Attribute | Data Type | Description |
| --- | --- | --- | --- | 
| 16 | emp.var.rate | Float | employment variation rate - quarterly indicator |
| 17 | cons.price.idx | Float | consumer price index - monthly indicator |
| 18 | cons.conf.idx | Float | consumer confidence index - monthly indicator |
| 19 | euribor3m | Float | euribor 3 month rate - daily indicator |
| 20 | nr.employed | Float | number of employees - quarterly indicator |


**Ouput variable (desired target)**
| # | Attribute | Data Type | Description |
| --- | --- | --- | --- | 
| 21 | y | Text | has the client subscribed a term deposit? (binary: 'yes','no') |


## **4. Exploratory Data Analysis (EDA)**

This section will explore the data through visualizations and address the following:
- What is the distribution of deposits ("yes" compared to "no")?
- Which clients have the highest probabilities of accepting a deposit given certain characteristics?

Further analysis can be found in the accompanying notebook.

## **5. Feature Engineering**

Preprocessing steps performed before modeling:
- Data Imputation
- Data Encoding and Normalization

## **6. Methodology (Modeling/Analysis)**

Further analysis can be found in the accompanying notebook.

## **7. Conclusion & Recommendation**

**Conclusion**
In the dataset , we found several features that proved to stand out for having a significant deposit conversion rate :
- Poutcome (Success) = If the outcome of the previous campaign was successful, there is a higher chance compared to other outcomes that the person will deposit
- Job (Student) = If the occupation is student , there is a higher chance compared to other occupation that the person will deposit

While our black box model considers these features to have the most significant impact on its decision making :
- nr.employed = if the value is low , the model will be more inclined to predict deposit
- Euribor3M = if the value is low , the model will be more inclined to predict depost

Through our process, we identified Gradient Boosting Machine with Random Over Sampling as the best algorithm for this project, achieving an AUC score of 0.804.

The optimal threshold for prediction is 0.47, meaning that if the predicted probability is 0.47 or higher, the model will classify the client as likely to make a deposit.

The model demonstrates a false positive rate of 16.54% and a solid true positive rate of 65%.

To simplify, when tested on the data, our model successfully identified two-thirds of all clients who made a deposit, while contacting less than a quarter of all clients on the list.

**Recommendation**
Here are some recommendations to maximize business performance :
- ⁠Try to prioritize people whose outcome was successful on the previous campaign since they have the highest conversion rate.
- Also try to focus on those whose occupation is student and retired since they have high conversion rate.
- ⁠Focus on university degree since they have high chance of accepting the offer and high numbers,
- ⁠People whose age above 60 also have high chance of accepting the offer.
- ⁠If the Euribor3M value is below 1.2 , the offer will generally be accepted. beware of that momentum, as people will more likely to deposit
- ⁠During October, September, December and March, people have tendency to accept the offer. Try to increase the calling effort and increase reach during that time.
- ⁠People who don't experience credit card default generally have 2 times higher chance of accepting the offer than those who do. Try focusing on the first than the latter.
- ⁠Generally if the number of employees is below 5100 they have higher chance of accepting the offer. so beware of them.

Possible improvements :
- Future projects could benefit from deeper feature engineering, such as interaction terms or polynomial features, to capture more complex relationships.
- Other algorithms like CatBoost, or deep learning models could be explored to see if they offer superior performance.
- While the project includes a basic cost analysis, future projects could expand this to include a full cost-benefit analysis considering long-term customer retention and potential up-selling or cross-selling opportunities