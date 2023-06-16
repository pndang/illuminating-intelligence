# Illuminating Intelligence - ML
Assessing Predictive Power for Major Power Outages in the U.S.

Data Source: [data](https://engineering.purdue.edu/LASCI/research-data/outages/outagerisks)


My exploratory data analysis on this dataset can be found [here](https://pndang.com/illuminating-cognizance/).


## Framing the Problem

### Problem Identification

**Prediction problem**: Predict the severity of an outage (classes: 1-minor, 2-moderate, 3-major, 4-severe)

- Some available metrics at time of prediction:
    - climate region, anomaly level, climate category, cause category, outage start time, total customers, utility contribution to gross state product (GSP)

**Type**: Multiclass Classification

<b>Response variable</b>: Four severity categories - category name (label): <b>minor (1), moderate (2), major (3), severe (4)</b>
- Severity of an outage is measured using its duration and number of customers affected.
- The thresholds associated with each severity category are determined from the exploratory analysis right after this cell.
    - TLDR: 
        - Duration: <=12 hours (minor), <=48 hours (moderate), <=117 hours (major), over 117 hours (severe)
        - Number of customers affected: <=75k (minor), <=150k (moderate), <=360k (major), over 360k (severe) 
        - **Note**: for every outage, the more severe category of the two measures (if different) is selected
        <br>
        <br>
- **Rationale**: The duration and number of customers affected are relevant metrics to assess the nature and impacts of an outage; making these attributes ideal to measure outage severity when combined. 
    - To illustrate, a long outage that affects a high number of customers has immense economics and customer satisfaction impacts than others, and the ability to predict such an outage could assist officials in charge at coordinating response mechanisms accordingly and on-time.

<b>Evaluation metric</b>: Matthews' correlation coefficient
- **Rationale**: Instead of accuracy or F1-score, the MCC is chosen because, unlike accuracy, it considers all outcomes of classification in its calculation (TP, FP, TN, FN), as well as the quality of each metric across all classes (outage severity), which incorporates the subtle class imbalance present into account. In addition, unlike in the biomedical or similar unique contexts, a distinct emphasis on either recall or precision is not necessary. Hence, the MCC is fitting as an encompassing metric to assess outage severity prediction performance.


### Exploratory Analysis for Feature Engineering

#### Determining appropriate thresholds for outage severity (target variable)

Figure 1 
<iframe src='assets/ca_fig.html' width=900 height=500 frameBorder=0></iframe>

Figure 1 shows a boxplot of the value ranges for attribute number of customers affected in an outage. The observed ranges are used to determine outage severity in terms of customers affected.
- Number of customers affected: <=75k (minor), <=150k (moderate), <=360k (major), over 360k (severe)


Figure 2
<iframe src='assets/od_fig.html' width=900 height=500 frameBorder=0></iframe>

Figure 2 shows a boxplot of the value ranges for outage duration in hours. The observed ranges are used to determine outage severity in terms of outage duration.
- Duration: <=12 hours (minor), <=48 hours (moderate), <=117 hours (major), over 117 hours (severe)


#### Determining appropriate transformations for learning features

Figure 3
<iframe src='assets/util_fig.html' width=900 height=500 frameBorder=0></iframe>

Figure 3 shows the distribution of utility contribution (in %) to GSP, subset by severity. Since the distribution of every class appears to be centered at a point on the x-axis, binning the utility contribution percentages by 1-D k-means clusters could capture the individual pattern of each severity class. 


Figure 4
<iframe src='assets/ncs_fig.html' width=900 height=500 frameBorder=0></iframe>

Figure 4 shows the distribution of the annual number of customers served in the U.S. state where an outage occurred, subset by severity. The composition of severity classes seems to fluctuate and changes significantly at different levels of customers served, with slightly similar patterns every 3-4 bins; this suggest that a uniform discretization (binning) strategy could potentially group similar data points to reduce noise, while emphasizing relevant patterns across the groups.


Figure 5
<iframe src='assets/ano_fig.html' width=900 height=500 frameBorder=0></iframe>

Figure 5 shows the distribution of anomaly levels, subset by severity. Interestingly, across the classes, the distributions are already in a nice form with gradual changes across the anomaly levels, except for a spike in class 4 (severe) for levels above 1.5. To prevent encountering information loss from unnecessary transformations, the anomaly levels attribute will be kept in original form.


## Baseline Model

### Preliminary Description

**Model**: Random Forest Classifier

**Features** - name (type, transformation):
 - **anomaly level** (quantitative, keep as-is)
 - **climate category** (nominal, one-hot encode)
 - **cause category** (nominal, one-hot encode)
 - **total customers served** (quantitative, keep as-is)
 - **utility contribution** (quantitative, keep as-is) 
 - **outage start time** (ordinal, extract hour)


Figure 6 
<iframe src='assets/baseline-confusion-matrix.png' width=1000 height=500 frameBorder=0></iframe>

Figure 6 shows the confusion matrix of the baseline model. Importantly, it emphasizes the class imbalance present in the dataset through the high number of true positives and true negatives for the severe class.


Figure 7
<iframe src='assets/baseline-roc.png' width=1000 height=622 frameBorder=0></iframe>

Figure 7 contains one ROC curve for each class, generated independently using the One-vs-Rest multiclass mechanism on the test set. The model has good class-specific performance as shown by having AUC's consistently above 0.71 for every class, indicating a high capacity to correctly predict any given class from the rest.


### Detailed Description & Performance of Baseline Model

**Model**: Random Forest Classifier
- Details: max tree depth = 9

**Performance**:
- **Matthews' Correlation Coefficient** - MCC (primary metric of choice):
    - Training MCCs:
        - macro mcc: 0.7478743463001144
        - micro mcc: 0.7489375119471762
        - weighted mcc: 0.7566766349550407
    <br><br>
    - Testing MCCs
        - macro mcc: 0.3008953541537168
        - micro mcc: 0.3148629576357869
        - weighted mcc: 0.3152244912688657
<br><br>
- **Accuracy** (for comparison & contrast with MCC):
    - Training accuracy: 0.8197725284339458
    - Testing accuracy: 0.5104712041884817

**Assessment**:

**Overall**:  The baseline model achieved a training Matthews correlation coefficient (MCC) of 0.74, which indicates a strong correlation between actual and predicted labels. However, the model only achieved a testing MCC of 0.31, to distinguish against 0.31 in the context of accuracy, a 0.31 correlation indicates fair predictive power given there is a moderate degree of agreement in the model's predictions. Nonetheless, the observed drop in performance is indicative of overfitting, hence the model is **not** generalizing well to unseen data. Similarly, the baseline model achieved 81% training accuracy, and only 51% testing accuracy, which is also indicative of overfitting and an inability to generalize to unseen data.

**Addressing class imbalance**: As inspected, there is a subtle class imbalance present, where class 4 (severe) represents 42% of labels in the train set and 41% of labels in the test set. Thus, the MCC is an insightful performance indicator because it takes into account the nuance of false discoveries (FP, FN) in its calculation. In our multiclass context, the contrast among macro, micro, and weighted average MCC is also important to understand how the model performs for each class. Since the weighted MCC takes into account class frequencies and balances out each class's influence on the final metric value using weights, a higher weighted MCC than macro and micro MCC's indicates the model is performing **well** across all classes. Granted, our class imbalance was considerably subtle to begin with.

**Generalization capacity**: Although according to the above train and test performance metrics, the model is seemingly not generalizing well to unseen data, a close analysis of the Receiver Operating Characteristic (ROC) curves for each class, generated using the One-vs-Rest mechanism, on the test set reveal that the model is actually performing relatively well on unseen data. Across all classes, with areas under the ROC curve ranging from 0.72 to 0.81, the model illustrates a good capacity to distinguish any given class from the rest, which is a promising sign about the model's applicability on unseen data.

**Conclusion (baseline)**: The model is acceptable is its ability to differentiate any given class (out of 4) from the rest, on both train and test sets. However, it is **not good** for its inability to generalize to unseen data due to overfitting.


## Final Model


### Preliminary Description

**Model**: Random Forest Classifier

**Features from Baseline** - name (type, transformation):    
 - **anomaly level** (quantitative, keep as-is)
 - **climate category** (nominal, one-hot encode)
 - **cause category** (nominal, one-hot encode)
 - **outage start time**: (ordinal, extract hour)

**New Features** - name (type, transformation):
 - **total customers served** (quantitative, uniformly bin values into **n** intervals)
    - **Rationale**: Binning data points can reveal patterns on a broader level when similar points are combined. Binning can improve accuracy by emphasizing the overlying trend, which significantly reduces noise. Considering the data generating process, noise can easily arise from human errors or inconsistencies, especially with metrics such as total customers served that oftentimes rely on estimates and general counts.
    <br><br>
 - **utility contribution** (quantitative, bin values into **m** intervals by 1-D k-means clusters)
    - **Rationale**: As mentioned, grouping similar data points together highlights the broader pattern, and greatly reduces noise. From the above histogram of utility contribution, across all classes, the distribution centers around some central mean value, making 'kmeans' the ideal strategy to set bin widths, rather than 'uniform'.

**Methodology**: hyperparameters **max tree depth**, **n**, and **m** are determined by grid search

- **Tuning hyperparameters** - search for the best combination of hyperparameters that has the highest average validation accuracy among all combinations

    - **max_depth** ~ max depth of decision trees, determined using grid search, in combination with n and m, to find the optimal depth that balances model complexity and average validation accuracy.

    - **n** (n_bins) ~ the number of bins in identical widths used to discretize data points in total customers served, with ordinal bin encoding. 

    - **m** (n_bins) ~ the number of bins to discretize data points in utility contribution by k-means clusters, with ordinal encoding.


Figure 8
<iframe src='assets/final-confusion-matrix.png' width=9000 height=500 frameBorder=0></iframe>

Figure 8 shows the confusion matrix of the final model. Similar to baseline, the higher values along the diagonal indicate the model is consistently making correct predictions for each class. Conversely, several high values off the diagonal indicate those are instances where the model mistakenly predicted an incorrect class for some true label.


Figure 9
<iframe src='assets/final-roc.png' width=900 height=622 frameBorder=0></iframe>

Figure 9 shows the ROC curves for each class. Similar to baseline, the curves, and their areas, indicate strong class-specific performance for all classes, specifically the model's ability to distinguish any given class from the rest.


### Detailed Description & Performance of Final Model

**Model**: Random Forest Classifier
- **Tuned (Best) hyperparameters** (the combination with highest average validation accuracy): 
    - max tree depth = 10
    - n_bins for total customers discretizer = 16
    - n_bins for utility contribution discretizer: 10

**Performance**:
- **Matthews' Correlation Coefficient** - MCC (primary metric of choice):
    - **Final model**
        - Training MCCs:
            - macro mcc: 0.7871422008858858                          
            - micro mcc: 0.7886730533380014
            - weighted mcc: 0.7961616046803637
        <br><br>
        - Testing MCCs:
            - macro mcc: 0.3185588486551239
            - micro mcc: 0.32771363970602596
            - weighted mcc: 0.3292919894579185
    <br><br>
    - **Baseline model**
        - Training MCCs:
            - macro mcc: 0.7478743463001144
            - micro mcc: 0.7489375119471762
            - weighted mcc: 0.7566766349550407
        <br><br>
        - Testing MCCs
            - macro mcc: 0.3008953541537168
            - micro mcc: 0.3148629576357869
            - weighted mcc: 0.3152244912688657
<br><br>
- **Accuracy**:
    - **Final model**
        - Training accuracy: 0.847769028871391
        - Testing accuracy: 0.518324607329843
    <br><br>
    - **Baseline model**
        - Training accuracy: 0.8197725284339458
        - Testing accuracy: 0.5104712041884817

**Assessment**:

Across both performance metrics, accuracy and MCC, the final model seems to be a subtle improvement from the baseline model, though both models still **do not** generalize well to unseen data, as evident in the heavy performance drops when evaluated on the test set. Interestingly, it appears that the final model carries virtually every **pros** of the baseline model, which includes good performance across all classes (weighted MCC > micro and macro MCC), and good class-specific performance. The latter is shown by the Test ROC curves, where the areas under the curve are consistently high, which indicates the model's ability to discern between any given class versus the rest. Nonetheless, there is still a high degree of overfitting in the final model that is causing poor performance on unseen data.

**Improvements from Baseline**:

1) **Higher MCC and Accuracy**: The final model consistently achieve higher performance metrics over the baseline model; this is particularly significant with MCC, which considers both class-specific performance (macro MCC) and class frequencies (weighted MCC), making it more robust and reliable given our slightly imbalanced dataset.

2) **Stability**: Since the final model was implemented with hyperparameter-tuning through grid search and 5-fold cross-validation, meaning it has been tested much more thoroughly across different samples of the dataset and configurations, its performance results is likely to be significantly more stable and reliable than baseline.


## Fairness Analysis


### Fairness Analysis Description

**Purpose**: Assess how the final model perform with respect to two groups, or how "fair" it generates predictions for the two groups.

**Group 1**: Outages in states serving **higher** annual numbers of customers

**Group 2**: Outages in states serving **lower** annual numbers of customers

**Null Hypothesis**: The final model is fair, its Matthews correlation coefficient for Group 1 and Group 2 are approximately the same, any observed differences are due to chance.

**Alternative Hypothesis**: The final model is unfair, its Matthews correlation coefficient for Group 1 is lower than its MCC for Group 2. The observed difference is unlikely to have occurred by chance alone.

**Specifications**:
- **Evaluation metric**: Matthews correlation coefficient (MCC) - refer to the model descriptions above for why the MCC is our performance evaluation metric of choice.

- **Methodology**: Permutation testing to measure statistical significance of observed test statistic

- **Test statistic**: Difference in MCC (high - low)

- **Significance level**: 1%

- The threshold used to determine whether a state serves high, or low, number of customers annually is the **median** annual number of customers served, aka the median of column TOTAL.CUSTOMERS. As shown in the histogram immediately follows, the **median** is ideal to separate higher vs. lower numbers of customers served because it strikes the best split, or balance, between higher and lower numbers of customers, relative to other plausible choices, such as the mean or q3 (75th percentile).

**Rationale**: The two groups above were chosen because the distinction between them is relevant in comparing some given outages. In assessing whether the final model performs better, or worse, for one group over the other, or fair across both groups, we can assess how "fair" the model is toward the groups. 
- For illustration, suppose the model statistically significantly achieves higher Matthews correlation coeffcients in predicting severity for outages in states serving lower annual numbers of customers (as observed), this could (potentially) mean our model is biased toward that group, some plausible explanations can be:

    1) We have a large amount of data points belonging to the biased group, which may lead to better performance for that group.

    2) States that serve lower numbers of customers may have reported more accurate numbers of customers affected (smaller population --> faster, simpler to accurately assess the scope of an outage), which will likely influence the accuracy of our target feature (severity) labeling ealier.

- In effect, a fairness analysis will be greatly helpful to determine potential loopholes in the data generating process, representation of groups, as well as our own model for future improvements.


Figure 10
<iframe src='assets/tcs_fig.html' width=1000 height=500 frameBorder=0></iframe>

Figure 10 shows the distribution of annual total customers served in the U.S. state where an outage occurred, marked with three plausible thresholds to separate low vs. high values, that are the median, mean, and 75th percentile. Considering the right-skewed distribution of the data, and the best half split, the median is the ideal threshold.

Figure 11
<iframe src='assets/fair_fig.html' width=1000 height=500 frameBorder=0></iframe>

Figure 11 shows the empirical distribution of the difference in MCC under the null hypothesis that the final model is fair both in its predictions for outages in states serving lower numbers of customers, and outages in states serving larger numbers of customers. At a p-value of approximately 0.35, by chance, it is reasonable to have the observed difference of -0.03.


### Fairness Analysis Conclusion

**P-value**: 0.35
- \> 0.01 significance level

**Conclusion**: Since our p-value of 0.35 is signficantly greater than our significance level of 0.01, we **Fail to Reject** the null hypothesis, and conclude that it is reasonable to have a difference in MCC as observed. With a considerably low observed difference in MCC, of roughly -0.03, that is **not** statistically significant, our final model does likely to have achieved MCC parity.

