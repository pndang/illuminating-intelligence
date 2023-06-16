# Illuminating Intelligence
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


## Baseline Model

### Preliminary description

**Model**: Random Forest Classifier

**Features** - name (type, transformation):
 - **anomaly level** (quantitative, keep as-is)
 - **climate category** (nominal, one-hot encode)
 - **cause category** (nominal, one-hot encode)
 - **total customers served** (quantitative, keep as-is)
 - **utility contribution** (quantitative, keep as-is) 
 - **outage start time** (ordinal, extract hour)


Figure 1 
<iframe src='assets/baseline-confusion-matrix' width=900 height=500 frameBorder=0></iframe>


## Final Model


## Fairness Analysis


