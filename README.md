# Default-of-credit-card-clients
The analysis of the credit card default dataset, spanning from April 2005 to September 2005, focused on predicting whether a customer would default on their next month's payment. With 30,000 records and 25 attributes, the dataset underwent meticulous data cleaning and pre-processing, revealing a well-maintained dataset without missing records. Categorical variables were converted into factors and encoded for classification tasks. Exploratory Data Analysis (EDA) uncovered insights, notably the correlation between higher credit limits and an increased likelihood of default.
Three classification models were employed: Logistic Regression, Random Forest, and Decision Tree. The Logistic Regression model demonstrated commendable sensitivity and specificity, achieving an accuracy of 0.8115. However, limitations were observed in specificity. The Random Forest model, highlighted by a variable importance plot, outperformed others, emphasizing PAY_0 as a critical feature. The Decision Tree underscored recent payment behavior as a key predictor.
Inferentially, the Random Forest model exhibited superior discriminative power, as evidenced by the highest Area Under the Curve (AUC) in the ROC curve analysis. This suggests its reliability in predicting credit card default. The findings contribute valuable insights into credit risk prediction, with a recommendation for further feature refinement and exploration of advanced models to enhance predictive accuracy. This analysis lays the foundation for developing a robust credit risk prediction system, essential for informed decision-making in financial contexts.
 
