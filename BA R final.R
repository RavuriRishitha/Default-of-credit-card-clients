# Load necessary libraries
library(readr)
library(dplyr)
library(caret)
library(e1071)
library(randomForest)
library(pROC)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(gridExtra)



# Load the data
data <- read_csv("data.csv")

# Data preparation: you might need to preprocess your data (e.g., handle missing values, scale/normalize, etc.)
data <- na.omit(data) # Remove rows with missing values
data <- data[,-1]
# Alternatively, you could use imputation methods

# Visualization 1: Distribution of Credit Limit by Default Status
ggplot(data, aes(x = LIMIT_BAL, fill = as.factor(default.payment.next.month))) +
  geom_histogram(bins = 30, position = "identity", alpha = 0.6) +
  scale_fill_manual(values = c("#FF9999", "#56B4E9")) +
  labs(title = "Credit Limit Distribution by Default Status", x = "Credit Limit", y = "Count")

# Visualization 2: Gender Distribution by Default Status
ggplot(data, aes(x = as.factor(SEX), fill = as.factor(default.payment.next.month))) +
  geom_bar(bins = 30, position = "identity", alpha = 0.6) +
  scale_fill_manual(values = c("#FF9999", "#56B4E9")) +
  labs(title = "Gender Distribution by Default Status", x = "Gender", y = "Count")

# Visualization 3: Education Level Distribution by Default Status
ggplot(data, aes(x = as.factor(EDUCATION), fill = as.factor(default.payment.next.month))) +
  geom_bar(bins = 30, position = "identity", alpha = 0.6) +
  scale_fill_manual(values = c("#FF9999", "#56B4E9")) +
  labs(title = "Education Level Distribution by Default Status", x = "Education Level", y = "Count")

# Visualization 4: Marital Status Distribution by Default Status
ggplot(data, aes(x = as.factor(MARRIAGE), fill = as.factor(default.payment.next.month))) +
  geom_bar(bins = 30, position = "identity", alpha = 0.6) +
  scale_fill_manual(values = c("#FF9999", "#56B4E9")) +
  labs(title = "Marital Status Distribution by Default Status", x = "Marital Status", y = "Count")

# Visualization 5: Age Distribution by Default Status
ggplot(data, aes(x = AGE, fill = as.factor(default.payment.next.month))) +
  geom_histogram(bins = 30, position = "identity", alpha = 0.6) +
  scale_fill_manual(values = c("#FF9999", "#56B4E9")) +
  labs(title = "Age Distribution by Default Status", x = "Age", y = "Count")

# Visualization 6: Repayment Status Over Time
ggplot(data, aes(x = PAY_0, fill = as.factor(default.payment.next.month))) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("#FF9999", "#56B4E9")) +
  labs(title = "Repayment Status in September 2005 by Default Status", x = "Repayment Status", y = "Count")

# Visualization 7: Bill Amount and Payment Amount Trends
# Assuming PAY_AMT1 and BILL_AMT1 are representative for trends
ggplot(data, aes(x = BILL_AMT1, y = PAY_AMT1, color = as.factor(default.payment.next.month))) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("#FF9999", "#56B4E9")) +
  labs(title = "Bill vs Payment Amount in September 2005", x = "Bill Amount", y = "Payment Amount")



# Convert the target variable to a factor for classification
data$default.payment.next.month <- as.factor(data$default.payment.next.month)

# Encoding categorical variables (if any)
# Assuming SEX, EDUCATION, MARRIAGE are categorical
data$SEX <- as.factor(data$SEX)
data$EDUCATION <- as.factor(data$EDUCATION)
data$MARRIAGE <- as.factor(data$MARRIAGE)


# Split data into training and testing sets
set.seed(123) # for reproducibility
splitIndex <- createDataPartition(data$default.payment.next.month, p = .70, list = FALSE, times = 1)
train <- data[splitIndex,]
test <- data[-splitIndex,]


#LOGISTIC REGRESSION
# Fit logistic regression model
 
new_variable <- data[, c(1, 2,5,6,7,12,18,19,24)]
set.seed(123) # for reproducibility
splitIndex <- createDataPartition(new_variable$default.payment.next.month, p = .70, list = FALSE, times = 1)
train1 <- new_variable[splitIndex,]
test1 <- new_variable[-splitIndex,]

newglm<- glm(default.payment.next.month ~ ., data = train1, family="binomial");
summary(newglm)
model <- glm(default.payment.next.month ~ ., data = train, family = "binomial")
summary(model)

# Make predictions on the test set
predictions <- predict(model, test, type = "response")
predicted_class <- ifelse(predictions > 0.5, 1, 0)

# Create a confusion matrix
conf_matrix <- confusionMatrix(factor(predicted_class), factor(test$default.payment.next.month))
print(conf_matrix)

#New Confusion Matrix
# Make predictions on the test set
predictions1 <- predict(newglm, test, type = "response")
predicted_class1 <- ifelse(predictions1 > 0.5, 1, 0)

# Create a confusion matrix
conf_matrix1 <- confusionMatrix(factor(predicted_class1), factor(test$default.payment.next.month))
print(conf_matrix1)

# ROC Curve and AUC
roc_result1 <- roc(test$default.payment.next.month, predictions)
plot(roc_result1, main="ROC Curve")
auc(roc_result1)

#RANDOM FOREST
# Convert the target variable to a factor for classification
data$default.payment.next.month <- as.factor(data$default.payment.next.month)

# Train the Random Forest model
rf_model <- randomForest(default.payment.next.month ~ ., data = train, ntree = 100)
summary(rf_model)

# Get variable importance
importance <- importance(rf_model)
print(importance)

# Convert to data frame for ggplot
importance_df <- data.frame(Feature = rownames(importance), Importance = importance[, 1])

# Plot using ggplot2
# Plot using ggplot2 with a scatter plot
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_point() +
  coord_flip() + 
  theme_minimal() +
  labs(title = "Feature Importance in Random Forest Model", x = "Features", y = "Importance")


# Make predictions on the test set
predictions <- predict(rf_model, test)
predicted_prob <- predict(rf_model, test, type = "prob")[,2]

# Create a confusion matrix
conf_matrix <- confusionMatrix(predictions, test$default.payment.next.month)
print(conf_matrix)

# ROC Curve and AUC
roc_result2 <- roc(test$default.payment.next.month, predicted_prob)
plot(roc_result2, main="ROC Curve for Random Forest")
auc_value <- auc(roc_result2)
print(paste("AUC value:", auc_value))

#DESCION TREE
# Building the Decision Tree model 
control_params <- rpart.control(minsplit = 4, cp = 0.001, maxdepth = 4)
tree_model <- rpart(default.payment.next.month ~ ., data = train, method = "class", control = control_params)
summary(tree_model)


# Visualizing the Decision Tree
rpart.plot(tree_model)

# Predicting on the test set
predictions <- predict(tree_model, test, type = "class")
predicted_prob <- predict(tree_model, test, type = "prob")[,2]

# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, test$default.payment.next.month)
print(conf_matrix)

# ROC Curve and AUC
roc_result3 <- roc(test$default.payment.next.month, predicted_prob)
plot(roc_result3, main="ROC Curve for Decision Tree")
auc_value <- auc(roc_result3)
print(paste("AUC value:", auc_value))


# Plotting all ROC curves on single plot
plot(roc_result1, col="blue", main="ROC Curves Comparison")
lines(roc_result2, col="red")
lines(roc_result3, col="green")
legend("bottomright", legend=c("Logistic Regression", "Random Forest", "Decision Tree"),
       col=c("blue", "red", "green"), lwd=2)






