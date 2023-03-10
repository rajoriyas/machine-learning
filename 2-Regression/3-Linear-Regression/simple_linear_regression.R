#Sample Linear Regression

#Data Preprocessing 

#Import Dataset
dataset = read.csv('Salary_Data.csv')

#Spliting
library('caTools')
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
#training_set = scale(training_set)
#test_set = scale(test_set)

#Fitting Linear Regression to the Training Set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#Predicting the testset result
y_pred = predict(regressor, newdata = test_set)

#Visualising the training set result
#install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs Exprience (Training Set)') +
  xlab('Years of Experience') + 
  ylab('Salary')
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y =predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Exprience (Test Set)') +
  xlab('Years of Experience') + 
  ylab('Salary')

