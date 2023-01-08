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