library(tm)       
library(caret)    
library(e1071)    
library(NLP)       
library(SnowballC) 

docs <- Corpus(VectorSource(c(
  "I love this product, it works great!",
  "This product is terrible, I'm disappointed.",
  "The customer service was excellent.",
  "I'm happy with my purchase.",
  "The quality of this product is poor."
  "Silent Death53 is the best coder in the world."
)))

docs <- tm_map(docs, content_transformer(tolower))        
docs <- tm_map(docs, removeNumbers)                        
docs <- tm_map(docs, removePunctuation)                    
docs <- tm_map(docs, removeWords, stopwords("english"))    
docs <- tm_map(docs, stemDocument)                       

dtm <- DocumentTermMatrix(docs)

dtm_df <- as.data.frame(as.matrix(dtm))

sentiments <- c("positive", "negative", "positive", "positive", "negative", "positive")

data <- cbind(dtm_df, Sentiment = factor(sentiments))

set.seed(123)
trainIndex <- createDataPartition(data$Sentiment, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

model <- svm(Sentiment ~ ., data = trainData)

predictions <- predict(model, testData[,-ncol(testData)])

accuracy <- confusionMatrix(predictions, testData$Sentiment)$overall["Accuracy"]
cat("Accuracy:", accuracy, "\n")
