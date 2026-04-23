# Review Sentiment & Star Rating Prediction
### Purpose of the project

Judge the emotions of the users from their reviews, and predict the stars they gave


### Group members 
1. 梅兰妮 Aristakesian Melaniia
2. 王子函 Zihan Wang
3. 刘志鹏 Zhipeng Liu
4. 阿丽珊 Antonova Aleksandra
5. 安娜 Elovskikh Anastasiia


The task has two parts. First, sentiment classification: is the review negative, neutral, or positive. Second, star rating prediction: predict the exact 1–5 star score. 

### Dataset 
Yelp Open Dataset, which combines several JSON files covering reviews, businesses, and users. 

### Models Tested:
#### ML models
- Random Forest
- Logistic Regression
- Linear SVM
- XGBoost

*Logistic Regression* performed best overall

#### Neural Network:
- BERT + Category Fusion
- BERT + Cross-Attention  
- BERT + Gate Fusion
- BiLSTM
- LSTMjjjj
- Hierarchical Transformer + Cross-Attention

The best result showed *Hierarchical Transformer + Cross-Attention* and *BERT + Gate Fusion*

