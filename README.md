# Serverless MLOps System on AWS

![Architecture](architecture.png)

## Overview

This project consists of three main directories: database, ml-autocomplete-api (API), and model-development (data science). The aim is to create and deploy a machine learning model that predicts mental health disorders. The entire workflow includes training the model through AWS Lambda, deploying it to AWS S3 bucket, making predictions through AWS Lambda, and updating the AWS RDS Aurora database accordingly.

The model-development data science directory contains model training code and evaluation notebooks. I deployed a logistic regression model whose best performing metrics are displayed in the evaluation notebook.

## Architecture

### 1. AWS RDS Aurora Database

The SQL queries for creating tables, reading and inserting data, increasing and decreasing new table row counter and most importantly the stored procedure for triggering model retraining, are stored in the database directory. Each time a prediction is made, it stored in the table as a new row and the new data counter is increased. A stored procedure is triggered each time new data arrives. It increases the counter or, if the treshold of 100 new rows is met, resets the counter and invokes the AWS Lambda train function.

### 2. AWS Lambda Functions

I made a train and predict Lambda function. I deploy a Docker container to each one so I can easily build the environment and install the neccesary dependencies. When the train function is triggered it stores the fresh model on an AWS S3 bucket. When a prediction is triggered, it returns the top 3 suggested group services and stores the input and the top predicted output as a new table row to my database.

### 3. AWS S3 Bucket

The bucket serves a model storage where the train Lambda function saves the model, and the predict Lambda function loads the model from.

## Deployment
Deploy dockerized :whale: Lambda functions for prediction and training:

```npm install -g aws-cdk```

```cdk bootstrap --region [REGION]```

```cdk deploy```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
