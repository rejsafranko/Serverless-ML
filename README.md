# CodX-Autocomplete
![CodX Solutions](https://github.com/rejsafranko/CodX-Autocomplete/blob/main/logo.jpg)

## Setup
AWS EC2 Instance: ```pip install -r requirements.txt --no-cache-dir```

Local Machine: ```pip install -r requirements.txt```

AWS CDK: ```npm install -g aws-cdk```

## Deployment
Deploy Dockerized :whale: Lambda Functions for prediction and training:

```cdk bootstrap --region [REGION]```

```cdk deploy```
