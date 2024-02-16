# CodX-Autocomplete
<img src="https://github.com/rejsafranko/CodX-Autocomplete/blob/main/logo.jpg" alt="CodX Solutions" width="80" height="80">


## Setup
AWS EC2 Instance: ```pip install -r requirements.txt --no-cache-dir```

Local Machine: ```pip install -r requirements.txt```

AWS CDK: ```npm install -g aws-cdk```

## Deployment
Deploy dockerized :whale: Lambda functions for prediction and training:

```cdk bootstrap --region [REGION]```

```cdk deploy```
