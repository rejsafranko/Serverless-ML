import { Stack, StackProps, Duration } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";

export class ApiStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // predict Lambda function
    const predictFn = new lambda.DockerImageFunction(this, "PredictFn", {
      code: lambda.DockerImageCode.fromImageAsset("../../lambda/predict"),
      architecture: lambda.Architecture.ARM_64,
      memorySize: 512,
      timeout: Duration.seconds(30),
      environment: {
        MODEL_BUCKET: "ml-demo-models",
        CHAMPION_SSM_PARAM: "/serverless-ml/champion-model",
        FEATURE_TABLE: "mental_health_features",
      },
    });

    predictFn.addFunctionUrl({ authType: lambda.FunctionUrlAuthType.NONE });

    // drift detection Lambda function
    const driftFn = new lambda.DockerImageFunction(this, "DriftFn", {
      code: lambda.DockerImageCode.fromImageAsset("../../lambda/drift"),
      architecture: lambda.Architecture.ARM_64,
      memorySize: 512,
      timeout: Duration.seconds(30),
      environment: {
        FEATURE_TABLE: "mental_health_features",
        KS_RESULTS_TABLE: "ks_test_results",
      },
    });

    // EventBridge rule (once a week)
    const rule = new events.Rule(this, "DriftDetectionRule", {
      schedule: events.Schedule.rate(Duration.days(7)),
    });

    rule.addTarget(new targets.LambdaFunction(driftFn));
  }
}
