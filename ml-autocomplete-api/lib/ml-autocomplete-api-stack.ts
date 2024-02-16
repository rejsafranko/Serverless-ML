import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from "aws-cdk-lib/aws-lambda";

export class MlAutocompleteApiStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const dockerPredict = new lambda.DockerImageFunction(
      this,
      "DockerPredict",
      {
        code: lambda.DockerImageCode.fromImageAsset("./image", {
          cmd: ["predict.handler"],
        }),
        memorySize: 1024,
        timeout: cdk.Duration.seconds(30),
        architecture: lambda.Architecture.X86_64,
      }
    );

    const predictFunctionUrl = dockerPredict.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      cors: {
        allowedMethods: [lambda.HttpMethod.POST],
        allowedHeaders: ["*"],
        allowedOrigins: ["*"],
      },
    });

    const dockerTrain = new lambda.DockerImageFunction(this, "DockerTrain", {
      code: lambda.DockerImageCode.fromImageAsset("./image", {
        cmd: ["train.handler"],
      }),
      memorySize: 1024,
      timeout: cdk.Duration.seconds(30), // Pazi na ovo!!!!!!
      architecture: lambda.Architecture.X86_64,
    });

    const trainFunctionUrl = dockerTrain.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      cors: {
        allowedMethods: [lambda.HttpMethod.GET],
        allowedHeaders: ["*"],
        allowedOrigins: ["*"],
      },
    });
  }
}
