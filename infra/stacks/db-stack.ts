import { Stack, StackProps, Duration, RemovalPolicy } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as rds from "aws-cdk-lib/aws-rds";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as cdk from "aws-cdk-lib";

export class DbStack extends Stack {
  public readonly dbInstance: rds.DatabaseInstance;
  public readonly secret: secretsmanager.Secret;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const vpc = new ec2.Vpc(this, "ServerlessMLVpc", {
      maxAzs: 2,
    });

    this.secret = new secretsmanager.Secret(this, "DBSecret", {
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: "mluser" }),
        excludePunctuation: true,
        includeSpace: false,
        generateStringKey: "password",
      },
    });

    this.dbInstance = new rds.DatabaseInstance(
      this,
      "ServerlessMLRdsInstance",
      {
        engine: rds.DatabaseInstanceEngine.mysql({
          version: rds.MysqlEngineVersion.VER_8_0_32,
        }),
        vpc,
        credentials: rds.Credentials.fromSecret(this.secret),
        databaseName: "serverlessml",
        instanceType: ec2.InstanceType.of(
          ec2.InstanceClass.T3,
          ec2.InstanceSize.MICRO
        ),
        allocatedStorage: 20,
        maxAllocatedStorage: 100,
        vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
        removalPolicy: RemovalPolicy.DESTROY, // NOT for prod
        deletionProtection: false,
      }
    );

    new cdk.CfnOutput(this, "DBHost", {
      value: this.dbInstance.dbInstanceEndpointAddress,
    });

    new cdk.CfnOutput(this, "DBSecretArn", {
      value: this.secret.secretArn,
    });

    new cdk.CfnOutput(this, "DBName", {
      value: "serverlessml",
    });
  }
}
