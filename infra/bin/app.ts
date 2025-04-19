#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { ApiStack } from "../stacks/api-stack";
import { DbStack } from "../stacks/db-stack";

const app = new cdk.App();
const dbStack = new DbStack(app, "ServerlessMlDbStack");
new ApiStack(app, "ServerlessMlApiStack", {});
