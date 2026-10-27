# AWS footprint for the forecaster:
#   S3       model artifacts + prediction logs (versioned, encrypted, private)
#   ECR      API container images (scan on push, keep last 10)
#   AppRunner  the API: pulls :latest from ECR and auto-deploys on push
#   IAM      App Runner roles (ECR pull, S3 log writes) + GitHub Actions OIDC deploy role
#
#   cd infra/terraform && terraform init && terraform apply -var github_repo=rohilutture/mlops-demand-forecast
# App Runner needs an image in ECR before the service can start: run with
# -var create_service=false first, push an image (CD workflow), then apply again.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = { project = var.project, managed_by = "terraform" } }
}

data "aws_caller_identity" "me" {}

# ---------------------------------------------------------------- storage
resource "aws_s3_bucket" "ml" {
  bucket        = "${var.project}-${data.aws_caller_identity.me.account_id}"
  force_destroy = var.force_destroy
}

resource "aws_s3_bucket_versioning" "ml" {
  bucket = aws_s3_bucket.ml.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml" {
  bucket = aws_s3_bucket.ml.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "ml" {
  bucket                  = aws_s3_bucket.ml.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "ml" {
  bucket = aws_s3_bucket.ml.id
  rule {
    id     = "expire-old-prediction-logs"
    status = "Enabled"
    filter { prefix = "predictions/" }
    expiration { days = var.prediction_log_days }
  }
}

# ---------------------------------------------------------------- registry
resource "aws_ecr_repository" "api" {
  name                 = "${var.project}-api"
  image_tag_mutability = "MUTABLE" # :latest moves; :<git-sha> tags are the immutable record
  image_scanning_configuration { scan_on_push = true }
  force_delete = var.force_destroy
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1, description = "keep last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------- App Runner
resource "aws_iam_role" "apprunner_ecr" {
  name = "${var.project}-apprunner-ecr"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "sts:AssumeRole",
    Principal = { Service = "build.apprunner.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_iam_role" "apprunner_instance" {
  name = "${var.project}-apprunner-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "sts:AssumeRole",
    Principal = { Service = "tasks.apprunner.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy" "apprunner_instance" {
  role = aws_iam_role.apprunner_instance.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:PutObject"], Resource = "${aws_s3_bucket.ml.arn}/predictions/*" },
      { Effect = "Allow", Action = ["s3:GetObject", "s3:ListBucket"],
      Resource = [aws_s3_bucket.ml.arn, "${aws_s3_bucket.ml.arn}/models/*"] }
    ]
  })
}

resource "aws_apprunner_service" "api" {
  count        = var.create_service ? 1 : 0
  service_name = "${var.project}-api"

  source_configuration {
    auto_deployments_enabled = true
    authentication_configuration { access_role_arn = aws_iam_role.apprunner_ecr.arn }
    image_repository {
      image_identifier      = "${aws_ecr_repository.api.repository_url}:latest"
      image_repository_type = "ECR"
      image_configuration {
        port = "8000"
        runtime_environment_variables = {
          PRED_LOG = "s3://${aws_s3_bucket.ml.bucket}/predictions"
        }
      }
    }
  }

  instance_configuration {
    cpu               = "1024"
    memory            = "2048"
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  health_check_configuration {
    protocol = "HTTP"
    path     = "/health"
  }
}

# ---------------------------------------------------------------- CI/CD (GitHub OIDC, no static keys)
resource "aws_iam_openid_connect_provider" "github" {
  count           = var.create_github_oidc ? 1 : 0
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

locals {
  oidc_arn = var.create_github_oidc ? aws_iam_openid_connect_provider.github[0].arn : var.existing_oidc_provider_arn
}

resource "aws_iam_role" "github_deploy" {
  name = "${var.project}-github-deploy"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow", Action = "sts:AssumeRoleWithWebIdentity"
      Principal = { Federated = local.oidc_arn }
      Condition = {
        StringEquals = { "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com" }
        StringLike   = { "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:ref:refs/heads/main" }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_deploy" {
  role = aws_iam_role.github_deploy.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
      { Effect = "Allow", Resource = aws_ecr_repository.api.arn, Action = [
        "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage"] }
    ]
  })
}
