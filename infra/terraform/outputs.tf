output "bucket" {
  value = aws_s3_bucket.ml.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "api_url" {
  value = var.create_service ? "https://${aws_apprunner_service.api[0].service_url}" : null
}

output "github_deploy_role_arn" {
  description = "put this in the repo secret AWS_DEPLOY_ROLE_ARN"
  value       = aws_iam_role.github_deploy.arn
}
