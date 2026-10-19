variable "region" {
  type    = string
  default = "us-west-2"
}

variable "project" {
  type    = string
  default = "bike-demand"
}

variable "github_repo" {
  description = "owner/name allowed to deploy via OIDC, e.g. rohilutture/mlops-demand-forecast"
  type        = string
}

variable "create_service" {
  description = "false on the very first apply (ECR is empty), true after the first image push"
  type        = bool
  default     = true
}

variable "create_github_oidc" {
  description = "set false if your account already has the GitHub OIDC provider"
  type        = bool
  default     = true
}

variable "existing_oidc_provider_arn" {
  type    = string
  default = ""
}

variable "prediction_log_days" {
  type    = number
  default = 90
}

variable "force_destroy" {
  description = "allow terraform destroy to delete a non-empty bucket/repo (portfolio teardown)"
  type        = bool
  default     = true
}
