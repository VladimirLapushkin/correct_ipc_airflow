resource "yandex_airflow_cluster" "airflow_cluster" {
  name               = var.instance_name
  subnet_ids         = [var.subnet_id]
  service_account_id = var.service_account_id
  admin_password     = var.admin_password

  code_sync = {
    git_sync = {
      repo     = var.git_repo
      branch   = "main"
      sub_path = "/dags"
      ssh_key  = file(pathexpand(var.git_ssh_private_key))
    }
  }

  webserver = {
    count              = 1
    resource_preset_id = "c1-m4"
  }

  scheduler = {
    count              = 1
    resource_preset_id = "c1-m4"
  }

  worker = {
    min_count          = 1
    max_count          = 2
    resource_preset_id = "c1-m4"
  }

  airflow_config = {
    "api" = {
      "auth_backends" = "airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session"
    }

    "scheduler" = {
      "dag_dir_list_interval" = "10"
    }

    "bootstrap" = {
      "bucket"   = var.bucket_name
      "key"      = "vars/variables.json"
      "endpoint" = var.yc_storage_endpoint_url
    }
  }

  pip_packages = [
    "boto3"
  ]

  logging = {
    enabled   = true
    folder_id = var.provider_config.folder_id
    min_level = "INFO"
  }
}

#Обновляем .env файл с ID кластера Airflow

resource "null_resource" "update_env" {
  provisioner "local-exec" {
    interpreter = ["/usr/bin/env", "bash", "-lc"]
    command = <<EOT
    set -eu
      AIRFLOW_CLUSTER_ID="${yandex_airflow_cluster.airflow_cluster.id}"
      if grep -q "^AIRFLOW_CLUSTER_ID=" ../.env; then
        sed -i "s|^AIRFLOW_CLUSTER_ID=.*|AIRFLOW_CLUSTER_ID=$AIRFLOW_CLUSTER_ID|" ../.env
      else
        echo "AIRFLOW_CLUSTER_ID=$AIRFLOW_CLUSTER_ID" >> ../.env
      fi
    EOT
  }
  depends_on = [
    yandex_airflow_cluster.airflow_cluster
  ]
}

# resource "null_resource" "update_env" {
#   provisioner "local-exec" {
#     interpreter = ["/usr/bin/env", "bash", "-lc"]
#     command = <<-EOT
#       set -eu
#       set -o pipefail

#       ENV_FILE="../.env"
#       NEW_LINE="AIRFLOW_CLUSTER_ID=${yandex_airflow_cluster.airflow_cluster.id}"

#       # На всякий случай нормализуем файл (если он уже существует с CRLF)
#       if [ -f "$ENV_FILE" ]; then
#         sed -i 's/\r$//' "$ENV_FILE"
#       else
#         touch "$ENV_FILE"
#       fi

#       # Переписываем файл без старой строки и добавляем новую
#       grep -v '^AIRFLOW_CLUSTER_ID=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
#       echo "$NEW_LINE" >> "$ENV_FILE.tmp"
#       mv "$ENV_FILE.tmp" "$ENV_FILE"
#     EOT
#   }

#   depends_on = [
#     yandex_airflow_cluster.airflow_cluster
#   ]
# }