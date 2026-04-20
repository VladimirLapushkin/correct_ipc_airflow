SHELL := /bin/bash

include .env

.EXPORT_ALL_VARIABLES:


.PHONY: upload-src-to-bucket
upload-src-to-bucket:
	@echo "Uploading src to $(S3_BUCKET_NAME)..."
	s3cmd put --recursive src/ s3://$(S3_BUCKET_NAME)/src/
	@echo "Src uploaded successfully"

.PHONY: upload-vars-to-bucket
upload-vars-to-bucket:
	@echo "Uploading vars to $(S3_BUCKET_NAME)..."
	s3cmd put infra/variables.json s3://$(S3_BUCKET_NAME)/vars/
	@echo "vars uploaded successfully"

.PHONY: upload-all
upload-all: upload-data-to-bucket upload-src-to-bucket upload-dags-to-bucket

.PHONY: clean-s3-bucket
clean-s3-bucket:
	@echo "Cleaning S3 bucket $(S3_BUCKET_NAME)..."
	s3cmd del --force --recursive s3://$(S3_BUCKET_NAME)/
	@echo "S3 bucket cleaned"

.PHONY: remove-s3-bucket
remove-s3-bucket: clean-s3-bucket
	@echo "Removing S3 bucket $(S3_BUCKET_NAME)..."
	yc storage bucket delete --name $(S3_BUCKET_NAME)
	@echo "S3 bucket removed"

.PHONY: download-output-data-from-bucket
download-output-data-from-bucket:
	@echo "Downloading output data from $(S3_BUCKET_NAME)..."
	s3cmd get --recursive s3://$(S3_BUCKET_NAME)/output_data/ data/output_data/
	@echo "Output data downloaded successfully"

.PHONY: instance-list
instance-list:
	@echo "Listing instances..."
	yc compute instance list

.PHONY: git-push-secrets
git-push-secrets:
	@echo "Pushing secrets to github..."
	python3 utils/push_secrets_to_github_repo.py

.PHONY: create-venv-archive
create-venv-archive:
	@echo "Creating .venv archive..."
	mkdir -p venvs
	chmod +x ./scripts/create_venv_archive.sh
	bash ./scripts/create_venv_archive.sh
	@echo "Archive created successfully"

.PHONY: upload-venv-to-bucket
upload-venv-to-bucket:
	@echo "Uploading virtual environment archive to $(S3_BUCKET_NAME)..."
	s3cmd put venvs/venv38.tar.gz s3://$(S3_BUCKET_NAME)/venvs/venv38.tar.gz
	@echo "Virtual environment archive uploaded successfully"

.PHONY: create-s3cmd
create-s3cmd:
	bash scripts/create_s3cmd.sh

.PHONY: deploy-full
deploy-full: create-s3cmd  upload-vars-to-bucket upload-venv-to-bucket upload-src-to-bucket

apply:
		$(MAKE) -C infra apply_infra
		$(MAKE) deploy-full

destroy:
		$(MAKE) remove-s3-bucket
		$(MAKE) -C infra destroy_infra
