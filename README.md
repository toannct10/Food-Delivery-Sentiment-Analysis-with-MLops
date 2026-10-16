# Food-Delivery-Sentiment-Analysis-with-MLops

conda create -n mlops_project python=3.12 -y

conda activate mlops_project

pip install -r requirements.txt

## DVC

dvc init

dvc repro

dvc dag

## AWS

aws configure

763348960323.dkr.ecr.ap-southeast-1.amazonaws.com/mlflowproject

#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws

## Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
