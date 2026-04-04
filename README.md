Revenue Predictor: Full Stack ML System with automated CI/CD and Deployment on AWS using Docker
==============================

This project demonstrates how to build a scalable infrastructure around a machine learning model, taking it from experimentation phase to a production deployed application.  

**Live application:** http://ec2-3-239-56-232.compute-1.amazonaws.com/  

**Experiment tracking:** https://dagshub.com/shubhamyadav2442/Revenue-predictor-model-trained-on-engineered-features.mlflow/#/experiments  

**Model registry:** https://dagshub.com/shubhamyadav2442/Revenue-predictor-model-trained-on-engineered-features.mlflow/#/models  

## Overview of project:  
The goal of this project is to predict quarterly revenue index using engineered features created from cleaned historical order-number and transaction data. Beyond model development, the focus is on building production ready ML system that supports:  
- Reproducability  
- Automated deployment  
- Continous improvement  
This project reflects how ML systems are actually designed and maintained in real-world environments.  

## Key Highlights of the project:  
- Code versioned using **Git and GitHub**. **Master branch always in deployable state.**  
- Data versioned using **DVC and AWS S3.**  
- Different experiments conducted to build the revenue predictor model with different strategies. Each experiment was tracked by **DAGSHUB and MLflow** for reproducability and collaboration among developers.  
- Best result giving experiment was selected to build **DVC pipeline**. Each run of DVC pipeline was also tracked on MLflow.  
- Implemented **CI/CD pipeline using GitHub actions'** for single-click deployment and updates to application running on production when new code is pushed on master branch.  
    - Runs DVC pipeline pushing pushing the model to **model registry** in staging state for testing.  
    - Performs test on model to promote it to production state on model registry.  
    - Performs test on **flask application**.  
    - Dockerises the application and push the docker image to **AWS ECR.**  
    - Launches an **EC2** instance, pulls the latest docker image from ECR and runs the container.  

## Architecture:  
Code push → GitHub → CI/CD Pipeline → DVC Pipeline → MLflow Tracking → Model Registry → Docker Build → ECR → EC2 → Live Application  

## Tech stack:  
- Python  
- sci-kit learn  
- Flask  
- DVC  
- MLflow  
- DAGsHUB  
- Docker  
- GitHub Actions  
- AWS S3  
- AWS ECR  
- AWS EC2  

## Application on Internet  



