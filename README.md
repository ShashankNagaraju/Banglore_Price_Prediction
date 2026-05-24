# Realytics - Bengaluru Real Estate Price Prediction

A machine learning project that predicts residential property prices in Bengaluru based on location, area, number of bedrooms, and bathrooms.

## Overview

This project walks through a complete data science workflow — from raw data to a deployable prediction model - using the Bengaluru House Price dataset.

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Model:** Linear Regression
- **Deployment-ready:** Flask API (pickle export)

## Workflow

1. **Data Cleaning** - Handled missing values, dropped irrelevant features
2. **Feature Engineering** - Extracted BHK from size column, parsed sqft ranges, computed price per sqft
3. **Outlier Removal** - Business logic (min sqft per BHK) + standard deviation based filtering per location
4. **Dimensionality Reduction** - Consolidated 1,300+ locations into 242 meaningful categories
5. **Encoding** - One-hot encoding for location
6. **Model Training** - Linear Regression with 5-fold cross-validation (~84% accuracy)
7. **Export** - Model saved as `.pickle`, column schema saved as `columns.json` for API use

## Results

| Metric | Score |
|--------|-------|
| Cross-Validation Accuracy | ~82–85% |
