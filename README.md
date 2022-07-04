# Predicting Los Angeles House Price

[EDA notebook](https://github.com/JoKerDii/Predicting-Airbnb-House-Price/blob/main/main-notebook/Airbnb_losangeles_eda.ipynb), [Model notebook](https://github.com/JoKerDii/Predicting-Airbnb-House-Price/blob/main/main-notebook/Airbnb_losangeles_modeling.ipynb)

## Overview

There are two goals for this project:

1. Build a predictor that can accurately predict house prices in Los Angeles based on the information about geographic location, hosts, etc. 
2. Develop the predictor to a web app so that users can easily use it to predict house prices based on given relevant information.

The significance of this project is that people who want to rent houses in Los Angeles can use this predictor to have a good sense of the value of the target houses.

## Data Source

The dataset contains information about hosts, geographic locations, neighborhood areas, room types, reviews, availability, house price, etc. The version of the dataset used throughout this project was reported in 2022. It contains 42,041 records and 18 variables. The latest version of the house price data in LA is available from the Inside Airbnb [website](http://insideairbnb.com/los-angeles/). 

We preprocessed the data and then went through exploratory data analysis, and feature engineering. Please refer to the notebook for more information.

## Model

The response variable is 'price'. Other variables as well as engineered variables are predictors. We built several linear models and non parametric models, and tuned hyperparameters through searching techniques. We used RMSE metric from 5-fold cross-validation to evaluate and compare model performance.

We fitted and compared three linear models: Ridge regression, Lasso regression, and Huber regression models. Ridge regression model works better than others. In addition, we built and compared two non-parametric models: random forest regressor and XGBoost regressor. In general, non-parametric models perform a lot better than linear models. **XGBRegressor** achieves the lowest cross-validation RMSE.

## Results and Discussion

The cross-validation RMSE values are summarized in the following table.

| Algorithms              | Cross-validation RMSE |
| ----------------------- | --------------------- |
| Ridge Regression        | 0.430441              |
| LASSO Regression        | 0.430443              |
| Huber Regression        | 0.435920              |
| Random Forest Regressor | 0.277689              |
| XGB Regressor           | 0.272552              |

From the coefficient estimates of Lasso Regression model, we learn that availability365, calculated host listings count, all year availability, shared room type are positively related to the price, while no reviews, unincorporated areas, other cities, latitude contribute to the price in the negative direction.

With the help of LIME, SHAP, and ELI5 explainer, we can interpret regression models, i.e. how the models are making predictions based on features, which features are important, etc. For example, we find houses with entire home/apt type of room, unknown neighborhood, fewer minimum nights required, more reviews, and larger latitude and longitude tend to be more expensive.  

For non-parametric models random forest regressor and XGB regressor, entire home/apt variables appear to be the most important variable.

## Future Work

One of the limitations of this project is that feature engineering was not creative enough. There are a lot more engineering ideas we can implement using this rich data. For example, the neighborhood is a sparse feature with ~300 different levels, but it's important in predicting the price. From the exploratory analysis we found houses close to the beach are more expensive. However, we didn't have enough time to explore and make the most use of it. For future work, we could apply dimensionality reduction techniques such as k-means, DBSCAN, PCA, to cluster neighborhoods into groups with fewer levels. 