# Bikes Rented Out

The model we will deploy is predicting using the xgboost package. Our model will perform regression on the data, using three environmental variables (temperature, humidity, windspeed) to predict the number of bikes that will be rented out from a bike sharing scheme. 

## Methodology

### 1.Pre-processing the data
The variable we want to predict is count and we'll do this using temperature, humidity and windspeed.We won’t use date for our prediction, but it might be useful for plotting.  Now, convert it to a datetime object so that we can easily extract information from it, like the day of the week, month and year.

### 2.Visualising the data
We can look quickly at the pattern of bike usage over time with some simple barplots. The information extracted earlier from the date column makes it easy to adjust granularity from days to months, as well as compare years.
It’s also useful to quickly visualise the variables we are interested in. The seaborn package is great for this. A pairplot will show the relationship between the target variable count and all other variables. By passing kind='reg' to the plotting function, a regression line can be fitted. This gives an indication of whether there might be a useful linear relation between variables.
It looks like there may be a good relationship between temperature and the number of bikes hired.
![1_geyc3NB82OEuW3egWZm2Rw](https://user-images.githubusercontent.com/74424623/139555991-eeb6f09b-364f-4407-a07e-89bac2678a7e.png)

### 3. Evaluating the model

There are several intrinsic evaluation metrics for regression models. To see how our predictions match the true values, we can calculate R2, Mean Absolute Error and the Explained Variance. Briefley, we want R2 and EVS to be close to 1.0 and MAE to be close to 0. We can also plot predictions against truth in a regression plot, which will automatically fit a line for us.

![1_Lt8qwaQG7DUZBGJ_3b9VUA](https://user-images.githubusercontent.com/74424623/139556141-81d6fa14-5fe0-406b-b3c4-d2add1fce66d.png)

### 4. Getting the model ready for deployment
We can save our trained classifier model to disk using pickle. It can then be reloaded later on and used exactly as if we had trained it.


