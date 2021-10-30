# Bikes Rented Out

The model we will deploy is predicting using the xgboost package. Our model will perform regression on the data, using three environmental variables (temperature, humidity, windspeed) to predict the number of bikes that will be rented out from a bike sharing scheme. 

## Methodology

### 1.Pre-processing the data
The variable we want to predict is count and we'll do this using temperature, humidity and windspeed.We won’t use date for our prediction, but it might be useful for plotting.  Now, convert it to a datetime object so that we can easily extract information from it, like the day of the week, month and year.

### 2.Visualising the data
We can look quickly at the pattern of bike usage over time with some simple barplots. The information extracted earlier from the date column makes it easy to adjust granularity from days to months, as well as compare years.
It’s also useful to quickly visualise the variables we are interested in. The seaborn package is great for this. A pairplot will show the relationship between the target variable count and all other variables. By passing kind='reg' to the plotting function, a regression line can be fitted. This gives an indication of whether there might be a useful linear relation between variables.
It looks like there may be a good relationship between temperature and the number of bikes hired.
