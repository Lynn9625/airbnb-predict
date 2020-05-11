# airbnb-predict


Abstract: This project focused on predicting price of Airbnb in NYC. We approached it using Decision Tree, Random Forest and Gradient Boosting. The results suggested that Random Forest and Gradient Boosting showed higher accuracy than Decision Tree. And distance to closest subway was the most important predictor.


1. Introduction and Motivation
Airbnb is a marketplace for short term house or apartment rental. Hosts can rent out their house for a few days to months while they are away or rent out a spare bedroom in their apartment. One challenge for hosts is to determine a reasonable price for rental. The market is dynamic due to environment change and holiday seasons, so the rental price need to be updated in time. If a host charges above market, renters would be more likely to choose similar houses around the area and host will lose chance to make money.
The goal of this project is to determine a model to predict rent price and help hosts formulate reasonable price strategy. Factors used for prediction include number of reviews, room type, borough, amenities, accommodates, number of bedrooms, number of bathrooms, cancellation policy, days on the market and distance to the nearest subway station.


2. Methodology 
2.1. Datasets
(1) New York City Airbnb 2019 Dataset: scraped on Sep. 2019; contains 47,754 homestay records and relevant information like host name, number of reviews, amenities, longitude, latitude, room type, borough, price, etc.
(2) The New York City Subway Station: provides geographical information (latitude and longitude) of 473 NYC subway stations.
2.2. Packages
Pandas/ Numpy/ Matplotlib/ Scikit-learn
2.3. Data Cleaning and Data Integration
(1) Drop outliers: removed records with prices exceeding 99% of the homes.
(2) Variable manipulation
    (a) Price: divided price into 3 bins (“low”, “medium” or “high”), so each home’s price would fall into one of them.
    (b) Amenity: among almost 100 kinds of amenities, we picked up the 20 most common ones like wifi and hair dryer. Then, had 
        all of them dummy-coded.
    (c) Days on the market: obtained each home’s days on the market by calculating the days from the hosting start date to the
        data scraped date; removed records which had a null value for this variable.
    (d) Distance to the closest subway: calculated the distance between every home and its nearest subway station based on
        latitude and longitude. Next, made the distance as an additional column and used it as a predictor in the following           modeling process.
2.4. Data Analysis
(1) Descriptive analysis: in order to get a general idea about Airbnb homestays in NYC, we gave several visualizations like
    distribution of room types in five boroughs of NYC​, Airbnb prices on NYC map, and Airbnb prices in five boroughs of NYC.
(2) Modeling: we built three machine learning models to predict price of Airbnb in NYC, which were Decision Tree, Gradient     
    Boosting and Random Forest. Then we evaluated each model’s performance and explored the importance of each predictor.


3. Results
(1) Built Decision Tree model with training sample and tested it within testing sample. Accuracy: 0.604.
(2) Built Gradient Boosting model with training sample and tested it within testing sample. Accuracy: 0.701.
(3) Built Random Forest model with training sample, used cross validation to pick the best parameters, and tested it within 
    testing sample. Accuracy (number of trees=150): 0.706.
(4) According to the results above, we got to know that Gradient Boosting and Random Forest beat Decision Tree had higher 
    accuracy. And the 3 most important predictors are Distance to Closest Subway Station, Days on Market, and Number of 
    Reviews.
  


4. Discussions 
4.1. Strength
Although predicting price of Airbnb is not a pioneering topic anymore, we still did some innovative work. The most remarkable thing was adding transportation factor into consideration, and the final results of modeling also have proven that such kind of effort was valuable, because the distance to the nearest subway had the most sway over price compared to other predictors.
4.2. Limitation and Future Direction
Undeniably, the accuracy of all of these three models weren’t that high, suggesting that there was still room for improvement. We thought two factors were likely to contribute to this result. (1) Parameters haven’t been adjusted to the best level (2) Omission of other key predictors.
Therefore, in future research, we could try to use other models and optimize all kinds of parameters. Of course, taking other factors into account is also a considerable way. Possible options include the quality of image shown on Airbnb and hosts’ description.


5. Conclusion
Random Forest and Gradient Boosting demonstrated higher accuracy than Decision Tree. And distance to closest subway was the most important predictor. But none of these three models was very accurate, which indicates that more future effort is worth trying.
