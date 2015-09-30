#!usr/bin/python


import graphlab
# import matplotlib.pyplot as plt

# load the data
sales = graphlab.SFrame('home_data.gl/')
# print sales

graphlab.canvas.set_target('browser')
# sales.show(view='Scatter Plot', x='sqft_living', y='price')

# split data .8 and seed = 0
train_data, test_data = sales.random_split(0.8, seed=0)
print 'train data mean price: ', train_data['price'].mean()
print 'test data mean price: ', test_data['price'].mean()

my_feature = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
# train model
my_feature_model = graphlab.linear_regression.create(train_data, target='price', features=my_feature)

# evaluate the model
print 'Evaluation of the model with my feature', my_feature_model.evaluate(test_data)

# get coefficients
my_feature_model.get('coefficients')
# sales[my_feature].show()

# Box whisker plot
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')
# get the zipcode with highest mean price
highPriceZipcode = '98039'
sales_highPriceZipcode = sales[sales['zipcode']==highPriceZipcode]
print 'mean price of house in 98039', sales_highPriceZipcode['price'].mean()
print 'number of houses in the filtered data: ', len(sales_highPriceZipcode)

# filter by living space
sales_filtered = sales_highPriceZipcode[(sales_highPriceZipcode['sqft_living'] > 2000) & (sales_highPriceZipcode['sqft_living'] <= 4000)]
print 'number of houses in the filtered data: ', len(sales_filtered)

# advanced feature
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house
'grade', # measure of quality of construction
'waterfront', # waterfront property
'view', # type of view
'sqft_above', # square feet above ground
'sqft_basement', # square feet in basement
'yr_built', # the year built
'yr_renovated', # the year renovated
'lat', 'long', # the lat-long of the parcel
'sqft_living15', # average sq.ft. of 15 nearest neighbors
'sqft_lot15', # average lot size of 15 nearest neighbors
]

# modeling with the advanced features
advanced_features_model = graphlab.linear_regression.create(train_data, target='price', features=advanced_features)

# evaluate the model
print 'Evaluation of the model with my feature', my_feature_model.evaluate(test_data)
print 'Evaluation of the model with advanced feature', advanced_features_model.evaluate(test_data)


# plot the data, not working...
# plt.plot(test_data['sqft_living'], test_data['price'], '.', test_data['sqft_living'], my_feature_model.predict(test_data))
