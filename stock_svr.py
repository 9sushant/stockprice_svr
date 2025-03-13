import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

# plt.switch_backend('newbackend')  # Remove or comment out this line

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		temp_prices = [] # Temporary list to hold prices before scaling
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			temp_prices.append(float(row[1]))

		# Scale the prices using StandardScaler
		scaler = StandardScaler()
		scaled_prices = scaler.fit_transform(np.array(temp_prices).reshape(-1, 1))
		prices.extend(scaled_prices.flatten()) # Flatten back to 1D array

	return

def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

	# Scale dates as well (optional, but good practice for consistency)
	date_scaler = StandardScaler()
	scaled_dates = date_scaler.fit_transform(dates)
	x_scaled = date_scaler.transform(np.array([[x]])) # Scale the prediction date 'x'

	svr_lin = SVR(kernel= 'linear', C= 1e3)
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 3) # Increased degree to 3 (try 4, 5 later)
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # Keep gamma=0.1 for now, experiment with it later

	svr_rbf.fit(scaled_dates, prices) # fitting the scaled data points in the models
	svr_lin.fit(scaled_dates, prices)
	svr_poly.fit(scaled_dates, prices)

	plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints
	plt.plot(dates, svr_rbf.predict(scaled_dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
	plt.plot(dates,svr_lin.predict(scaled_dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
	plt.plot(dates,svr_poly.predict(scaled_dates), color= 'blue', label= 'Polynomial model (degree=3)') # Updated label
	plt.xlabel('Date')
	plt.ylabel('Price (Scaled)') # Update y-axis label to indicate scaling
	plt.title('Support Vector Regression on Scaled Stock Prices') # Update title
	plt.legend()
	plt.show()

	# Predict using scaled input 'x_scaled' and inverse_transform the predictions
	rbf_predicted_scaled = svr_rbf.predict(x_scaled)
	lin_predicted_scaled = svr_lin.predict(x_scaled)
	poly_predicted_scaled = svr_poly.predict(x_scaled)

	# Inverse transform the scaled predictions back to the original price scale
	rbf_predicted = scaler.inverse_transform(rbf_predicted_scaled.reshape(-1, 1)).flatten()[0]
	lin_predicted = scaler.inverse_transform(lin_predicted_scaled.reshape(-1, 1)).flatten()[0]
	poly_predicted = scaler.inverse_transform(poly_predicted_scaled.reshape(-1, 1)).flatten()[0]

	return rbf_predicted, lin_predicted, poly_predicted

get_data('GOOGL.csv') # calling get_data method by passing the csv file to it
#print "Dates- ", dates
#print "Prices- ", prices

predicted_price = predict_price(dates, prices, 29)

print(predicted_price) # To see the predicted prices