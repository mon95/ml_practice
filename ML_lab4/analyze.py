"""
	This program analyzes the CSV of training data and creates a profile for each user.
	The profile contains information about the user's history of opening and clicking on emails.
	This data will be used later to improve prediction accuracy.
"""


import pandas as pd 	#import pandas module which will be used to process the csv

train_df = pd.read_csv('training_dataset.csv')		#Read the given csv and load it into a DataFrame object
train_df = train_df[['user_id', 'opened', 'clicked', 'unsubscribed']]		#Extract only the necessary columns

print "data_loaded"

users_dict = {}			#Create a dictionary which will store user history

"""
	users_dict is a nested dictionary. It has the following structure


	users_dict = {
					user_id_1: {
								"user_id" : user_id_1,
								"total" : 34,					#dummy values
								"n_open" : 32,
								"n_click" : 23,
								"unsub" : False
					}

					user_id_2: {
								"user_id" : user_id_n,
								"total" : 93,					#dummy values
								"n_open" : 65,
								"n_click" : 23,
								"unsub" : False
					}

					................................. 			There are n such entries, one for each user

					user_id_n: {
								"user_id" : user_id_n,
								"total" : 54,					#dummy values
								"n_open" : 12,
								"n_click" : 34,
								"unsub" : True
					}
	}

	The goal is to read the dataframe and load information into this dictionary
	This information can then be exported into a CSV so that it can be incorporated into the feature vector

"""

for row in train_df.itertuples():				#iterate through the rows of the dataframe. The row is represented as a tuple of values

	index, user_id, opened, clicked, unsubscribed = row 		#tuple unpacking automatically unpacks the row elements.
																#note that the first element is the index
	
	if user_id not in users_dict:
		users_dict[user_id] = {'user_id':user_id,'total':0,'n_open':0,'n_click':0,'unsub':False}

		#Check if there is any entry for the user in the users_dict. If not, create a new blank entry.

	users_dict[user_id]['total'] += 1 		#increment the total number of emails received by one
	
	if opened:
		users_dict[user_id]['n_open'] += 1 	#increment the number of opened emails by one if opened == True

	if clicked:								
		users_dict[user_id]['n_click'] += 1 	#increment the number of clicked emails by one if clicked == True

	if unsubscribed:
		users_dict[user_id]['unsub'] = True		#Set the unsubscribed flag if unsubscribed
 
users_df = pd.DataFrame(users_dict.values())	#Flatten the dictionary into a list of dictionaries and convert to a dataframe
										
"""

users_dict.values() has the following structure

	[
					{
								"user_id" : user_id_1,
								"total" : 34,					#dummy values
								"n_open" : 32,
								"n_click" : 23,
								"unsub" : False
					},

					{
								"user_id" : user_id_2,
								"total" : 93,					#dummy values
								"n_open" : 65,
								"n_click" : 23,
								"unsub" : False
					},

					................................. 			There are n such entries, one for each user

					user_id_n: {
								"user_id" : user_id_n,
								"total" : 54,					#dummy values
								"n_open" : 12,
								"n_click" : 34,
								"unsub" : True
					}
	]
	
	Pandas automatically converts this list of dictionaries into a dataframe with the column names as the keys

	So the resulting dataframe looks like this

	user_id          total             n_open         n_click           unsub
	user_id_1			34				32				23				False
	user_id_2			93				65				23				False
	.........................................................................
	user_id_n			54				12				34				True

"""

users_df.to_csv('users.csv', index = False)			#Save the DataFrame to csv. index = False ensures that the
													#csv is not stored with the default index of 0,1,2,3 ....