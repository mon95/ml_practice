"""
	The program is used to read the training_data and vectorize it
	so that the vectors can be given as input to the ML algorithms
	The data needs to be cleaned and processed and augmented with
	the user history before it can be fed to the ML models.

"""

import pandas as pd 	#import pandas module which will be used to process csv files
import numpy as np 		#import numpy which will be used to work with the feature vectors

train_df = pd.read_csv("training_dataset.csv")					#read the training data from the csv
users_df = pd.read_csv("users.csv").set_index('user_id')		#read the user profile data and set the index as user_id
																#instead of the autoincrement int values 1,2,3 ....

print "training data loaded"

df = train_df.join(users_df, on = 'user_id')					#perform an inner join using user_id as the key to join on
																#Go throguh http://pandas.pydata.org/pandas-docs/stable/merging.html

ignore_list = [	
				'user_id',
				'mail_id', 
				'open_time', 
				'click_time', 
				'unsubscribe_time', 
				'clicked', 
				'unsubscribed', 
				'mail_type', 
				'mail_category',
				'opened',
				'hacker_timezone',					
			]							#make list of columns to ignore (fiddle around a bit here)
										#Note - we are only making a list of columns to drop, we're not actually dropping them yet


pos_df = df[df.opened == True][[c for c in df.columns if c not in ignore_list]].dropna(how = 'any')
neg_df = df[df.opened == False][[c for c in df.columns if c not in ignore_list]].dropna(how = 'any')

"""
	We wish to split our dataset into two parts, one which contains rows where the email was opened,
	and another, which contains the rows where the email was not opened. Henceforth, they shall be
	referred to as the positive and negative set respectively.

	We perform this split using the pandas equivalent of the SQL select statement.
	It's something like SELECT * FROM df where opened == True for the positive set.

	The pandas equivalent is df[df.opened == True]
	df.opened == True is a parallel comparison, it returns a list of row indices where the given
	boolean condition is satisfied. Note that the column with name "opened" is referred to using df.opened

	A similar expression is used to get rows where the emails are not opened

	Now let's get rid of the unwanted columns specified in the ignore_list.
	[c for c in df.columns if c not in ignore_list] is a list of columns not in the dataframe.
	If this syntax looks alien, please read up on "python list comprehensions"

	Now that we have a list of valid columns, it's easy to extract them from the dataframe

	let valid_cols = [c for c in df.columns if c not in ignore_list]
	then df[valid_cols] returns a data frame with only the valid columns 

	We then want to remove those rows which have some null fields. We could interpolate, but it's easier to
	just drop them from the training set as they comprise a small fraction.
	The dropna() funtion is used for this

"""

pos_vecs = pos_df.as_matrix().astype(np.float32)
neg_vecs = neg_df.as_matrix().astype(np.float32)

"""
	the data now needs to be vectorized.
	Luckily, our data is already in a numerical form, so no feature generation is required.
	If you have data with stuff like categorical variables, you need to vectorize separately
	using a one hot encoder or something

	The as_matrix() function converts a data frame to a raw numpy array. Try this for a small
	dataframe to see how it works. This numpy array is then explicitly type cast to 32bit floats.
	Booleans are converted to 1s and 0s in the process.
"""

cols = pos_df.columns.tolist()

"""
	We'll store a list of the valid columns so that we can use them later to
	reorder the columns in testing data as well
"""

np.random.shuffle(pos_vecs)
np.random.shuffle(neg_vecs) #Always good to shuffle your vectors, as discussed in class.

np.save('pos_vecs', pos_vecs)
np.save('neg_vecs', neg_vecs)   #save the vectors to disk. The .npy extension gets added automatically

print "Training data vectorized"
print pos_vecs.shape, neg_vecs.shape

########################################################

# Repeat similar stuff for the test data

test_df = pd.read_csv('test_dataset.csv')		#read test data

print "test data loaded"

df = test_df.join(users_df, on = 'user_id')		#do the join operation
df = df[cols]									#reorder columns according to the order in training data
												#this automatically gets rid of the columns in ignore_list

df["last_online"].fillna(df["last_online"].mean(), inplace = True)
df["n_open"].fillna(df["n_open"].mean(), inplace = True)
df["n_click"].fillna(df["n_click"].mean(), inplace = True)
df["total"].fillna(df["total"].mean(), inplace = True)
df["unsub"].fillna(df["unsub"].mean(), inplace = True)

#Do some interpolation to fill in the missing values. In this case, the empty fields are filled with the mean.
#You can try all sorts of interpolation and see what results you get

test_vecs = df.as_matrix().astype(np.float32)	#convert dataframe to vectors and typecast to float

np.save('test_vecs', test_vecs)		#save vectors to disk

print test_vecs.shape