import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool

# General notes:
# 1 = Asher, 0 means Ben
def read_single_file(path):

	# Initial read
	df = pd.read_csv(path, header=None)

	# The first line is who is serving 
	first_server = 1 if 'a' in df.iloc[0, 0] else 0
	# Alternation us how many consecutive serves a player gets
	alternation = int(df.iloc[0, 2]) 
	# Now we can drop the first line
	df = df.iloc[1:]

	# Add column names, drop useless column
	df.columns = ['point', 'time', 'extra_col']
	df = df.drop('extra_col', axis='columns')

	# Process points, time, add game start time
	df['point'] = df['point'].str.contains('a').astype(int)
	df['time'] = df['time'].astype(float)

	df['gametime'] = int(path.split('.csv')[0].split('/')[-1])

	# Cumulative point information
	df['ash_points'] = df['point'].cumsum()
	df['ben_points'] = (1-df['point']).cumsum()

	# Add information about who is serving
	df['point_num'] = np.arange(0, df.shape[0], 1)
	df = df.set_index('point_num')
	df['server'] = (first_server + df.index // alternation) % alternation

	# Adjust serving information to account for games which go to 
	# over 21, where we alternate serves.
	if df.shape[0] > 40:

		# The first person to serve in the endgame
		first_endgame_server = (first_server + 41 // alternation) % 2

		# Account for switching serves
		inds = df.index[df.index > 40]
		df.loc[inds, 'server'] = (first_endgame_server + inds) % 2

	return df

def process_data():

	# Read and process all the data
	filenames = glob.glob('data/*.csv')
	all_data = []
	for i, filename in enumerate(sorted(filenames)):
		df = read_single_file(filename)
		df['game_no'] = i
		all_data.append(df)
	all_data = pd.concat(all_data, axis='index')
	
	# TODO: we could cache all of the stuff we have currently read
	# and only read new stuff for the future.

	return all_data

if __name__ == '__main__':

	print(process_data())