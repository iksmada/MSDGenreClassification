import os
import hdf5_getters
import numpy as np
from glob import glob

# Bring TagTraum's genres into memory
genres = {}
f = open('./data/TagTraum/msd_tagtraum_cd2c.cls', 'r')
for line in f:
	[track_id, track_genre] = line.strip().split(maxsplit=1)
	genres[track_id] = track_genre
f.close

# Feature names
NUM_FEATS = 34
featnames = 'genre,track_id,artist_name,title,loudness,tempo,time_signature,key,mode,duration,'
featnames += 'avg_timbre1,avg_timbre2,avg_timbre3,avg_timbre4,avg_timbre5,avg_timbre6,'
featnames += 'avg_timbre7,avg_timbre8,avg_timbre9,avg_timbre10,avg_timbre11,avg_timbre12,'
featnames += 'var_timbre1,var_timbre2,var_timbre3,var_timbre4,var_timbre5,var_timbre6,'
featnames += 'var_timbre7,var_timbre8,var_timbre9,var_timbre10,var_timbre11,var_timbre12\n'

# Generate output file
output = open('./output.csv', 'w')
output.write(featnames);

# Go through all files in ./data/MSD
files = [y for x in os.walk('./data/MSD') for y in glob(os.path.join(x[0], '*.h5'))]
for file in files:
	filename = os.path.splitext(os.path.basename(file))[0]
	if filename in genres:
		h5 = hdf5_getters.open_h5_file_read(file)

		# Get track's features
		feats = [genres[filename], filename]
		feats.append(hdf5_getters.get_artist_name(h5).decode('utf8').replace(',',''))
		feats.append(hdf5_getters.get_title(h5).decode('utf8').replace(',',''))
		feats.append(str(hdf5_getters.get_loudness(h5)))
		feats.append(str(hdf5_getters.get_tempo(h5)))
		feats.append(str(hdf5_getters.get_time_signature(h5)))
		feats.append(str(hdf5_getters.get_key(h5)))
		feats.append(str(hdf5_getters.get_mode(h5)))
		feats.append(str(hdf5_getters.get_duration(h5)))

		# Calculate timbre avg/var
		timbre = hdf5_getters.get_segments_timbre(h5)
		avg_timbre = np.average(timbre, axis=0)
		for avg in avg_timbre:
		    feats.append(str(avg))
		var_timbre = np.var(timbre, axis=0)
		for var in var_timbre:
		    feats.append(str(var))

		# Close h5 and write into output file
		h5.close()
		assert len(feats) == NUM_FEATS,'feat length problem, len(feats)='+str(len(feats))
		output.write(','.join(feats) + '\n')
output.close()
