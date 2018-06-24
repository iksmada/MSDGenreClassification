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
featnames = ''
featnames += 'genre,track_id,artist_name,title,loudness,tempo,time_signature,key,mode,duration,'
featnames += 'avg_timbre1,avg_timbre2,avg_timbre3,avg_timbre4,avg_timbre5,avg_timbre6,'
featnames += 'avg_timbre7,avg_timbre8,avg_timbre9,avg_timbre10,avg_timbre11,avg_timbre12,'
featnames += 'var_timbre1,var_timbre2,var_timbre3,var_timbre4,var_timbre5,var_timbre6,'
featnames += 'var_timbre7,var_timbre8,var_timbre9,var_timbre10,var_timbre11,var_timbre12,'
featnames += 'min_timbre1,min_timbre2,min_timbre3,min_timbre4,min_timbre5,min_timbre6,'
featnames += 'min_timbre7,min_timbre8,min_timbre9,min_timbre10,min_timbre11,min_timbre12,'
featnames += 'max_timbre1,max_timbre2,max_timbre3,max_timbre4,max_timbre5,max_timbre6,'
featnames += 'max_timbre7,max_timbre8,max_timbre9,max_timbre10,max_timbre11,max_timbre12,'
featnames += 'ptp_timbre1,ptp_timbre2,ptp_timbre3,ptp_timbre4,ptp_timbre5,ptp_timbre6,'
featnames += 'ptp_timbre7,ptp_timbre8,ptp_timbre9,ptp_timbre10,ptp_timbre11,ptp_timbre12,'
featnames += 'mu_timbre1,mu_timbre2,mu_timbre3,mu_timbre4,mu_timbre5,mu_timbre6,'
featnames += 'mu_timbre7,mu_timbre8,mu_timbre9,mu_timbre10,mu_timbre11,mu_timbre12,'
featnames += 'med_timbre1,med_timbre2,med_timbre3,med_timbre4,med_timbre5,med_timbre6,'
featnames += 'med_timbre7,med_timbre8,med_timbre9,med_timbre10,med_timbre11,med_timbre12,'
featnames += 'std_timbre1,std_timbre2,std_timbre3,std_timbre4,std_timbre5,std_timbre6,'
featnames += 'std_timbre7,std_timbre8,std_timbre9,std_timbre10,std_timbre11,std_timbre12,'
featnames += 'avg_pitches1,avg_pitches2,avg_pitches3,avg_pitches4,avg_pitches5,avg_pitches6,'
featnames += 'avg_pitches7,avg_pitches8,avg_pitches9,avg_pitches10,avg_pitches11,avg_pitches12,'
featnames += 'var_pitches1,var_pitches2,var_pitches3,var_pitches4,var_pitches5,var_pitches6,'
featnames += 'var_pitches7,var_pitches8,var_pitches9,var_pitches10,var_pitches11,var_pitches12,'
featnames += 'min_pitches1,min_pitches2,min_pitches3,min_pitches4,min_pitches5,min_pitches6,'
featnames += 'min_pitches7,min_pitches8,min_pitches9,min_pitches10,min_pitches11,min_pitches12,'
featnames += 'max_pitches1,max_pitches2,max_pitches3,max_pitches4,max_pitches5,max_pitches6,'
featnames += 'max_pitches7,max_pitches8,max_pitches9,max_pitches10,max_pitches11,max_pitches12,'
featnames += 'ptp_pitches1,ptp_pitches2,ptp_pitches3,ptp_pitches4,ptp_pitches5,ptp_pitches6,'
featnames += 'ptp_pitches7,ptp_pitches8,ptp_pitches9,ptp_pitches10,ptp_pitches11,ptp_pitches12,'
featnames += 'mu_pitches1,mu_pitches2,mu_pitches3,mu_pitches4,mu_pitches5,mu_pitches6,'
featnames += 'mu_pitches7,mu_pitches8,mu_pitches9,mu_pitches10,mu_pitches11,mu_pitches12,'
featnames += 'med_pitches1,med_pitches2,med_pitches3,med_pitches4,med_pitches5,med_pitches6,'
featnames += 'med_pitches7,med_pitches8,med_pitches9,med_pitches10,med_pitches11,med_pitches12,'
featnames += 'std_pitches1,std_pitches2,std_pitches3,std_pitches4,std_pitches5,std_pitches6,'
featnames += 'std_pitches7,std_pitches8,std_pitches9,std_pitches10,std_pitches11,std_pitches12,'
featnames += 'avg_loudness_max,var_loudness_max,min_loudness_max,max_loudness_max,'
featnames += 'ptp_loudness_max,mu_loudness_max,med_loudness_max,std_loudness_max,'
featnames += 'avg_loudness_max_time,var_loudness_max_time,min_loudness_max_time,max_loudness_max_time,'
featnames += 'ptp_loudness_max_time,mu_loudness_max_time,med_loudness_max_time,std_loudness_max_time,'
featnames += 'avg_loudness_start,var_loudness_start,min_loudness_start,max_loudness_start,'
featnames += 'ptp_loudness_start,mu_loudness_start,med_loudness_start,std_loudness_start,'
featnames += 'hotttnesss,danceability,end_of_fade_in,energy,start_of_fade_out,year\n'
num_feats = len(featnames.split(','))

def get_statistical_feats(feat):
	sf = []
	min = np.amin(feat, axis=0)
	max = np.amax(feat, axis=0)
	ptp = np.ptp(feat, axis=0)
	mu  = np.mean(feat, axis=0)
	avg = np.average(feat, axis=0)
	med = np.median(feat, axis=0)
	std = np.std(feat, axis=0)
	var = np.var(feat, axis=0)
	# Scalars (np.generic) and arrays (np.ndarray) are handled differently
	sf.append(str(avg)) if isinstance(avg, np.generic) else sf.extend([str(k) for k in avg])
	sf.append(str(var)) if isinstance(var, np.generic) else sf.extend([str(k) for k in var])
	sf.append(str(min)) if isinstance(min, np.generic) else sf.extend([str(k) for k in min])
	sf.append(str(max)) if isinstance(max, np.generic) else sf.extend([str(k) for k in max])
	sf.append(str(ptp)) if isinstance(ptp, np.generic) else sf.extend([str(k) for k in ptp])
	sf.append(str(mu))  if isinstance(mu,  np.generic) else sf.extend([str(k) for k in  mu])
	sf.append(str(med)) if isinstance(med, np.generic) else sf.extend([str(k) for k in med])
	sf.append(str(std)) if isinstance(std, np.generic) else sf.extend([str(k) for k in std])
	return sf

def get_feats(h5):
	f = []
	f.append(hdf5_getters.get_artist_name(h5).decode('utf8').replace(',',''))
	f.append(hdf5_getters.get_title(h5).decode('utf8').replace(',',''))
	f.append(str(hdf5_getters.get_loudness(h5)))
	f.append(str(hdf5_getters.get_tempo(h5)))
	f.append(str(hdf5_getters.get_time_signature(h5)))
	f.append(str(hdf5_getters.get_key(h5)))
	f.append(str(hdf5_getters.get_mode(h5)))
	f.append(str(hdf5_getters.get_duration(h5)))
	f.extend(get_statistical_feats(hdf5_getters.get_segments_timbre(h5)))
	f.extend(get_statistical_feats(hdf5_getters.get_segments_pitches(h5)))
	f.extend(get_statistical_feats(hdf5_getters.get_segments_loudness_max(h5)))
	f.extend(get_statistical_feats(hdf5_getters.get_segments_loudness_max_time(h5)))
	f.extend(get_statistical_feats(hdf5_getters.get_segments_loudness_start(h5)))
	f.append(str(hdf5_getters.get_song_hotttnesss(h5)))
	f.append(str(hdf5_getters.get_danceability(h5)))
	f.append(str(hdf5_getters.get_end_of_fade_in(h5)))
	f.append(str(hdf5_getters.get_energy(h5)))
	f.append(str(hdf5_getters.get_start_of_fade_out(h5)))
	f.append(str(hdf5_getters.get_year(h5)))
	return f

# Generate output file
output = open('./output.csv', 'w')
output.write(featnames);

# Go through all files in ./data/MSD
files = [y for x in os.walk('./data/MSD') for y in glob(os.path.join(x[0], '*.h5'))]
for file in files:
	filename = os.path.splitext(os.path.basename(file))[0]

	# Get track's features
	if filename in genres:
		h5 = hdf5_getters.open_h5_file_read(file)
		feats = [genres[filename], filename]
		feats.extend(get_feats(h5))
		# Close h5 and write into output file
		h5.close()
		assert len(feats) == num_feats,'feat length problem, len(feats)='+str(len(feats))
		output.write(','.join(feats) + '\n')
output.close()
