Extended version of NRI with the following options

1. 
# Load planet data from NASA as text files:
# For example load the planets earth and mars

python get_planetary_data.py --planet_list 'earth mars'

2. 
# Convert planet data text files to input of NRI:
# Convert earth and mars text files to input NRI. 

python load_planets.py --planet_list 'earth mars'

# All the specified planets are combined into one dataset which is used for the NRI


# If your model is saved under logs/... use that and set --only-testing to True if you only want to test on this model.
python train.py --num-atoms 2 --suffix _planets --only-testing True --load-folder logs/exp2019-05-06T13\:48\:09.880299/


