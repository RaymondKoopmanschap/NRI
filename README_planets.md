Extended version of NRI with the following options

1 
Load planet data from NASA as text files:
For example load the planets earth and mars

```
python get_planetary_data.py --planets 'earth mars'
```

2 
Convert planet data text files to input of NRI:
Convert earth and mars text files to input NRI.
You can also give arguments of num-train, num-valid and num-test. 
```
python gen_planet_dataset.py --planets 'earth mars'
```
All the specified planets are combined into one dataset which is used for the NRI


If your model is saved under logs/... use that and set --only-testing if you only want to test on this model.
```
python train.py --suffix _planets2 --only-testing --load-folder logs/folder_name
```

