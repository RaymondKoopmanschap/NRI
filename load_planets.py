import pandas as pd
import numpy as np
import argparse

def planet_data2loc_vel_edges(planets):
    first = True
    planets = planets.split()
    for planet in planets:
        filename = "planet_data/" + planet + ".txt"
        planet_data = pd.read_csv(filename, header=None)
        # Split train and test data, for now train and test are the same
        len_train = len(planet_data) // 2
        x_train = planet_data.iloc[0:len_train, 2]
        y_train = planet_data.iloc[0:len_train, 3]
        z_train = planet_data.iloc[0:len_train, 4]
        x_test = planet_data.iloc[:, 2]
        y_test = planet_data.iloc[:, 3]
        z_test = planet_data.iloc[:, 4]
        vx_train = planet_data.iloc[0:len_train, 5]
        vy_train = planet_data.iloc[0:len_train, 6]
        vz_train = planet_data.iloc[0:len_train, 7]
        vx_test = planet_data.iloc[:, 5]
        vy_test = planet_data.iloc[:, 6]
        vz_test = planet_data.iloc[:, 7]
        loc_train = np.vstack((x_train, y_train)).T
        loc_train = loc_train[None, :, :, None]
        loc_test = np.vstack((x_test, y_test)).T
        loc_test = loc_test[None, :, :, None]
        vel_train = np.vstack((vx_train, vy_train)).T
        vel_train = vel_train[None, :, :, None]
        vel_test = np.vstack((vx_test, vy_test)).T
        vel_test = vel_test[None, :, :, None]
        # plt.plot(x, y)
        if first:
            loc_train_all = loc_train
            loc_test_all = loc_test
            vel_train_all = vel_train
            vel_test_all = vel_test
            first = False
        else:
            loc_train_all = np.concatenate([loc_train_all, loc_train], axis=3)
            loc_test_all = np.concatenate([loc_test_all, loc_test], axis=3)
            vel_train_all = np.concatenate([vel_train_all, vel_train], axis=3)
            vel_test_all = np.concatenate([vel_test_all, vel_test], axis=3)
    edges = np.zeros((len(planets), len(planets)))

    num_planets = str(len(planets))
    np.save('./data/loc_train_planets' + num_planets + '.npy', loc_train_all)
    np.save('./data/vel_train_planets' + num_planets + '.npy', vel_train_all)
    np.save('./data/edges_train_planets' + num_planets + '.npy', edges)

    np.save('./data/loc_valid_planets' + num_planets + '.npy', loc_train_all)
    np.save('./data/vel_valid_planets' + num_planets + '.npy', vel_train_all)
    np.save('./data/edges_valid_planets' + num_planets + '.npy', edges)

    np.save('./data/loc_test_planets' + num_planets + '.npy', loc_test_all)
    np.save('./data/vel_test_planets' + num_planets + '.npy', vel_test_all)
    np.save('./data/edges_test_planets' + num_planets + '.npy', edges)
    # plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--planet_list', type=str, default=[], help="for example: 'earth mars' "
                                                                "(only use spaces between planets")
args = parser.parse_args()

planet_data2loc_vel_edges(args.planet_list)
