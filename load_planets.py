import pandas as pd
import numpy as np
import argparse

def planet_data2loc_vel_edges(planets):
    first = True
    planets = planets.split()
    for planet in planets:
        filename = "planet_data/" + planet + ".txt"
        planet_data = pd.read_csv(filename, header=None)
        x = planet_data.iloc[:, 2]
        y = planet_data.iloc[:, 3]
        z = planet_data.iloc[:, 4]
        vx = planet_data.iloc[:, 5]
        vy = planet_data.iloc[:, 6]
        vz = planet_data.iloc[:, 7]
        loc = np.vstack((x, y)).T
        loc = loc[None, :, :, None]
        vel = np.vstack((vx, vy)).T
        vel = vel[None, :, :, None]
        # plt.plot(x, y)
        if first:
            loc_all = loc
            vel_all = vel
            first = False
        else:
            loc_all = np.concatenate([loc_all, loc], axis=3)
            vel_all = np.concatenate([vel_all, vel], axis=3)
    edges = np.zeros((len(planets), len(planets)))

    np.save('./data/loc_train_planets.npy', loc_all)
    np.save('./data/vel_train_planets.npy', vel_all)
    np.save('./data/edges_train_planets.npy', edges)

    np.save('./data/loc_valid_planets.npy', loc_all)
    np.save('./data/vel_valid_planets.npy', vel_all)
    np.save('./data/edges_valid_planets.npy', edges)

    np.save('./data/loc_test_planets.npy', loc_all)
    np.save('./data/vel_test_planets.npy', vel_all)
    np.save('./data/edges_test_planets.npy', edges)
    # plt.show()

    return loc_all, vel_all, edges


parser = argparse.ArgumentParser()
parser.add_argument('--planet_list', type=str, default=[], help="for example: 'earth mars' "
                                                                "(only use spaces between planets")
args = parser.parse_args()

planet_data2loc_vel_edges(args.planet_list)
