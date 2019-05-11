import pandas as pd
import numpy as np
import argparse
import random


parser = argparse.ArgumentParser()

parser.add_argument('--planets', type=str, default=[], help="for example: 'earth mars' "
                                                                "(only use spaces between planets")
parser.add_argument('--num_timesteps', type=int, default=49, help='number of time steps for training data')
parser.add_argument('--num-pred_steps', type=int, default=50, help='number of steps you want to predict')
parser.add_argument('--num-train', type=int, default=100, help='number of training examples')
parser.add_argument('--num-valid', type=int, default=20, help='number of validation examples')
parser.add_argument('--num-test', type=int, default=20, help='number of test examples')
parser.add_argument('--suffix', type=str, default='', help='add another name to distinguish versions')
parser.add_argument('--threeD', action='store_true', default=False, help='Creates 3D trajectories')

args = parser.parse_args()


def generate_data(prefix, planets, num_samples, num_steps, suffix, threeD):
    first = True
    planets = planets.split()
    for planet in planets:
        filename = "planet_data/" + planet + ".txt"
        planet_data = pd.read_csv(filename, header=None)
        len_data = len(planet_data)
        # Make batches of specific length for train and test set
        first_sample = True
        for i in range(num_samples):
            start = round(random.random() * (len_data - num_steps))
            end = start + num_steps
            x = planet_data.iloc[start:end, 2]
            y = planet_data.iloc[start:end, 3]
            vx = planet_data.iloc[start:end, 5]
            vy = planet_data.iloc[start:end, 6]

            if threeD:
                z = planet_data.iloc[start:end, 4]
                vz = planet_data.iloc[start:end, 7]
                loc = np.vstack((x, y, z)).T
                loc = loc[None, :, :, None]
                vel = np.vstack((vx, vy, vz)).T
                vel = vel[None, :, :, None]
            else:
                loc = np.vstack((x, y)).T
                loc = loc[None, :, :, None]
                vel = np.vstack((vx, vy)).T
                vel = vel[None, :, :, None]

            if first_sample:
                loc_planet = loc
                vel_planet = vel
                first_sample = False
            else:
                loc_planet = np.concatenate([loc_planet, loc], axis=0)
                vel_planet = np.concatenate([vel_planet, vel], axis=0)

            if (i+1) % 1000 == 0:
                print(prefix + ": number of samples for " + planet + " completed: " + str(i+1))

        if first:
            loc_all = loc_planet
            vel_all = vel_planet
            first = False
        else:
            loc_all = np.concatenate([loc_all, loc_planet], axis=3)
            vel_all = np.concatenate([vel_all, vel_planet], axis=3)

    num_planets = str(len(planets))
    edges = np.zeros((1, len(planets), len(planets)))

    # This makes sure that the earth influences the moon and sun influences all other planets
    if 'earth' in planets:
        if 'moon' in planets:
            edges[:, planets.index('earth'), planets.index('moon')] = 1
    if 'sun' in planets:
        edges[:, planets.index('sun'), :] = 1

    if threeD:
        filename = prefix + '_planets' + num_planets + '_' + suffix + '3D' + '.npy'
    else:
        filename = prefix + '_planets' + num_planets + '_' + suffix + '.npy'

    edges = np.repeat(edges, num_samples, axis=0)
    np.save('./data/loc_' + filename, loc_all)
    np.save('./data/vel_' + filename, vel_all)
    np.save('./data/edges_' + filename, edges)


planets = args.planets
num_test_steps = args.num_timesteps + args.num_pred_steps
generate_data('train', planets, args.num_train, args.num_timesteps, args.suffix, args.threeD)
generate_data('valid', planets, args.num_valid, args.num_timesteps, args.suffix, args.threeD)
generate_data('test', planets, args.num_test, num_test_steps, args.suffix, args.threeD)
