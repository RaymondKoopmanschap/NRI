import urllib.request
from urllib.parse import quote_plus
import pandas as pd
import argparse
import re

# Documentation of the VALUES given below can be downloaded by pasting the following url in your browser:
# ftp://ssd.jpl.nasa.gov/pub/ssd/horizons_batch_example.long

VALUES = {
    'command': '399',
    'make_ephem': 'YES',
    'table_type': 'VECTOR',
    'center': '500@10',
    'ref_plane': 'FRAME',
    'start_time': '1755-01-01',
    'stop_time': '2000-01-01',
    'step_size': '1 d',
    'quantities': '1',   # Only used when table_type is OBSERVER
    'ref_system': 'J200',
    'out_units':'AU-D',
    'vec_table': '2',  # 1 is position, 2 is state (position + velocity)
    'csv_format': 'YES'
}


BASE_URL="""
https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1&COMMAND='{command}'&MAKE_EPHEM='{make_ephem}'\
&TABLE_TYPE='{table_type}'&START_TIME='{start_time}'&STOP_TIME='{stop_time}'\
&STEP_SIZE='{step_size}'&QUANTITIES='{quantities}'&CSV_FORMAT='{csv_format}'\
&CENTER='{center}'&REF_PLANE='{ref_plane}'&REF_SYTEM='{ref_system}'&OUT_UNITS='{out_units}'\
&VEC_TABLE='{vec_table}'
"""


def main(planet_list):

    planets = planet_list.split()
    # To use more planets see: https://ssd.jpl.nasa.gov/horizons.cgi?s_target=1#top
    planet_dict = {'mercury': '199', 'venus': '299', 'moon': '301', 'earth': '399', 'mars': '499', 'jupiter': '599',
                   'saturn': '699', 'uranus': '799', 'neptune': '899', 'pluto': '999'}

    for planet in planets:
        planet_idx = planet_dict[planet]
        VALUES['command'] = planet_idx

        a = zip(VALUES.keys(), map(quote_plus, VALUES.values()))
        a = {k: v for k, v in a}
        contents = urllib.request.urlopen(BASE_URL.format(**a)).read()

        data = []
        start = False
        for l in contents.decode().splitlines():
            if l == '$$EOE':
                break
            elif l == '$$SOE':
                start = True
                continue

            if start:
                data.append(l)

        data = '\n'.join(data)

        filename = 'planet_data/' + planet + '.txt'
        with open(filename, 'w') as file:
            file.write(data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--planets', type=str, default=[], help="for example: 'earth mars' "
                                                                    "(only use spaces between planets")
    args = parser.parse_args()

    main(args.planet_list)