# -*- coding: utf-8 -*-

from os import mkdir, path
import pandas as pd
import json


# Functions
def save_csv(data, outFile, metadata={}, **kwargs):

    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    if not path.exists(path.dirname(outFile)):
        mkdir(path.dirname(outFile))

    with open(outFile, 'w') as f:
        if metadata:
            f.write('# {}\n'.format(json.dumps(metadata)))
        df.to_csv(
            f,
            mode='a',
            **kwargs
        )

def read_csv(inFile, **kwargs):

    with open(inFile, 'r') as f:
        firstLine = f.readline()

        if firstLine.startswith('#'):
            metadata = json.loads(firstLine.strip()[2:])
            df = pd.read_csv(
                f,
                **kwargs
            )
        else:
            metadata = {}
            # print(firstLine.strip().split(','))
            df = pd.read_csv(
                f,
                names=firstLine.strip().split(','),
                **kwargs
            )

    return df, metadata