#!/usr/bin/env python3

# Copyright (C) 2023 Codeplay Software Limited
# This work is licensed under the MIT License License.
# For a copy, see https://opensource.org/licenses/MIT.

import sys
import pandas as pd

import matplotlib as mpl
mpl.use('pdf')
from matplotlib import pyplot as plt


_plotStyles = ['o-','s-','D-','^-','v-','p-']


def plotStyle(index):
    return _plotStyles[index % len(_plotStyles)]


def getSortedDevices(data):
    devices = {}
    for device in data.device.unique():
        meanFPS = data[data.device==device].compute_fps.mean()
        devices[device]=meanFPS
    return dict(sorted(devices.items(), key=lambda x:x[1], reverse=True)).keys()


def deviceColour(deviceName):
    smallName = deviceName.lower()
    if 'nvidia' in smallName:
        return '#76b900'
    if 'intel' in smallName:
        if 'graphics' in smallName:
            return '#894299' # lighter than original '#653171'
        return '#0068b5'
    if 'radeon' in smallName:
        return '#dd283d'
    if 'epyc' in smallName:
        return 'black'
    return 'gray'


def translateLabel(label):
    d = {
        'parallel': 'parallel SYCL',
        'sequential': 'sequential C++',
        '12th Gen Intel(R) Core(TM) i9-12900K': 'Intel Core i9-12900K',
        'Intel(R) UHD Graphics 770': 'Intel UHD Graphics 770',
    }
    return d.get(label,label)


def main():
    assert len(sys.argv) > 1, f'Usage: {sys.argv[0]} file.csv'

    data = pd.read_csv(sys.argv[1],names=('code_path','device','grid_size','compute_fps'))
    data = data[data.grid_size>=5]
    sortedDevices = getSortedDevices(data)

    deviceCodePath = {}
    for device in sortedDevices:
        codePath = data[data.device==device].code_path.iloc[0]
        deviceCodePath[device] = codePath

    for i,device in enumerate(sortedDevices):
        deviceData = data[(data.device==device)]
        label = translateLabel(deviceCodePath[device]) + ', ' + translateLabel(device)
        plt.plot(deviceData.grid_size,
                deviceData.compute_fps,
                plotStyle(i),
                label=label,
                color=deviceColour(device))

    plt.xlabel('Actor grid size')
    plt.ylabel('Compute frames per second')
    plt.title('Collision Simulation Performance')
    plt.ylim(0,1000*(1.2*data.compute_fps.max()//1000))
    plt.legend()
    plt.tight_layout()

    plt.savefig('benchmarkResults.png')
    plt.savefig('benchmarkResults.pdf')

    return 0


if __name__=='__main__':
    sys.exit(main())
