#!/usr/bin/env python3

# Copyright (C) 2023 Codeplay Software Limited
# This work is licensed under the MIT License License.
# For a copy, see https://opensource.org/licenses/MIT.

import sys
import pandas as pd

import matplotlib as mpl
mpl.use('pdf')
from matplotlib import pyplot as plt


_plotStyles = ['o-','s-','D-','^-','v-','H-','P-','*-','X-']
_plotColours = {
    'nvidia':    ['#76b900','#96d921','#518000'],
    'amd-gpu':   ['#dd283d','#f25567','#991c2a'],
    'amd-cpu':   ['#4d4d4d','#808080','#000000'],
    'intel-cpu': ['#0068b5','#2393e6','#004273'],
    'intel-gpu': ['#9448a6','#653171','#b658cc'],
}
_currentColourIndex = dict([(k,0) for k in _plotColours.keys()])


def plotStyle(index):
    return _plotStyles[index % len(_plotStyles)]


def plotColour(key):
    i = _currentColourIndex[key]
    colour = _plotColours[key][i % len(_plotColours[key])]
    _currentColourIndex[key] += 1
    return colour


def getSortedDevices(data):
    devices = {}
    for device in data.device.unique():
        meanFPS = data[data.device==device].compute_fps.mean()
        devices[device]=meanFPS
    return dict(sorted(devices.items(), key=lambda x:x[1], reverse=True)).keys()


def deviceColour(deviceName):
    smallName = deviceName.lower()
    if 'nvidia' in smallName:
        return plotColour('nvidia')
    if 'intel' in smallName:
        if 'graphics' in smallName:
            return plotColour('intel-gpu')
        return plotColour('intel-cpu')
    if 'radeon' in smallName:
        return plotColour('amd-gpu')
    if 'epyc' in smallName:
        return plotColour('amd-cpu')
    return 'gray'


def translateLabel(label):
    deviceDict = {
        'parallel': 'SYCL',
        'sequential': 'sequential C++',
        '11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz': 'Intel Core i9-11900K',
        '12th Gen Intel(R) Core(TM) i9-12900K': 'Intel Core i9-12900K',
        'Intel(R) UHD Graphics 750 [0x4c8a]': 'Intel UHD Graphics 750',
        'Intel(R) UHD Graphics 770': 'Intel UHD Graphics 770',
        'Intel(R) Graphics [0x56a0]': 'Intel Arc A770',
        'AMD EPYC 7402 24-Core Processor': 'AMD EPYC 7402',
        'NVIDIA A100-PCIE-40GB': 'NVIDIA A100',
    }
    backendDict = {
        'cpp, ': '',
        'cuda': 'CUDA',
        'hip': 'HIP',
        'level_zero': 'Level Zero',
        'opencl': 'OpenCL',
    }
    for k,v in deviceDict.items():
        label = label.replace(k,v)
    for k,v in backendDict.items():
        label = label.replace(k,v)
    return label


def main():
    assert len(sys.argv) > 1, f'Usage: {sys.argv[0]} file.csv'

    data = pd.read_csv(sys.argv[1],names=('code_path','backend','device','grid_size','compute_fps'))
    data = data[(data.grid_size>=4)]
    data.device = data.backend + ', ' + data.device
    sortedDevices = getSortedDevices(data)

    deviceCodePath = {}
    for device in sortedDevices:
        codePath = data[data.device==device].code_path.iloc[0]
        deviceCodePath[device] = codePath

    plt.figure(figsize=(10,5))

    for i,device in enumerate(sortedDevices):
        deviceData = data[(data.device==device)]
        label = translateLabel(deviceCodePath[device]) + ', ' + translateLabel(device)
        plt.plot(deviceData.grid_size,
                deviceData.compute_fps,
                plotStyle(i),
                label=label,
                color=deviceColour(device),
                fillstyle='none',
                markersize=7)

    plt.xlabel('Actor grid size')
    plt.ylabel('Compute frames per second')
    plt.title('Collision Simulation Performance')
    plt.xlim(data.grid_size.min()-1, data.grid_size.max()+1)
    plt.ylim(10,1000*(1.5*data.compute_fps.max()//1000))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1.02))
    plt.tight_layout()
    plt.yscale('log')

    plt.savefig('benchmarkResults.png')
    plt.savefig('benchmarkResults.pdf')

    return 0


if __name__=='__main__':
    sys.exit(main())
