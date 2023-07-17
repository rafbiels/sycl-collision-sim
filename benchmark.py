#!/usr/bin/env python3

# Copyright (C) 2023 Codeplay Software Limited
# This work is licensed under the MIT License License.
# For a copy, see https://opensource.org/licenses/MIT.

import subprocess
import re
import sys
import datetime
from argparse import ArgumentParser


def getArgs():
    parser = ArgumentParser()
    parser.add_argument('--cpu', choices=['true','false'], default='true',
                        help='Execute sequential CPU code, default=%(default)s')
    parser.add_argument('--gpu', metavar='GPU', action='append',
                        choices=['cuda','hip','level_zero','opencl'],
                        help='Execute on GPU with the given selector, '
                             'can be specified multiple times. Choose from: '
                             '%(choices)s')
    parser.add_argument('-g','--gridSize', metavar='SIZE', type=int, default=-1,
                        help='Actor grid size (affects --build and is also '
                             'printed to CSV), default: %(default)s')
    parser.add_argument('-n','--numIters', metavar='N', type=int, default=5,
                        help='Number of iterations per device, '
                             'default: %(default)s')
    parser.add_argument('-o','--outputFile', metavar='NAME',
                        default='collisionSimBenchmark.csv',
                        help='Output CSV file name, default: %(default)s')
    parser.add_argument('-b','--build', action='store_true',
                        help='If enabled, compile the project before executing')

    return parser.parse_args()


def getCpuName():
    return subprocess.run(
        "lscpu | grep 'Model name:' | cut -d ':' -f 2",
        shell=True, capture_output=True).stdout.decode('utf-8').strip()


def getGpuNames(selectors):
    syclDevices = subprocess.run(
        'sycl-ls', shell=True, capture_output=True
        ).stdout.decode('utf-8').strip()
    gpuNames = []
    for sel in selectors:
        pattern = re.compile(f'\[.*{sel}:gpu:0\] (.*), (.*) [0-9]+\.[0-9]+ \[.*\]')
        m = re.search(pattern, syclDevices)
        if m is None or len(m.groups()) < 2:
            raise RuntimeError(f'Failed to find "{sel}" in sycl-ls output')
        gpuNames.append(m.groups()[1])
    return gpuNames


def getMeanOfNRuns(cmd, nRuns):
    resultPatternStr = r'Average compute FPS:\s*([0-9]*\.?[0-9]*)'
    resultPattern = re.compile(resultPatternStr)
    results = []
    for i in range(nRuns):
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        print(f'{timestamp} Start run {i} of: {cmd}')
        out = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        outText = out.stdout.decode('utf-8')
        if out.returncode != 0:
            raise RuntimeError(
                f'Command failed with code {out.returncode}: {cmd}\n{outText}')

        m = re.search(resultPattern, outText)
        if not m or not m.groups():
            raise RuntimeError(
                f'FPS extraction failed from match object {m};'
                f'\nFull output:\n{outText}')
        try:
            fps = float(m.group(1))
        except Exception as e:
            raise RuntimeError(
                f'FPS extraction failed from match object {m} '
                f'with groups {m.groups()};\nFull output:\n{outText}')
        results.append(fps)

    results = sorted(results)
    midResults = results[1:-1] if len(results)>2 else results
    meanResult = sum(midResults)/len(midResults)
    return meanResult


def saveOutput(fname,line,alsoPrint=True):
    with open(fname,'a') as f:
        f.write(line+'\n')
    if alsoPrint:
        print(line)


def build(gridSize):
    assert gridSize>0, 'Grid size must be a positive number'
    cmd = ' '.join([
        'CC=clang CXX=clang++',
        'cmake',
        '-DHEADLESS=ON',
        f'-DACTOR_GRID_SIZE={gridSize}',
        '-DCMAKE_BUILD_TYPE=Release',
        '-B./build',
        '-G Ninja',
        '.'
    ])
    out = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if out.returncode != 0:
        raise RuntimeError(f'CMake failed with code {out.returncode}: '
                           f'{cmd}\n{out.stdout.decode("utf-8")}')
    out = subprocess.run(
        'cmake --build ./build -- -j4',
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if out.returncode != 0:
        raise RuntimeError(f'Build failed with code {out.returncode}: '
                           f'{cmd}\n{out.stdout.decode("utf-8")}')

def main():
    args = getArgs()
    numIters = args.numIters
    outputFile = args.outputFile

    if args.build:
        build(args.gridSize)

    baseCmd = './build/collision-sim'

    if args.cpu == 'true':
        cpuCmd = f'{baseCmd} --cpu'
        cpuName = getCpuName()
        meanCpu = getMeanOfNRuns(cpuCmd, numIters)
        saveOutput(outputFile, f'sequential,{cpuName},{args.gridSize},{meanCpu:.2f}')

    if args.gpu:
        gpuNames = getGpuNames(args.gpu)
        for gpuSelector,gpuName in zip(args.gpu,gpuNames):
            gpuCmd = f'ONEAPI_DEVICE_SELECTOR={gpuSelector}:gpu {baseCmd}'
            meanGpu = getMeanOfNRuns(gpuCmd, numIters)
            saveOutput(outputFile, f'parallel,{gpuName},{args.gridSize},{meanGpu:.2f}')

    return 0


if __name__=='__main__':
    sys.exit(main())
