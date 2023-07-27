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
                        choices=[f'{backend}:{device}' for backend in
                                 ['cuda','hip','level_zero','opencl'] for device in
                                 ['gpu','0','1','2','3']] + ['opencl:cpu'],
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
    parser.add_argument('-d','--buildDir', metavar='PATH',
                        default='./build',
                        help='Path to the build dir, default: %(default)s')
    parser.add_argument('--cuda', action='store_true',
                        help='Build including the CUDA target')
    parser.add_argument('--hip', action='store_true',
                        help='Build including the HIP target')
    args = parser.parse_args()

    # Assert flag dependencies
    if (args.cuda or args.hip) and not args.build:
        print('The flags --cuda/--hip can only be used with -b/--build')
        sys.exit(1)

    return args


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
        splitSel = sel.split(':')
        backend = splitSel[0]
        deviceNumber = splitSel[-1] if splitSel[-1].isdecimal() else -1
        deviceType = splitSel[-1] if splitSel[-1].isalpha() else 'gpu'
        formattedSelector = f'{backend}:{deviceType}' + (f':{deviceNumber}' if deviceNumber>=0 else '')
        pattern = re.compile(f'\[.*{formattedSelector}.*\] (.*), (.*) [0-9]+\.[0-9]+ \[.*\]')
        m = re.search(pattern, syclDevices)
        if m is None or len(m.groups()) < 2:
            raise RuntimeError(f'Failed to find "{formattedSelector}" in sycl-ls output')
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


def build(args):
    assert args.gridSize>0, 'Grid size must be a positive number'
    cmd = ' '.join([
        'CC=clang CXX=clang++',
        'cmake',
        '-DHEADLESS=ON',
        f'-DACTOR_GRID_SIZE={args.gridSize}',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DENABLE_CUDA=ON' if args.cuda else '',
        '-DENABLE_HIP=ON' if args.hip else '',
        f'-B{args.buildDir}',
        '-G Ninja',
        '.'
    ])
    out = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if out.returncode != 0:
        raise RuntimeError(f'CMake failed with code {out.returncode}: '
                           f'{cmd}\n{out.stdout.decode("utf-8")}')
    out = subprocess.run(
        f'cmake --build {args.buildDir} -- -j4',
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if out.returncode != 0:
        raise RuntimeError(f'Build failed with code {out.returncode}: '
                           f'{cmd}\n{out.stdout.decode("utf-8")}')

def main():
    args = getArgs()
    numIters = args.numIters
    outputFile = args.outputFile

    if args.build:
        build(args)

    baseCmd = f'{args.buildDir}/collision-sim'

    if args.cpu == 'true':
        backend = 'cpp'
        cpuCmd = f'{baseCmd} --cpu'
        cpuName = getCpuName()
        meanCpu = getMeanOfNRuns(cpuCmd, numIters)
        saveOutput(outputFile, f'sequential,{backend},{cpuName},{args.gridSize},{meanCpu:.2f}')

    if args.gpu:
        gpuNames = getGpuNames(args.gpu)
        for gpuSelector,gpuName in zip(args.gpu,gpuNames):
            backend = gpuSelector.split(':')[0]
            gpuCmd = f'ONEAPI_DEVICE_SELECTOR={gpuSelector} {baseCmd}'
            meanGpu = getMeanOfNRuns(gpuCmd, numIters)
            saveOutput(outputFile, f'parallel,{backend},{gpuName},{args.gridSize},{meanGpu:.2f}')

    return 0


if __name__=='__main__':
    sys.exit(main())
