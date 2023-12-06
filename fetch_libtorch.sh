#!/bin/bash

VERSION=2.1.0

curl "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${VERSION}%2Bcpu.zip" -o libtorch.zip
unzip libtorch.zip
