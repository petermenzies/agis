#!/bin/bash

uv build --wheel

uv tool install --no-build --python 3.12 dist/*.whl