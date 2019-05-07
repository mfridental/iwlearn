#!/usr/bin/env bash
rm dist/*
python setup.py sdist upload -r pypi
