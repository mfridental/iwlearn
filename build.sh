#!/usr/bin/env bash
rm dist/*
python setup.py sdist
sudo pip install --upgrade dist/*

