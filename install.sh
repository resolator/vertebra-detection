#!/bin/bash
sudo apt update
sudo apt install git-lfs python3-pip
git-lfs pull
sudo -H pip3 install -r requirements.txt