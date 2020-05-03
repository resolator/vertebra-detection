#!/bin/bash
REPODIR="$(cd "$(dirname "$0")" || exit; pwd -P)"
cd "$REPODIR" || exit;

sudo apt update
sudo apt install -y git-lfs python3-pip libgtk2.0-dev pkg-config
git-lfs pull
sudo -H pip3 install -r requirements.txt
