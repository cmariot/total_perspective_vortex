#!/bin/sh


# You must source this script to activate the virtualenv in your shell.
# source my_script.sh


ENV_NAME=.venv
BLUE='\033[0;34m'
NC='\033[0m'


# Create a virtualenv based on the ENV_NAME variable.
python3 -m venv $ENV_NAME
echo "${BLUE}${ENV_NAME} virtual environment has been created.${NC}"
echo

# Activate the virtualenv.
source $ENV_NAME/bin/activate
echo "${BLUE}The virtual environment is activated.${NC}"
echo

# Install the requirements in the virtualenv.
echo "${BLUE}Installing the requirements...${NC}"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo

# Show the installed packages.
echo "${BLUE}Installed packages:${NC}"
python3 -m pip freeze
echo

echo "${BLUE}Running the Total Perspective Vortex program...${NC}"
python3 main.py $1
