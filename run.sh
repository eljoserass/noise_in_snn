
# maybe shold check the python version we can work with


# check .venv is created, if not create it and install requirements
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi


# -- normal set

# option of only on rgb or only on eb

# option for eval from checkpoint

# option for train
#   check that the preprocessing has been done


# preprocessing -> train -> eval

# -- noisy set 
#