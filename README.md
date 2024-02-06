To make princessrvc work, first create a venv (virtual environment):

- Right click in your folder and open a terminal inside of it.
- Type this command: `python -m venv [insert venv name here]`.
- To initialise the venv, type: `.\[venvname]\Scripts\activate`. If this doesn't work, you will need to derestrict your execution-policy in Powershell (https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/set-executionpolicy?view=powershell-7.4). 


After this, your folder should show up with a green **(venv)** to the left of it. Now, we have to install the requirements.:

- Type this command for **Nvidia GPUs**: `python pip install -r .\requirements-cuda.txt`.
- Type this command for **AMD GPUs**: `python pip install -r .\requirements-amd.txt`.
- Type this command for **INTEL GPUs**: `python pip install -r .\requirements-ipex.txt`.

After installing requirements, you are going to need a few files first before Inference or Training:

To do this simply run the command: `python tools/download_models.py` 

To start RVC (ensure your venv environment is initialised with `.\[venvname]\Scripts\activate`:

- Type this command: `python .\infer-web.py`.

**Hopefully your RVC should now be running perfectly! Enjoy <3**
