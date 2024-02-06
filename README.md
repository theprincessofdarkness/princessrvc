To make princessrvc work, first create a venv (virtual environment):

- Right click in your folder and open a terminal inside of it.
- Type this command: `python -m venv [insert venv name here]`.
- To initialise the venv, type: `.\[venvname]\Scripts\activate`. If this doesn't work, you will need to derestrict your execution-policy in Powershell (https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/set-executionpolicy?view=powershell-7.4). 


After this, your folder should show up with a green **(venv)** to the left of it. Now, we have to install the requirements.:

- Type this command for **Nvidia GPUs**: `python pip install -r .\requirements-cuda.txt`.
- Type this command for **AMD GPUs**: `python pip install -r .\requirements-amd.txt`.
- Type this command for **INTEL GPUs**: `python pip install -r .\requirements-ipex.txt`.

After installing requirements, you are going to need a few files first before Inference or Training:

Hubert: https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt?download=true - **Place this in `assets\hubert`**
RMVPE.py: https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt?download=true - **Place this in `assets\rmvpe`**
Pretrains (For V2): https://mega.nz/file/lfUW1RwR#24sVZMyxDIr5jsh3fPb0x7Jy5becXukycrrJ1xIwsQg - **Place this in `assets\pretrained_v2`**

To start RVC (ensure your venv environment is initialised with `.\[venvname]\Scripts\activate`:

- Type this command: `python .\infer-web.py`.

**Hopefully your RVC should now be running perfectly! Enjoy <3**
