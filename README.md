# PPNE
PPNE is a privacy-preserving network embedding publishing framework which protects private links on the network and retains data utility of embeddings for downstream tasks as much as possibile by iteratively manipulating the network structure.

## Software Environment

- Python 3.7.9. All necessary packages are listed in file requirements.txt
- CUDA 10.1 and cuDNN

## Python Packages Installation

```
pip install -r requirements.txt
```

## How to use

Main codes are stored in the *code* directory, and each code file is named after the corresponding experimential part in the article. Simply run the *.py* files to obtain the results (for example, you could run *code/tradeoff/cora_ppne.py* to obtain the tradeoff experimental results of PPNE on Cora). 

## Large-scale dataset link

- Flickr: http://socialnetworks.mpi-sws.org/data-imc2007.html
