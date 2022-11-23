import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles,venn3, venn3_circles, venn3_unweighted
import pandas as pd
from functools import reduce
import os
import re

import urllib.parse
import urllib.request

from pathlib import Path
import argparse

def foward_look(value):
    p = re.compile('.+(?=\\.)')
    m = p.search(value)
    return m.group()

def uniprot_retrieve(ensembl):    
#     ensembl_query = "\t".join(list(map(foward_look, ensembl)))
    ensembl_query = "\t".join(ensembl)
    
    url = 'https://www.uniprot.org/uploadlists/'
    params = {
    'from': 'ENSEMBL_PRO_ID',
    'to': 'ID',
    'format': 'tab',
    'query' : ''
    }
    
    params["query"] = ensembl_query
    
    # uniprotKB
    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()
        
    UniProtKB_ID = list(map(lambda x : x.split('\t'), response.decode('utf-8').split('\n')))[1:-1] # first - last row 
    UniProtKB_ID = pd.DataFrame(UniProtKB_ID)
    UniProtKB_ID.columns = ["proteinB", "UniProtKB_ID"]
    
    params["to"] = "SWISSPROT"
    #swissprot
    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()
        
    SWISSPROT = list(map(lambda x : x.split('\t'), response.decode('utf-8').split('\n')))[1:-1] # first - last row 
    SWISSPROT = pd.DataFrame(SWISSPROT)
    SWISSPROT.columns = ["proteinB", "SWISSPROT"]
    
    return pd.merge(UniProtKB_ID, SWISSPROT, on = "proteinB")

if __name__ == "__main__":
  # argpaser
  parser = argparse.ArgumentParser(description='PIPR prediction')
  parser.add_argument('-b', '--base', default='~/gitworking/ppi_prediction',required=True, type=str, help='base path')
  parser.add_argument('-o', '--out', default=os.getcwd(), type=str, help='save path')