### **Preprocessing**

```
Usage: run_PPI_preprocessing.R [options]


Options:
        -a CHARACTER, --add_file=CHARACTER
                add file path

        -r CHARACTER, --reference=CHARACTER
                reference sequence path

        -s SPLIT, --split=SPLIT
                split count numver for dscript

        -t TYPE, --type=TYPE
                pipr dscript deeptro

        -o OUT, --out=OUT
                save path

        -h, --help
                Show this help message and exit
```


* ```Rscript --vanilla run_PPI_preprocessing.R -a **ADD SEQUENCE** -r **REFERENCE** -t **pipr/dscript/deeptrio** -o PATH```

```
<Example>
- PIPR
cd ppi_prediction
Rscript --vanilla run_PPI_preprocessing.R -a RAW_DATA/IGSF1_protein.txt -r RAW_DATA/Homo_sapiens.GRCh38.cdhit_5.fa -t pipr -o PIPR/preprocessing/

- D-SCRIPT
cd ppi_prediction
Rscript --vanilla run_PPI_preprocessing.R -a RAW_DATA/IGSF1_protein.txt -r RAW_DATA/Homo_sapiens.GRCh38.cdhit_5.fa -t dscript -s 6000 -o D-SCRIPT/preprocessing/

- DeepTrio
cd ppi_prediction
Rscript --vanilla run_PPI_preprocessing.R -a RAW_DATA/IGSF1_protein.txt -r RAW_DATA/Homo_sapiens.GRCh38.cdhit_5.fa -t deeptrio -o DeepTrio/preprocessing
```

<br>

### **PIPR Prediction**

```
PIPR prediction

optional arguments:
  -h, --help            show this help message and exit
  -b PATH, --path PATH  model path
  -t TYPE, --type TYPE  model type : bepler, prottrans_t5u50, baseline
  -d DATABASE, --database DATABASE
                        configuration of the protein sequence database, which
                        contains the protein id and its sequence, and they are
                        splited by table key
  -p PPI, --ppi PPI     configuration of the PPI file, which contains the
                        protein 1 id, protein 2 id and the class they belong
                        to, and they are splited by table key
  -l LOAD, --load LOAD  preprocessing npz file
  -e EMBED, --embed EMBED
                        baseline embedder
```

```
<Example>
cd PIPR
python src/prediction.py -b models/ -t baseline -d preprocessing/PIPR_pairs.fasta -p preprocessing/PIPR_pairs.tsv -e models/ac5_aph.txt -o RESULT/
```


<br>

### **D-SCRIPT Prediction**

```
D-SCRIPT prediction

optional arguments:
  -b preprocessing file directory path
  -r save result
  -d Using GPU 0, not -1
  -m D-SCRIPT human model
```

```
cd D-SCRIPT/
bash src/embed_predict.sh -b preprocessing/ -r RESULT/ -d 0 -m models/human_v1.sav
```

<br>

### **DeepTrio Prediction**

```
usage: main.py [-h] -p1 PROTEIN1 -p2 PROTEIN2 -m MODEL [-o OUTPUT]

run DeepTrio for PPI prediction

optional arguments:
  -h, --help            show this help message and exit
  -p1 PROTEIN1, --protein1 PROTEIN1
                        configuration of the first protein group in fasta
                        format with its path, which can contain multiply
                        sequences
  -p2 PROTEIN2, --protein2 PROTEIN2
                        configuration of the second protein group in fasta
                        format with its path, whcih can contain multiply
                        sequences
  -m MODEL, --model MODEL
                        configuration of the DeepTrio model with its path
  -o OUTPUT, --output OUTPUT
                        configuration of the name of output without a filename
                        extension
```

```
<Example>
cd DeepTrio
python src/main.py -p1 preprocessing/DeepTrio_preprocessing/DeepTrio_p1.fasta -p2 preprocessing/DeepTrio_preprocessing/DeepTrio_p2.fasta -m models/20220318-093520/DeepTrio_search_41.h5 -o RESULT
```