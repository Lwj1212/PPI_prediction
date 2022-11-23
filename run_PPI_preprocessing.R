#!/usr/bin/env Rscript
# command line input
library(optparse)

### arg 2 : add_file ---> add_id [tab] add_seq  path
### arg 3 : seq_path ---> reference seqence path
### arg 4 : n_split ---> split number
### arg 5 : type ---> pipr dscript deeptrio

option_list = list(
  make_option(c("-a", "--add_file"), type="character", default=NULL, 
              help="add file path", metavar="character"),
  make_option(c("-r", "--reference"), type="character", default=NULL, 
              help="reference sequence path", metavar="character"),
  make_option(c("-s", "--split"), type="integer", default=NULL, 
              help="split count numver"),
  make_option(c("-t", "--type"), type="character", default="pipr", 
              help="pipr dscript deeptro"),
  make_option(c("-o", "--out"), type="character", default=getwd(), 
              help="save path")
  
)

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

if(length(opt) == 0){
  stop("no paremeter.", call.=FALSE) 
} else {
    suppressPackageStartupMessages({
      library(tidyverse)
    })
    ## Model preprocessing  
    GRCh38_seq_pair_PIPR <- function(add_file, seq_path, out){
      ppi_seq <- read_lines(file = seq_path)
      add_file <- read_delim(file = add_file, delim = "\t", col_names = F, show_col_types = F)
      add_id <- add_file %>% pull(1)
      add_seq <- add_file %>% pull(2)
      
      id_list <- list()
      seq_list <- list()
      temp_seq <- c()
      
      for(index in 1:length(ppi_seq)){
        temp <- ppi_seq[index]
        if(startsWith(temp, ">")){
          temp <- str_replace(temp, ">", "")
          id_list <- append(id_list, temp)
          if(index != 1){
            seq_list <- append(seq_list, paste0(temp_seq, collapse = ""))
            temp_seq <- c()
          }
        } else {
          temp_seq <- c(temp_seq, temp)
        }
      }
      seq_list <- append(seq_list, paste0(temp_seq, collapse = ""))
      seq_list <- seq_list %>% unlist() %>% tibble(seq = .)
      
      
      id_list <- id_list %>% as.character() %>% lapply(X = ., FUN = function(value){
        tmp <- str_split(value, pattern = " ") %>% unlist()
        return(tmp[1])
      }) %>% unlist() %>% tibble(id = .)
      
      id2seq <- bind_cols(id_list, seq_list)
      id2seq <- tibble(id = add_id, seq = add_seq) %>% bind_rows(id2seq) %>% 
        filter(50 < nchar(seq) & nchar(seq) < 1500)
      
      ### pair
      pairs <- id2seq %>% pull(1) %>% lapply(X = ., FUN = function(value){
        tibble(seqA = add_id, seqB = value)
      }) %>% bind_rows()
      pairs <- pairs[-1, ]
      
      
      dir.create(out, showWarnings = F)
      write_delim(id2seq, file = paste0(out, "/PIPR_pairs.fasta"), delim = "\t", col_names = F)
      write_delim(pairs, file = paste0(out, "/PIPR_pairs.tsv"), delim = "\t", col_names = F)
      print("Done!!")
      # return(list(id2seq, pairs))
    }
    GRCh38_seq_pair_DSCRIPT <- function(add_file, seq_path, n_split, out){
      ppi_seq <- read_lines(file = seq_path)
      add_file <- read_delim(file = add_file, delim = "\t", col_names = F, show_col_types = F)
      add_id <- add_file %>% pull(1)
      add_seq <- add_file %>% pull(2)
      
      ### id2seq
      id_list <- list()
      seq_list <- list()
      temp_seq <- c()
      
      for(index in 1:length(ppi_seq)){
        temp <- ppi_seq[index]
        if(startsWith(temp, ">")){
          temp <- str_replace(temp, ">", "")
          id_list <- append(id_list, temp)
          if(index != 1){
            seq_list <- append(seq_list, paste0(temp_seq, collapse = ""))
            temp_seq <- c()
          }
        } else {
          temp_seq <- c(temp_seq, temp)
        }
      }
      seq_list <- append(seq_list, paste0(temp_seq, collapse = ""))
      seq_list <- seq_list %>% unlist() %>% tibble(seq = .)
      
      
      id_list <- id_list %>% as.character() %>% 
        lapply(X = ., FUN = function(value){
          tmp <- str_split(value, pattern = " ") %>% unlist()
          return(tmp[1])
        }) %>% unlist() %>% tibble(id = .)
      
      # print(paste0("id : ", length(id_list)))
      
      id2seq <- bind_cols(id_list, seq_list) %>% 
        filter(50 < nchar(seq) & nchar(seq) < 1500)
      # 17,601
      
      ### pair , split
      pairs <- id2seq %>% pull(1) %>% 
        split(., ceiling(seq_along(.) / n_split))
      
      dir.create(out, showWarnings = F)
      lapply(X = names(pairs), FUN = function(index){
        id2seq_split <- pairs[[index]] %>% lapply(X = ., FUN = function(value){
          temp <- id2seq %>% 
            filter(id == value)
          return(temp)
        }) %>% bind_rows()
        
        pairs_split <- pairs[[index]] %>% 
          lapply(X = ., FUN = function(value){
            tibble(seqA = add_id, seqB = value)
          }) %>% bind_rows()
        
        id2seq_split <- tibble(id = add_id, seq = add_seq) %>% 
          bind_rows(id2seq_split)
        
        id2seq_split <- id2seq_split %>% 
          mutate(id = paste0(">", id)) %>% 
          pivot_longer(id:seq) %>% 
          select(value)
        
        
        write_delim(id2seq_split, file = paste0(out, "/DSCRIPT_pairs_",index,".fasta"), delim = "\t", col_names = F)
        write_delim(pairs_split, file = paste0(out, "/DSCRIPT_pairs_",index,".tsv"), delim = "\t", col_names = F)
        
        # return(list(id2seq_split, pairs_split))
      })
      print("Done!!")
    }
  GRCh38_seq_pair_DeepTrio <- function(add_file, seq_path, out){
    ppi_seq <- read_lines(file = seq_path)
    #
    dir.create(out, showWarnings = F)

    read_delim(file = add_file, delim = "\t", col_names = F, show_col_types = F) %>% 
      mutate(X1 = paste0(">", X1)) %>% 
      gather() %>% 
      select(value) %>% 
      write_delim(file = paste0(out, "/DeepTrio_p1.fasta"), col_names = F) ###### change
    
    ### id2seq
    id_list <- list()
    seq_list <- list()
    temp_seq <- c()
    
    for(index in 1:length(ppi_seq)){
      temp <- ppi_seq[index]
      if(startsWith(temp, ">")){
        id_list <- append(id_list, temp)
        if(index != 1){
          seq_list <- append(seq_list, paste0(temp_seq, collapse = ""))
          temp_seq <- c()
        }
      } else {
        temp_seq <- c(temp_seq, temp)
      }
    }
    seq_list <- append(seq_list, paste0(temp_seq, collapse = ""))
    seq_list <- seq_list %>% unlist() %>% tibble(seq = .)
    
    
    id_list <- id_list %>% as.character() %>% 
      lapply(X = ., FUN = function(value){
        tmp <- str_split(value, pattern = " ") %>% unlist()
        return(tmp[1])
      }) %>% unlist() %>% tibble(id = .)
    
    id2seq <- bind_cols(id_list, seq_list) %>% 
      filter(50 < nchar(seq) & nchar(seq) < 1500)
    # 17,601
    
    for(row in 1:nrow(id2seq)){
      for(col in 1:ncol(id2seq)){
        id2seq[row, col] %>% 
          write_lines(file = paste0(out, "/DeepTrio_p2.fasta"), append = T)
      }}
  }

}

#add_file <- "http://192.168.0.7:8080/ppi/IGSF1_protein.txt"
#seq_path <- "http://192.168.0.7:8080/ppi/Homo_sapiens.GRCh38.cdhit.fa"
add_file <- opt$add_file
seq_path <- opt$reference
n_split <- opt$split
type <- opt$type
out <- opt$out

if(type == "pipr"){
  # PIPR
  GRCh38_seq_pair_PIPR(add_file, seq_path, out)
} else if(type == "dscript"){
  # D-SCRIPT
  GRCh38_seq_pair_DSCRIPT(add_file, seq_path, n_split, out)
} else if(type == "deeptrio"){
  # deeptrio
  GRCh38_seq_pair_DeepTrio(add_file, seq_path, out)
} else{
  print("not to do!")
}



