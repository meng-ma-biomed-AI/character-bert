import pandas as pd
from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation,strip_numeric,remove_stopwords
from os import walk
from os import listdir
from os.path import isfile, join
import numpy as np
import re 
import pickle


import nltk
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

from tqdm import tqdm

# remove the last section that came along from previous split
def remove_section(txt):
  data = txt.split("\n")
  # for i in range(len(data)-1, -1,-1):
  #   print(i,data[i])
  #   if data[i] == "\n":
  #     data = data[0:i]
  #     print("test ",i)
  #     break

  return " ".join(data[:-1])

def remove_noise_text(txt):

  txt = txt.lower()
  txt = re.sub("primary site:", ' ', txt)

  #txt = re.sub('post-surgical changes', ' ', txt.lower()) 

  # Remove any mentions to " Findings were discussed with...."
  txt = txt.split("findings were discussed with")[0] 

  # Remove any other occurance of PI's Information
  txt = txt.split("this study has been reviewed and interpreted")[0] 
  txt = txt.split("this finding was communicated to")[0] 
  txt = txt.split("important findings were identified")[0] 
  txt = txt.split("these findings")[0] 
  txt = txt.split("findings above were")[0] 
  txt = txt.split("findings regarding")[0] 
  txt = txt.split("were discussed")[0] 
  txt = txt.split("these images were")[0] 
  txt = txt.split("important finding")[0] 

  # remove any section headers
  txt = re.sub("post-surgical changes:", ' ', txt)
  txt = re.sub("post surgical changes:", ' ', txt)
  txt = re.sub("primary site:", ' ', txt)
  txt = re.sub("primary site", ' ', txt)
  txt = re.sub("neck:", ' ', txt)
  txt = re.sub("post-treatment changes:", ' ', txt)
  txt = re.sub("post treatment changes:", ' ', txt)
  txt = re.sub("brain, orbits, spine and lungs:", ' ', txt)
  txt = re.sub("primary :", ' ', txt)
  txt = re.sub("neck:", ' ', txt)
  txt = re.sub("aerodigestive tract:", ' ', txt)
  txt = re.sub("calvarium, skull base, and spine:", ' ', txt)
  txt = re.sub("other:", ' ', txt)
  txt = re.sub("upper neck:", ' ', txt)
  txt = re.sub("perineural disease:", ' ', txt)
  txt = re.sub("technique:", ' ', txt)
  txt = re.sub("comparison:", ' ', txt)
  txt = re.sub("paranasal sinuses:", ' ', txt)
  txt = re.sub("included orbits:", ' ', txt)
  txt = re.sub("nasopharynx:", ' ', txt)
  txt = re.sub("tympanomastoid cavities:", ' ', txt)
  txt = re.sub("skull base and calvarium:", ' ', txt)
  txt = re.sub("included intracranial structures:", ' ', txt)
  txt = re.sub("impression:", ' ', txt)
  txt = re.sub("nodes:", ' ', txt)
  txt = re.sub("mri orbits:", ' ', txt)
  txt = re.sub("mri brain:", ' ', txt)
  txt = re.sub("brain:", ' ', txt)
  txt = re.sub("ct face w/:", ' ', txt)
  txt = re.sub("transspatial extension:", ' ', txt)
  txt = re.sub("thyroid bed:", ' ', txt)
  txt = re.sub("additional findings:", ' ', txt)
  txt = re.sub("series_image", ' ', txt) 
  txt = re.sub("series image", ' ', txt)
  txt = re.sub("image series", ' ', txt)
  txt = re.sub("post_treatment", 'post treatment', txt)
  txt = re.sub("post-treatment", 'post treatment', txt)
  txt = re.sub('expected post-treatment changes in the neck without evidence of recurrent disease in the primary site', "",txt)
  
  txt = re.sub("study reviewed", ' ', txt)
  txt = re.sub("study", ' ', txt)
  txt = re.sub("reviewed", ' ', txt)
  txt = re.sub("please see", ' ', txt)
  txt = re.sub("please", ' ', txt)
  txt = re.sub("please see chest ct for further evaluation of known lung mass", ' ', txt)
  txt = re.sub("iia", ' ', txt)
  txt = re.sub("nonmasslike", 'non mass like', txt)
  txt = re.sub("non_mass_like", 'non mass like', txt)
  txt = re.sub("non-mass-like", 'non mass like', txt)
  txt = re.sub("statuspost", 'status post', txt)

  txt = re.sub("image series", ' ', txt)
  txt = re.sub("series image", ' ', txt)
  txt = re.sub("image|images|series|imaging", ' ', txt)
  txt = re.sub("january|february|march|april|may|june|july|august|september|octuber|november|dezember", ' ', txt)
  

  # in the worst case, just replace the name from PI to empty string
  txt = re.sub("dr\\.\\s[^\\s]+", ' ', txt)  
  return txt



# set only_data = True if no need to get scores or if dataaset doesn't have a score
def text_cleaning(data, seeds=None, data_target="all", only_data = False,steam=False, lemma=False,min_lenght=2): # target is all or only neck and primary sections for classification

  if data_target == "all":
    return text_cleaning_all(data, seeds, only_data)

  primary_sentences, neck_sentences, = [], []
  y_p_delete, y_n_delete= [], []

  if only_data == False:
    y_primary = data.iloc[::]['NIRADS_Primary'].to_numpy()
    y_neck = data.iloc[::]['NIRADS_Neck'].to_numpy()

  #data = data.drop(columns=['Exam Name', 'Exam Date', 'Patient EMPI Nbr', 'Unnamed: 0'])

  for i in tqdm(range(data.shape[0]),desc ="Cleaning Data"):
    txt = data.iloc[i]['Radiology Text']
    exam = data.iloc[i]['Exam Name'].lower() # if report does not include neck, then do not add it
    y_p = data.iloc[i]['NIRADS_Primary']
    y_n = data.iloc[i]['NIRADS_Neck']

    if re.search("FINDINGS:", txt):

      txt = txt.split("FINDINGS:")[1].split("Legend")[0]#.replace('\n', ' ')
      txt = txt.lower()

      # # remove the numbers and primary/neck
      # txt = re.sub('primary?\\s?:?\\s\\d|neck?\\s?:?\\s\\d', "", txt)
      
      #print("\n\n",i,txt,"\n\n")
      if (re.search("primary site:", txt)):
        d_p = txt.split("primary site:")[1].split(":")[0].lower() # split until find next section
      else:
        d_p = txt.split(":")[0].lower()

      primary_data = remove_section(d_p)  # remove the last section that came along from previous split

      if re.search("primary :", txt):
        d_p = txt.split("primary :")[1].split(":")[0].lower() 
      else:
        d_p = txt.split("impression:")[1].split(":")[0].lower() # use whole impression since neck is not presented
      
      primary_data += remove_section(d_p) # " ".join(d_p.split(" ")[:-1])
      primary_data = re.sub('\\d', "", primary_data)
      primary_data = re.sub('-{2,}', "", primary_data) # remove dashes
      primary_data = re.sub(' +', ' ', primary_data) # remove double+ spaces


      if y_p !=0:      
        primary_sentences.append(remove_noise_text(primary_data))
      else:
        y_p_delete.append(i)


      if re.search("neck", exam) and re.search("neck:", txt):
        
        #print("\n\n",i,txt,"\n\n")
        aux = txt.split("impression:")
        if re.search("neck:", aux[0]):
          d_n = aux[0].split("neck:")[1].split(":")[0].lower()
        else:
          d_n = txt.split(":")[0].lower()

        neck_data = remove_section(d_n) #" ".join(d_n.split(" ")[:-1])

        if re.search("neck:", aux[1]):
          d_n = aux[1].split("neck:")[1].split(":")[0].lower()
        else:
          aux = re.sub("primary :", "", aux[1]) # make sure to remove the primary, to don't get garbage
          d_n = aux.split(":")[0].lower()

        neck_data += remove_section(d_n) #" ".join(d_n.split(" ")[:-1])

        neck_data = re.sub('\\d', "", neck_data)
        neck_data = re.sub('-{2,}', "", neck_data) # remove dashes
        neck_data = re.sub(' +', ' ', neck_data) # remove double+ spaces

        if y_n !=0:      
          neck_sentences.append(remove_noise_text(neck_data))
        else:
          y_n_delete.append(i)
          
      else:
        y_n_delete.append(i) # remove neck report for the classification


    elif only_data == False:
      y_p_delete.append(i)
      y_n_delete.append(i)

  print("\n\tDone Cleaning Data")
  if only_data == False:
    y_primary = np.delete(y_primary, y_p_delete, 0)
    y_neck = np.delete(y_neck, y_n_delete, 0)
    return primary_sentences, neck_sentences, y_primary, y_neck

  else:
    return primary_sentences, neck_sentences


if __name__ == '__main__':
  exit(1)