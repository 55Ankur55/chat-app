import sys
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
test_sentence = str(sys.argv[1])
results=[]
data = pd.read_csv("../dataset/ner_dataset.csv", encoding="latin1").fillna(method="ffill")

# print("data readed")
 


tag_values = list(set(data["Tag"].values))
tag_values.append("PAD")
tag_values.sort()
tag2idx = {t: i for i, t in enumerate(tag_values)}

 

import torch
from transformers import BertTokenizer, BertConfig


 


device = torch.device("cpu")


from transformers import BertForTokenClassification, AdamW

 


output_dir = '../output/ner'
model = BertForTokenClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)
model.to(device)
 

model.eval()
input_ids = []
tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence])
with torch.no_grad():
    output = model(input_ids)
label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)


 

# 





train=pd.read_csv("../dataset/train.csv")
labels = train.intent.values
input_labels = []
k = 1;
mp = {}
mpp= {}
for sent in labels:
    mp[sent]=0;
for sent in labels:
    if(mp[sent] == 0):
        mp[sent] = k
        mpp[k]=sent
        k = k + 1
    input_labels.append(mp[sent]-1)

from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, random_split

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

output_dir = '../output/intent/'
model1 = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer1 = BertTokenizer.from_pretrained(output_dir)
model1.to(device)
sentences = [
  test_sentence,
]
input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer1.encode_plus(
                        sent,                      
                        add_special_tokens = True, 
                        max_length = 64,           
                        pad_to_max_length = True,
                        return_attention_mask = True,    
                        return_tensors = 'pt', 
                        truncation=True,     
                )    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

batch_size = 32  

prediction_data = TensorDataset(input_ids, attention_masks )
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
model1.eval()
predictions , true_labels = [], []
for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask  = batch
    with torch.no_grad():
        outputs = model1(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask)

    logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    predictions.append(logits)

# print('    DONE.')

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
intent_output=mpp[flat_predictions[0]+1]
# print(intent_output)


 

database=pd.read_csv("../dataset/database.csv")
database.head()
database.columns



# In[16]:


entities={}
flag = False;
for key, word in zip(new_labels,new_tokens):
  if(key == "B-geo" and flag==False):
    entities["From"] = word	
    flag = True;
  if(key=="B-geo" and flag):
    entities["To"] = word
  if(key == "B-tim" and  word.isdigit()==False):
    entities["Time"] = word
  if(len(word)==5 and word.isdigit()):
      entities["TrainNumber"]=word
  
# print(entities)


# In[17]:


def CallIntent(intent_output,entities):
  switcher = { 
        "Trainlocation": TrainLocation,
        "TrainAvailable": TrainAvailable,
        "GetDistance": GetDistance,
        "TrainRoute": TrainRoute,
        "TrainFare": TrainFare,
        "BookTickets": BookTickets
    } 
  item=switcher.get(intent_output)
  if(callable(item)):
    item(entities)
  # return switcher.get(intent_output,"nothing") 
  # return switcher[intent_output]


# In[18]:


def GetDistance(entities):
  for i,row in database.iterrows():
    if(entities["From"].lower()==row["From"].lower() and entities["To"].lower()==row["To"].lower() and entities["Time"].lower()==row["Time"].lower()):
      results.append("The distance from {} to {} is {}" .format(entities["From"],entities["To"],row["Distance"]))


# In[19]:


def TrainAvailable(entities):
  results.append("Trains Available from {} to {} is" .format(entities["From"],entities["To"]))
  results.append("TrainNo\tTrainName\tTime\tFare")
  for i,row in database.iterrows():
    if(entities["From"].lower()==row["From"].lower() and entities["To"].lower()==row["To"].lower()):
      results.append("{}\t{}\t{}\t{}" .format(row["TrainNumber"],row["TrainName"],row["Time"],row["Fare "]))


# In[20]:


def TrainLocation(entities):
  for i,row in database.iterrows():
    if(int(entities["TrainNumber"])==row["TrainNumber"]):
      results.append("Location of this train is {}" .format(row["Location "]))


# In[21]:


def TrainRoute(entities):
  if "TrainNumber" in entities.keys():
    for i,row in database.iterrows():
      if(int(entities["TrainNumber"])==row["TrainNumber"]):
        results.append("This train runs from {} to {}" .format(row["From"],row["To"]))
    else:
      TrainAvailable(entities)  


# In[22]:


def TrainFare(entities):
  for i,row in database.iterrows():
    if(entities["From"].lower()==row["From"].lower() and entities["To"].lower()==row["To"].lower()):
      results.append("The fare from {} is {} is {}" .format(entities["From"],entities["To"],row["Fare "],))


# In[23]:


def BookTickets(entities):
   TrainAvailable(entities) 
   trainno=input("Input the Train Number For booking tickets ")
   if(len(trainno)==5 and trainno.isdigit()==True):
      for i,row in database.iterrows():
        if(int(trainno)==row["TrainNumber"]):
          results.append("Ticket for {} is Booked Successfully" .format(row["TrainName"]))


# In[24]:


CallIntent(intent_output,entities)

sys.stdout.write(str(results))
# print(str(results))
sys.stdout.flush()
sys.exit(0)

def simulate():
  return results


# In[ ]:





# In[ ]:




