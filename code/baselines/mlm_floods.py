import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast, BertConfig, AutoTokenizer, BertTokenizer, XLMRobertaConfig, XLMRobertaTokenizer, AutoModelWithLMHead, XLMConfig, AutoModelForMaskedLM, RobertaForSequenceClassification
import sys, statistics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm, trange
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
import copy
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import pickle, time
from transformers import AutoTokenizer, AutoModelForMaskedLM 
device = torch.device("cuda")


class mBERT(nn.Module):

    def __init__(self, bert, max_length, num_classes, hidden_layers, drop=0.1):
      
      super(mBERT, self).__init__()

      self.bert = bert
    #define the forward pass
    def forward(self, sent_id_1, mask_1, labels=None):

      if labels is None:
          out = self.bert(sent_id_1, 
              token_type_ids=None, 
              attention_mask=mask_1)
      else:
          out = self.bert(sent_id_1, 
              token_type_ids=None, 
              attention_mask=mask_1, 
              labels=labels)

      return out[0]


def is_changing(arr, delta):
    curr = arr[0]
    count = len(arr)
    for x in arr[1:]:
        if x > curr:
            count -= 1
            curr = x
    if count < 3:
        return False

    new_list = set(arr)
    new_list.remove(max(new_list))
    curr = max(new_list)
    count = len(arr)
    for x in arr:
        if abs(x-curr) < delta:
          count -= 1
    if count < 2:
        return False

    return True

def tokenize_and_preserve_labels(sentence, sent_labels):
    tokenized_sentence = []
    labels = []

    sentence = [x for x in sentence if not x.isdigit()]
    sentence = sentence[:500]

    full_sentence = ' '.join(sentence)
    encoded = tokenizer.encode_plus(
        text=full_sentence,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length = 30,  # maximum length of a sentence
        padding=True,  # Add [PAD]s
        return_attention_mask = True,  # Generate the attention mask
        return_tensors = 'pt',  # ask the function to return PyTorch tensors
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'].squeeze())
  
    return encoded, sent_labels, tokens

def tokenize(sentence):
    tokenized_sentence = []

    for word in sentence:

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

    return tokenized_sentence

# tokenize and encode sequences in the training set
def tokenize_and_encode(split, tokenizer, max_length):
    return tokenizer.batch_encode_plus(
        split.tolist(),
        max_length = max_length,
        pad_to_max_length=True,
        truncation=True
    )

def getData4Bert(FNAME1, FNAME2, tokenizer, MAX_LEN=None):

    POS_LINES = []
    f = open(FNAME1)
    POS_LINES = f.readlines()
    f.close()
    POS_LINES = [x.strip().split()[:200] for x in POS_LINES]

    NEG_LINES = []
    f = open(FNAME2)
    NEG_LINES = f.readlines()
    f.close()
    NEG_LINES = [x.strip().split()[:200] for x in NEG_LINES]

    sentences = POS_LINES + NEG_LINES
    labels = [1]*len(POS_LINES) + [0]*len(NEG_LINES)

    seq_len = [len(sentence) for sentence in sentences]

    if MAX_LEN is None:
        MAX_LEN = int(max(seq_len))


    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, slabs)
        for sent, slabs in zip(sentences, labels)
    ]

    sent_labels = labels

    tokenized_texts = [token_label_pair[2] for token_label_pair in tokenized_texts_and_labels]
    input_ids = [token_label_pair[0]['input_ids'].squeeze() for token_label_pair in tokenized_texts_and_labels]
    #attention_masks = [token_label_pair[0]['attention_mask'].squeeze() for token_label_pair in tokenized_texts_and_labels]
    sent_labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    input_ids = pad_sequences([input_id for input_id in input_ids],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]


    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(sent_labels), MAX_LEN

############### LOADER METHOD ##########################
def create_loaders(seq, mask, y):
  # wrap tensors
  data = TensorDataset(seq, mask, y)
  # sampler for sampling the data during training
  sampler = RandomSampler(data)
  # dataLoader for train set
  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
  return dataloader

############### FINE TUNE ##########################

# function to train the model
def train(model, train_dataloader, optimizer, schedular, alpha=1.0, beta=1.0):
  
    model.train()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds=[]
  
    # iterate over batches
    steps = []
    batches = []

    for step,batch in enumerate(train_dataloader):
        steps.append(step)
        batches.append(batch)

    for index in range(len(steps)):
        step = steps[index]
        batch = batches[index]
    
        # progress update after every 50 batches.
        #if step % 50 == 0 and not step == 0:
            #print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        #batch = [r for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients 
        model.zero_grad()       

        # get model predictions for the current batch
        predsA = model(sent_id, mask, labels)

        # compute the loss between actual and predicted values
        lossA = alpha*predsA #cross_entropy(predsA, labels)
        total_loss = total_loss + lossA.item()

        # backward pass to calculate the gradients
        lossA.backward() #retain_graph=True)

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()
        schedular.step()

        # model predictions are stored on GPU. So, push it to CPU
        predsA=predsA.detach().cpu().numpy()
        '''
        for index in range(len(slots_preds)):
            slots_preds[index]=slots_preds[index].detach().cpu().numpy()
        '''
        # append the model predictions
        #total_preds.append(predsA)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    #total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return model, avg_loss, total_preds


# function for evaluating the model
# function for evaluating the model
def evaluate(model, val_dataloader, alpha=1.0, beta=1.0):
  
    #print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        #if step % 500 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)
               
            # Report progress.
            #print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        #batch = [t for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            predsA = model(sent_id, mask)

            #preds = preds.squeeze()
            #print("preds shape: ", preds.shape)

            # compute the validation loss between actual and predicted values

            lossA = alpha*cross_entropy(predsA,labels)
            total_loss = total_loss + lossA.item()
                
            predsA = predsA.detach().cpu().numpy()
            #total_preds.append(predsA)
            #outputs=outputs.detach().cpu().numpy()

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    #total_preds  = np.concatenate(total_preds, axis=0)

    return model, avg_loss, total_preds

def final_model(train_FNAME1, train_FNAME2, val_FNAME1, val_FNAME2, mode='ml'):
    global MAX_LEN, SCRATCH_FNAME, base_model_name, FREEZE, SAVE

    ############### LOAD DATASET ##########################
    train_seq, train_mask, train_y, MAX_LEN = getData4Bert(train_FNAME1, train_FNAME2, tokenizer,MAX_LEN=MAX_LEN)
    val_seq, val_mask, val_y, _ = getData4Bert(val_FNAME1, val_FNAME2, tokenizer, MAX_LEN)

    TEST_DATA = []
    if mode == 'ml':
        td = getData4Bert('data/appen/kf_pos', 'data/appen/kf_neg', tokenizer, MAX_LEN)
        TEST_DATA.append(td[:3])
    if mode == 'hi':
        td = getData4Bert('data/appen/ni_pos', 'data/appen/ni_neg2', tokenizer, MAX_LEN)
        TEST_DATA.append(td[:3])

    train_dataloader = create_loaders(train_seq, train_mask, train_y)
    val_dataloader = create_loaders(val_seq, val_mask, val_y)


    ############### LOAD BERT ##########################
    bert = None
    if base_model_name == "xlm-roberta-base":
        bert = RobertaForSequenceClassification.from_pretrained(base_model_name, num_labels = 2, output_attentions = False, output_hidden_states = True, return_dict=True)
    else:
        bert = BertForSequenceClassification.from_pretrained(base_model_name, num_labels = 2, output_attentions = False, output_hidden_states = True, return_dict=True)

    ############### FREEZE ##########################
    if FREEZE == 0:
        pass
    else:
        if base_model_name == "xlm-roberta-base":
            for param in bert.roberta.embeddings.parameters():
                param.requires_grad = False
            for index in range(len(bert.roberta.encoder.layer)):
                if index < FREEZE:
                    for param in bert.roberta.encoder.layer[index].parameters():
                        param.requires_grad = False
        else:
            for param in bert.bert.embeddings.parameters():
                param.requires_grad = False
            for index in range(len(bert.bert.encoder.layer)):
                if index < FREEZE:
                    for param in bert.bert.encoder.layer[index].parameters():
                        param.requires_grad = False


    ############### LOAD BERT ##########################
    model = mBERT(bert, MAX_LEN, 2, hidden_layers)

    # push the model to GPU
    model = model.to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr = lr)          # learning rate

    schedular = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader)*epochs,
    )

    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]
    best_epoch = 0

    start_time = time.time()
    #for each epoch
    for epoch in range(epochs):
        
        #print('\nEpoch {:} / {:}'.format(epoch + 1, epochs), end=" ")
        
        #train model
        model, train_loss, _ = train(model, train_dataloader, optimizer, schedular)
        
        #evaluate model
        model, valid_loss, _ = evaluate(model, val_dataloader)
        
        #save the best model
        if valid_loss < best_valid_loss and epoch > 5:
            best_valid_loss = valid_loss
            if SAVE == 1:
                torch.save(model.state_dict(), SCRATCH_FNAME+".pt")
            best_epoch = epoch+1
        
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        #print(f'Training Loss: {train_loss:.5f} Validation Loss: {valid_loss:.5f}')

        if epoch > patience:
            if not is_changing(valid_losses[-1*patience:], delta):
                #print("STOPPING - NO CHANGE")
                break

    #print('BEST EPOCH= ', best_epoch)
    print("--- %s seconds ---" % (time.time() - start_time))
    ############### predict ########################## 

    accuracy_scores = []
    f1_scores = []

    for index in range(1):
        test_seq, test_mask, test_y = TEST_DATA[index]
        with torch.no_grad():
            preds = model(test_seq.to(device), test_mask.to(device))
            preds = preds.detach().cpu().numpy()

        preds = np.argmax(preds, axis = 1)=
        acc = accuracy_score(test_y, preds)
        f1micro = f1_score(test_y, preds, average='micro')
        f1macro = f1_score(test_y, preds, average='macro')
        f1weighted = f1_score(test_y, preds, average='weighted')
        acc = round(acc*100,2)
        f1 = round(f1weighted*100,2)
        accuracy_scores.append(acc)
        f1_scores.append(f1)

    return accuracy_scores, f1_scores

##########################
#define a batch size
cross_entropy  = nn.CrossEntropyLoss()

alpha = 1.0
beta = 1.0

hidden_layers = 768
# number of training epochs
epochs = 40
delta = 5e-5
lr = 5e-5
patience = 10 #int(sys.argv[2]) ###
batch_size = 4
target = str(sys.argv[1]) #ml, hi
model = str(sys.argv[2]) #en, trt, tlt, combo
base_model_name = str(sys.argv[3])
FREEZE = 8 #int(sys.argv[4])
num_runs= 5 #int(sys.argv[5])
SAVE = 0 #int(sys.argv[6])
#run_name = str(sys.argv[7])
MAX_LEN = 30

############### CREATE TOKENIZER ##########################
 
tokenizer = AutoTokenizer.from_pretrained(base_model_name) #bert-base-multilingual-uncased


mode = 'mgl'
if target == 'ml':
    mode = 'mgl'
if target == 'hi':
    mode = 'hgl'

transliterated_acc = []
transliterated_f1 = []

print("target=", target)
print("model=", model)
print("base_model_name=", base_model_name)
print("FREEZE=", FREEZE)

for index in range(num_runs):

    accuracy_scores, f1_scores = [], []
    if model == 'en':
        accuracy_scores, f1_scores = final_model('data/appen/appen_en_train_pos', 'data/appen/appen_en_train_neg', 'data/appen/appen_en_val_pos', 'data/appen/appen_en_val_neg', mode=target)
    if model == 'trt':
        accuracy_scores, f1_scores = final_model('data/appen/appen_en_train_pos_'+target, 'data/appen/appen_en_train_neg_'+target, 'data/appen/appen_en_val_pos_'+target, 'data/appen/appen_en_val_neg_'+target, mode=target)
    if model == 'tlt':
        accuracy_scores, f1_scores = final_model('data/appen/appen_en_train_pos_'+target+'_'+mode, 'data/appen/appen_en_train_neg_'+target+'_'+mode, 'data/appen/appen_en_val_pos_'+target+'_'+mode, 'data/appen/appen_en_val_neg_'+target+'_'+mode, mode=target)
    if model == 'combo':
        accuracy_scores, f1_scores = final_model('data/appen/appen_en_train_pos_combo_'+target, 'data/appen/appen_en_train_neg_combo_'+target, 'data/appen/appen_en_val_pos_combo_'+target, 'data/appen/appen_en_val_neg_combo_'+target, mode=target)
    transliterated_acc.append(accuracy_scores[0])
    transliterated_f1.append(f1_scores[0])

print("-----average-----")
print("Acc= ", np.mean(translated_acc))
print("F1= ", np.mean(translated_f1))
