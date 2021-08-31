import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast, BertConfig, AutoTokenizer, BertTokenizer, XLMRobertaConfig, XLMRobertaTokenizer, AutoModelWithLMHead, XLMConfig, AutoModelForMaskedLM
import sys, statistics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm, trange
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
#from seqeval.metrics import f1_score, accuracy_score
import copy
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import pickle, time
from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaForSequenceClassification
device = torch.device("cuda")

class mBERT(nn.Module):

    def __init__(self, bert_teacher, bert_student, num_classes, hidden_layers, drop=0.1):
      
      super(mBERT, self).__init__()

      self.bert_teacher = bert_teacher
      self.bert_student = bert_student
      self.sigmoid = nn.Sigmoid()
      self.cosine = nn.CosineSimilarity()
      #self.relu =  nn.ReLU()
      #self.dropout = nn.Dropout(p=drop, inplace=False)

    #define the forward pass
    def forward(self, sent_id_1, mask_1, sent_id_2, mask_2, sent_id_3, mask_3, labels=None, TEST = False):
        global semi_sup
        global gamma

        if TEST:
            out = self.bert_student(sent_id_1, token_type_ids=None, attention_mask=mask_1)
            return out[0]

        res = 0
        if semi_sup:
            out_teacher = self.bert_teacher(sent_id_1, token_type_ids=None, attention_mask=mask_1)
            out1 = self.bert_student(sent_id_1, token_type_ids=None, attention_mask=mask_1)
            out2 = self.bert_student(sent_id_2, token_type_ids=None, attention_mask=mask_2)
            out3 = self.bert_student(sent_id_3, token_type_ids=None, attention_mask=mask_3)
            ot = out_teacher.hidden_states[-1][:,0,:]
            a = out1.hidden_states[-1][:,0,:]
            b = out2.hidden_states[-1][:,0,:]
            c = out3.hidden_states[-1][:,0,:]
            output = 1-self.cosine(ot,a) + 1-self.cosine(ot,b) + 1-self.cosine(ot,c)
            res = torch.mean(output) 

        if labels is None:
            out1 = self.bert_student(sent_id_1, token_type_ids=None, attention_mask=mask_1)
            out2 = self.bert_student(sent_id_2, token_type_ids=None, attention_mask=mask_2)
            out3 = self.bert_student(sent_id_3, token_type_ids=None, attention_mask=mask_3)
            return res, out1[0], out2[0], out3[0]
        else:
            out1 = self.bert_student(sent_id_1, token_type_ids=None, attention_mask=mask_1, labels=labels)
            out2 = self.bert_student(sent_id_2, token_type_ids=None, attention_mask=mask_2, labels=labels)
            out3 = self.bert_student(sent_id_3, token_type_ids=None, attention_mask=mask_3, labels=labels)
            return gamma*res + out1[0] + out2[0] + out3[0]

        return None

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

def tokenize_and_preserve_labels(sentence1, sent_labels1, sentence2, sent_labels2, sentence3, sent_labels3):
    encoded1, tokens1 = tokenize(sentence1)
    encoded2, tokens2 = tokenize(sentence2)
    encoded3, tokens3 = tokenize(sentence3)
    
    return encoded1, sent_labels1, tokens1, encoded2, sent_labels2, tokens2, encoded3, sent_labels3, tokens3

def tokenize(sentence):
    sentence = [x for x in sentence if not x.isdigit()]
    sentence = sentence[:500]

    full_sentence = ' '.join(sentence)
    encoded = tokenizer.encode_plus(
        text=full_sentence,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length = 45,  # maximum length of a sentence
        padding=True,  # Add [PAD]s
        return_attention_mask = True,  # Generate the attention mask
        return_tensors = 'pt',  # ask the function to return PyTorch tensors
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'].squeeze())

    return encoded, tokens

# tokenize and encode sequences in the training set
def tokenize_and_encode(split, tokenizer, max_length):
    return tokenizer.batch_encode_plus(
        split.tolist(),
        max_length = max_length,
        pad_to_max_length=True,
        truncation=True
    )

def getData4Bert(FNAME1, FNAME2, FNAME3, FNAME4, FNAME5, FNAME6, tokenizer, MAX_LEN=None):
    global target
    if target == 'ml':
        maxl = 45
    if target == 'hi':
        maxl = 45

    POS_LINES = []
    f = open(FNAME1)
    POS_LINES = f.readlines()
    f.close()
    POS_LINES = [x.strip().split()[:maxl] for x in POS_LINES]

    NEG_LINES = []
    f = open(FNAME2)
    NEG_LINES = f.readlines()
    f.close()
    NEG_LINES = [x.strip().split()[:maxl] for x in NEG_LINES]

    sentences1 = POS_LINES + NEG_LINES
    labels1 = [1]*len(POS_LINES) + [0]*len(NEG_LINES)

    POS_LINES = []
    f = open(FNAME3)
    POS_LINES = f.readlines()
    f.close()
    POS_LINES = [x.strip().split()[:maxl] for x in POS_LINES]

    NEG_LINES = []
    f = open(FNAME4)
    NEG_LINES = f.readlines()
    f.close()
    NEG_LINES = [x.strip().split()[:maxl] for x in NEG_LINES]

    sentences2 = POS_LINES + NEG_LINES
    labels2 = [1]*len(POS_LINES) + [0]*len(NEG_LINES)

    POS_LINES = []
    f = open(FNAME5)
    POS_LINES = f.readlines()
    f.close()
    POS_LINES = [x.strip().split()[:maxl] for x in POS_LINES]

    NEG_LINES = []
    f = open(FNAME6)
    NEG_LINES = f.readlines()
    f.close()
    NEG_LINES = [x.strip().split()[:maxl] for x in NEG_LINES]

    sentences3 = POS_LINES + NEG_LINES
    labels3 = [1]*len(POS_LINES) + [0]*len(NEG_LINES)


    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent1, slabs1, sent2, slabs2, sent3, slabs3)
        for sent1, slabs1, sent2, slabs2, sent3, slabs3  in zip(sentences1, labels1, sentences2, labels2, sentences3, labels3)
    ]

    tokenized_texts1 = [token_label_pair[2] for token_label_pair in tokenized_texts_and_labels]
    sent_labels1 = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    tokenized_texts2 = [token_label_pair[5] for token_label_pair in tokenized_texts_and_labels]
    sent_labels2 = [token_label_pair[4] for token_label_pair in tokenized_texts_and_labels]
    tokenized_texts3 = [token_label_pair[8] for token_label_pair in tokenized_texts_and_labels]
    sent_labels3 = [token_label_pair[7] for token_label_pair in tokenized_texts_and_labels]

    input_ids1 = [token_label_pair[0]['input_ids'].squeeze() for token_label_pair in tokenized_texts_and_labels]
    input_ids2 = [token_label_pair[3]['input_ids'].squeeze() for token_label_pair in tokenized_texts_and_labels]
    input_ids3 = [token_label_pair[6]['input_ids'].squeeze() for token_label_pair in tokenized_texts_and_labels]
   
    input_ids1 = pad_sequences([input_id for input_id in input_ids1],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

    input_ids2 = pad_sequences([input_id for input_id in input_ids2],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

    input_ids3 = pad_sequences([input_id for input_id in input_ids3],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

    attention_masks1 = [[float(i != 0.0) for i in ii] for ii in input_ids1]
    attention_masks2 = [[float(i != 0.0) for i in ii] for ii in input_ids2]
    attention_masks3 = [[float(i != 0.0) for i in ii] for ii in input_ids3]

    assert(len(labels1)==len(labels2)==len(labels3))

    print("SHAPE= ", torch.tensor(input_ids1).shape)

    return torch.tensor(input_ids1), torch.tensor(attention_masks1), torch.tensor(input_ids2), torch.tensor(attention_masks2), torch.tensor(input_ids3), torch.tensor(attention_masks3), torch.tensor(labels3)

############### LOADER METHOD ##########################
def create_loaders(seq1, mask1, seq2, mask2, seq3, mask3, y):
  # wrap tensors
  data = TensorDataset(seq1, mask1, seq2, mask2, seq3, mask3, y)
  # sampler for sampling the data during training
  sampler = RandomSampler(data)
  # dataLoader for train set
  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
  return dataloader

############### FINE TUNE ##########################

# function to train the model
def train(train_dataloader, optimizer, schedular, alpha=1.0, beta=1.0):
    global model
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

        sent_id1, mask1, sent_id2, mask2, sent_id3, mask3, labels = batch

        # clear previously calculated gradients 
        model.zero_grad()       

        # get model predictions for the current batch
        predsA = model(sent_id1, mask1, sent_id2, mask2, sent_id3, mask3, labels)

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
    return avg_loss, total_preds


# function for evaluating the model
# function for evaluating the model
def evaluate(val_dataloader, alpha=1.0, beta=1.0):
    global model
    global gamma
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

        sent_id1, mask1, sent_id2, mask2, sent_id3, mask3, labels = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            semi_loss, predsA, predsB, predsC  = model(sent_id1, mask1, sent_id2, mask2, sent_id3, mask3)

            #preds = preds.squeeze()
            #print("preds shape: ", preds.shape)

            # compute the validation loss between actual and predicted values

            lossA = alpha*(cross_entropy(predsA,labels) + cross_entropy(predsB,labels) + cross_entropy(predsC,labels))
            
            if type(semi_loss) == int:
                total_loss = total_loss + lossA.item() + gamma*semi_loss
            else:
                total_loss = total_loss + lossA.item() + gamma*semi_loss.item()
                semi_loss = semi_loss.detach().cpu().numpy()  
            predsA = predsA.detach().cpu().numpy()
            predsB = predsB.detach().cpu().numpy()
            predsC = predsC.detach().cpu().numpy()
            #total_preds.append(predsA)
            #outputs=outputs.detach().cpu().numpy()

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    #total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

############### FINE TUNE ##########################
def final_model(train_FNAME1, train_FNAME2, train_FNAME3, train_FNAME4, train_FNAME5, train_FNAME6, val_FNAME1, val_FNAME2, val_FNAME3, val_FNAME4, val_FNAME5, val_FNAME6, mode='ml'):
    global model, base_model_name
    FREEZE = 8
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

    ############### LOAD DATASET ##########################
    MAX_LEN = 45
    train_seq1, train_mask1, train_seq2, train_mask2, train_seq3, train_mask3, train_y  = getData4Bert(train_FNAME1, train_FNAME2, train_FNAME3, train_FNAME4, train_FNAME5, train_FNAME6, tokenizer,MAX_LEN=MAX_LEN)
    val_seq1, val_mask1, val_seq2, val_mask2, val_seq3, val_mask3, val_y = getData4Bert(val_FNAME1, val_FNAME2, val_FNAME3, val_FNAME4, val_FNAME5, val_FNAME6, tokenizer, MAX_LEN=MAX_LEN)

    TEST_DATA = []
    if mode =='ml':
        td = getData4Bert('data/appen/kf_pos', 'data/appen/kf_neg', 'data/appen/kf_pos', 'data/appen/kf_neg', 'data/appen/kf_pos', 'data/appen/kf_neg', tokenizer, MAX_LEN)
        TEST_DATA.append(td[4:])
    if mode == 'hi':
        td = getData4Bert('data/appen/ni_pos', 'data/appen/ni_neg', 'data/appen/ni_pos', 'data/appen/ni_neg', 'data/appen/ni_pos', 'data/appen/ni_neg', tokenizer, MAX_LEN)
        TEST_DATA.append(td[4:])

    train_dataloader = create_loaders(train_seq1, train_mask1, train_seq2, train_mask2, train_seq3, train_mask3, train_y)
    val_dataloader = create_loaders(val_seq1, val_mask1, val_seq2, val_mask2, val_seq3, val_mask3, val_y)

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
        train_loss, _ = train(train_dataloader, optimizer, schedular)
        
        #evaluate model
        valid_loss, _ = evaluate(val_dataloader)
        
        #save the best model
        if valid_loss < best_valid_loss and epoch > 5:
            best_valid_loss = valid_loss
            #torch.save(model.state_dict(), SCRATCH_FNAME)
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

    for index in range(1):
        test_seq, test_mask, test_y = TEST_DATA[index]
        with torch.no_grad():
            preds = model(test_seq.to(device), test_mask.to(device), test_seq.to(device), test_mask.to(device), test_seq.to(device), test_mask.to(device), TEST = True)
            preds = preds.detach().cpu().numpy()

        preds = np.argmax(preds, axis = 1)
        #print(classification_report(test_y, preds, digits=4))
        acc = accuracy_score(test_y, preds)
        f1micro = f1_score(test_y, preds, average='micro')
        f1macro = f1_score(test_y, preds, average='macro')
        f1weighted = f1_score(test_y, preds, average='weighted')
        print("Acc: "+str(round(acc*100,4)))
        print("F1: "+str(round(f1weighted*100,4)))
        print("--------------")

alpha = 1.0
beta = 1.0
# number of training epochs ##
epochs = 40
delta = 5e-5
lr = 5e-5
patience = 10

batch_size = 4
target = str(sys.argv[1])
semi_sup = 1 #int(sys.argv[2])
gamma = float(sys.argv[2])
base_model_name = str(sys.argv[3])

#######################
hidden_layers = 768
############### CREATE TOKENIZER ##########################
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

##########################
#define a batch size
batch_size = 4

cross_entropy  = nn.CrossEntropyLoss()

############### LOAD BERT ##########################
bert_teacher = None
bert_student = None
if base_model_name == "xlm-roberta-base":
    bert_teacher = RobertaForSequenceClassification.from_pretrained(base_model_name, num_labels = 2, output_attentions = False, output_hidden_states = True, return_dict=True)
    bert_student = RobertaForSequenceClassification.from_pretrained(base_model_name, num_labels = 2, output_attentions = False, output_hidden_states = True, return_dict=True)
    for param in bert_teacher.roberta.parameters():
        param.requires_grad = False
else:
    bert_teacher = BertForSequenceClassification.from_pretrained(base_model_name, num_labels = 2, output_attentions = False, output_hidden_states = True, return_dict=True)
    bert_student = BertForSequenceClassification.from_pretrained(base_model_name, num_labels = 2, output_attentions = False, output_hidden_states = True, return_dict=True)
    for param in bert_teacher.bert.parameters():
        param.requires_grad = False

############### LOAD BERT ##########################
model = mBERT(bert_teacher, bert_student, 2, hidden_layers)

if target == 'ml':
    final_model('data/appen/appen_en_train_pos', 'data/appen/appen_en_train_neg', 
                'data/appen/appen_en_train_pos_ml', 'data/appen/appen_en_train_neg_ml',
                'data/appen/appen_en_train_pos_ml_ro', 'data/appen/appen_en_train_neg_ml_ro',
                'data/appen/appen_en_val_pos', 'data/appen/appen_en_val_neg',
                'data/appen/appen_en_val_pos_ml', 'data/appen/appen_en_val_neg_ml',
                'data/appen/appen_en_val_pos_ml_ro', 'data/appen/appen_en_val_neg_ml_ro',
                mode = target
                )

if target == 'hi':
    final_model('data/appen/appen_en_train_pos', 'data/appen/appen_en_train_neg', 
                'data/appen/appen_en_train_pos_hi', 'data/appen/appen_en_train_neg_hi',
                'data/appen/appen_en_train_pos_hi_ro', 'data/appen/appen_en_train_neg_hi_ro',
                'data/appen/appen_en_val_pos', 'data/appen/appen_en_val_neg',
                'data/appen/appen_en_val_pos_hi', 'data/appen/appen_en_val_neg_hi',
                'data/appen/appen_en_val_pos_hi_ro', 'data/appen/appen_en_val_neg_hi_ro',
                mode = target
                )
