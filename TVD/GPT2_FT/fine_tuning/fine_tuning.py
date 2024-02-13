#Importing relevant libraries
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.trainer_utils import set_seed
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse

import random
import os
import json
import time
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_provo_corpus(input_data):
    """Reading the raw Provo Corpus dataset and create a dictionary with all useful 
       information we might need from it"""
    predict_norms = pd.read_csv(input_data, sep='\t')
    paragraphs = predict_norms.groupby('Text_ID')['Text'].max()
    
    return(paragraphs)

class GPT2Dataset(data.Dataset):

    def __init__(self, df_txt, tokenizer):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.text_id = []

        self.bos_token = '<|endoftext|>'
        self.pad_token = '<|endoftext|>'
        self.eos_token = '<|endoftext|>'
        self.max_length = 81 #remember to change this
    
        for item in df_txt.index:
            self.text_id.append(torch.LongTensor([item]))
            encodings_dict = tokenizer(f'{tokenizer.bos_token} {df_txt[item]}', max_length = self.max_length, padding="max_length")
        
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.text_id[idx]

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def split_train_val_test(split_dict):
    """We have decided to train and the model on the first 40 paragraphs and validate on the last 15. Among the
    first 40, we keep the paragraphs listed in test_text_ids as testing and the remaining we use for training"""
	
    test_text_ids = split_dict['test_text_ids']
    dataset = split_dict['dataset'] 
    training_threshold = split_dict['training_threshold']
    
    training_set = []
    validation_set = []
    test_set = []
    
    for datapoint in dataset:
        if datapoint[2] > training_threshold:
            validation_set.append(datapoint)
        elif datapoint[2] in test_text_ids: 
            test_set.append(datapoint)
        else:
            training_set.append(datapoint)
    
    return training_set, validation_set, test_set

def prep_fine_tuning_data(tokenizer, test_text_ids, num_texts, batch_size, seed):
	#input_data = os.path.join(os.getcwd(), 'raw_data/Provo_Corpus.tsv') #for login node
	input_data = os.path.join(os.getcwd(), 'Provo_Corpus.tsv') #for submitted jobs
	texts = preprocess_provo_corpus(input_data)
	
	torch.manual_seed(seed)

    #We would like to fine tune GPT2 on a subset of the Provo corpus to remove some of the OOD bias when evaluating
    #calibration. We will evaluate calibration on the remaining part of the Provo Corpus dataset
	dataset = GPT2Dataset(texts, tokenizer) #[:num_texts]
	
	split_dict = {'test_text_ids': test_text_ids, 'dataset':dataset, 'training_threshold': num_texts}
	
	train_dataset, val_dataset, test_dataset = split_train_val_test(split_dict)

    #Create the DataLoaders for our training and validation datasets. We'll take training samples in random order. 
	train_dataloader = data.DataLoader(train_dataset,  # The training samples.
                                    #sampler = data.RandomSampler(train_dataset), # Select batches randomly
									shuffle=True,
                                    batch_size = batch_size)

    # For validation the order doesn't matter, so we'll just read them sequentially.
	validation_dataloader = data.DataLoader(val_dataset, # The validation samples.
                                          sampler = data.SequentialSampler(val_dataset), # Pull out batches sequentially.
                                          batch_size = batch_size)
										  
	return train_dataloader, validation_dataloader

def training(model, train_dataloader, validation_dataloader, epochs, optimizer, scheduler):
	total_t0 = time.time()
	training_stats = []
	
	model = model.to(device)

	for epoch in range(epochs):
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
		print('Training...')
		
		t0 = time.time()
    
		#Training
		total_train_loss = 0
		model.train()
		
		for batch in train_dataloader:
			b_input_ids = batch[0].to(device)
			b_labels = batch[0].to(device)
			b_masks = batch[1].to(device)
		
			model.zero_grad()
			outputs = model( b_input_ids, labels=b_labels, attention_mask = b_masks, token_type_ids=None)
		
			loss = outputs[0] #average of (input-length normalised) CE losses of the batch 
		
			batch_loss = loss.item()
			total_train_loss += batch_loss
		
			loss.backward()
			optimizer.step()
			scheduler.step()
    
    	# Calculate the average loss over all of the batches.
	
		avg_train_loss = total_train_loss / len(train_dataloader)       
    
    	# Measure how long this epoch took.
		training_time = format_time(time.time() - t0)
		
		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
		print("  Training epoch took: {:}".format(training_time))
		
		#Validation
		print("")
		print("Running Validation...")
		
		t0 = time.time()
		model.eval()
		
		total_eval_loss = 0
		
		# Evaluate data for one epoch
		for batch in validation_dataloader:
			b_input_ids = batch[0].to(device)
			b_labels = batch[0].to(device)
			b_masks = batch[1].to(device)
			
			with torch.no_grad():
				outputs  = model(b_input_ids, token_type_ids=None, attention_mask = b_masks, labels=b_labels)
				loss = outputs[0]  
            
			batch_loss = loss.item()
			total_eval_loss += batch_loss 
		
		avg_val_loss = total_eval_loss / len(validation_dataloader)
		validation_time = format_time(time.time() - t0)
		
		print("  Validation Loss: {0:.2f}".format(avg_val_loss))
		print("  Validation took: {:}".format(validation_time))

		# Record all statistics from this epoch.
		training_stats.append({'epoch': epoch + 1,
                           'Training Loss': avg_train_loss,
                           'Valid. Loss': avg_val_loss,
                           'Training Time': training_time,
                           'Validation Time': validation_time})
	
	print("")
	print("Training complete!")
	print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
	
	return model, training_stats

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def fine_tuning(model, train_dataloader, validation_dataloader, dict_parameters):
	# Set the seed value all over the place to make this reproducible
	set_seed(dict_parameters['seed_val'])
	
	optimizer = AdamW(model.parameters(),
                  lr = dict_parameters['learning_rate'],
                  eps = dict_parameters['epsilon'])
  
	# Total number of training steps is [number of batches] x [number of epochs]. 
	total_steps = len(train_dataloader) * dict_parameters['epochs']
	
	# Create the learning rate scheduler. This changes the learning rate as the training loop progresses
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = dict_parameters['warmup_steps'], num_training_steps = total_steps)

	#Training
	model, training_stats = training(model, train_dataloader, validation_dataloader, dict_parameters['epochs'], optimizer, scheduler)

	return model, training_stats
    

seed_val = 0
epochs = 10
learning_rate = 5e-4
warmup_steps = 10
epsilon = 1e-8
batch_size = 5

k_cross_fold_val = 4 #the parameter k of k-cross-fold validation (also the number of models that we will be training)
num_texts = 40 #number of paragraphs that will be used for training and validation - they will be allocated to the
#first num_texts of provo corpus (since we can assume paragrphs are i.i.d.)

dict_parameters = {'seed_val': seed_val, 'epochs': epochs, 'learning_rate': learning_rate, 
				   'warmup_steps':warmup_steps, 'epsilon':epsilon, 'batch_size': batch_size,
				   'k_cross_fold_val':k_cross_fold_val, 'num_texts': num_texts}


size_fold = round(num_texts/k_cross_fold_val)

for i in range(1, num_texts+1, size_fold):
	model = GPT2LMHeadModel.from_pretrained('gpt2')
	#Freeze all parameters 
	for param in model.parameters():
		param.requires_grad = False

	# Choose which layers we would like to continue training.
	# We only want to update the weights of the last layer
	module_to_unfreeze = [model.transformer.h[11].ln_1, model.transformer.h[11].ln_2, model.transformer.h[10].ln_1, model.transformer.h[10].ln_2, model.transformer.ln_f]

	for module in module_to_unfreeze:
		for param in module.parameters():
			param.requires_grad = True

	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	tokenizer.pad_token = tokenizer.eos_token

	test_text_ids = [*range(i, i + size_fold)]
	first_val_element = test_text_ids[0]
	last_val_element = test_text_ids[-1]
	model_version = '_test_' + str(first_val_element) + '-' + str(last_val_element)
	
	train_dataloader, validation_dataloader = prep_fine_tuning_data(tokenizer, test_text_ids, num_texts, batch_size, seed_val)
	model, training_stats = fine_tuning(model, train_dataloader, validation_dataloader, dict_parameters)
	
	#Save training statistics (training and validation losses)
	with open('training_stats' + model_version + '.json', 'w') as fp:
		json.dump(training_stats, fp)
		
	with open('training_parameters' + model_version + '.json', 'w') as fp:
		json.dump(dict_parameters, fp)
	
	#Saving model
	output_dir = './gpt2_provo_fine_tuned' + model_version + '/'

	# Create output directory if needed
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Save a trained model, configuration and tokenizer. They can then be reloaded using 'from_pretrained()'
	gpt2_provo_fine_tuned = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
	gpt2_provo_fine_tuned.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)