import os

import datasets
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from transformers import AutoTokenizer ,  AutoImageProcessor , default_data_collator,AutoModelForCausalLM


if torch.cuda.is_available():    

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

rouge = datasets.load_metric("rouge")
cer_metric = datasets.load_metric("cer")


class config : 
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    VAL_EPOCHS = 1
    LR = 5e-5
    SEED = 42
    MAX_LEN = 128
    SUMMARY_LEN = 20
    WEIGHT_DECAY = 0.01
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_PCT = 0.95
    NUM_WORKERS = mp.cpu_count()
    EPOCHS = 5
    IMG_SIZE = (224,224)
    LABEL_MASK = -100
    TOP_K = 1000
    TOP_P = 0.95
    
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    # cer = cer_metric.compute(predictions=pred_str,references=label_str)
    return {
        # "cer":cer,
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }
    
transforms = transforms.Compose(
    [
        transforms.Resize(config.IMG_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0.5, 
            std=0.5
        )
   ]
)

train_df = pd.read_csv('/home/ponleur.veng/image-caption/flickr8k/train.csv')
val_df = pd.read_csv('/home/ponleur.veng/image-caption/flickr8k/val.csv')

class ImgDataset(Dataset):
    def __init__(self, df,root_dir,tokenizer,feature_extractor, transform = None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer= tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = 50
    def __len__(self,):
        return len(self.df)
    def __getitem__(self,idx):
        caption = self.df['khmer'].iloc[idx]
        image = self.df['image'].iloc[idx]
        img_path = os.path.join(self.root_dir , image)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img= self.transform(img)
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        captions = self.tokenizer(caption,truncation=True,
                                 padding='max_length',
                                 max_length=self.max_length).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(captions)}
        return encoding
        


# Encoder and decoder models to loop through
encoder_models = "google/vit-base-patch16-224-in21k"
#decoder_models = "sentence-transformers/LaBSE"
decoder_models = "FacebookAI/xlm-roberta-base"


        # Initialize the feature extractor and tokenizer
feature_extractor = ViTImageProcessor.from_pretrained(encoder_models)
tokenizer = AutoTokenizer.from_pretrained(decoder_models)
        #tokenizer.pad_token = tokenizer.unk_token  # Fix for missing pad token
        
        # Custom dataset class remains the same
train_dataset = ImgDataset(train_df, root_dir="/home/ponleur.veng/image-caption/flickr8k/images", tokenizer=tokenizer, feature_extractor=feature_extractor)
val_dataset = ImgDataset(val_df, root_dir="/home/ponleur.veng/image-caption/flickr8k/images", tokenizer=tokenizer, feature_extractor=feature_extractor)

        # Initialize VisionEncoderDecoder model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_models, decoder_models)
model.to(device)

        # Configure the model
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

        # Define output directory for this particular combination
out_directory = f'/home/ponleur.veng/image-caption/output_training/{encoder_models.split("/")[1]}-{decoder_models.split("/")[1]}'
        
        # Training arguments
training_args = Seq2SeqTrainingArguments(
            output_dir=out_directory,
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.VAL_BATCH_SIZE,
            predict_with_generate=True,
            evaluation_strategy="steps",
            logging_steps=1024,
            save_steps=2048,
            num_train_epochs=config.EPOCHS,
            overwrite_output_dir=True,
            save_total_limit=1,
            load_best_model_at_end=True,
            learning_rate=config.LR,
        )

        # Define trainer
trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            tokenizer=feature_extractor,  # Use feature extractor as tokenizer for images
            compute_metrics=compute_metrics,
        )

        # Train the model
trainer.train()

        # Save the model and tokenizer
trainer.save_model(out_directory)
tokenizer.save_pretrained(out_directory)

        
