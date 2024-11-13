import requests
from PIL import Image
import datasets
from transformers import VisionEncoderDecoderModel, AutoImageProcessor,AutoTokenizer
import pandas as pd
import os
trained_model = '/home/ponleur.veng/image-caption/output_training/vit-base-patch16-224-in21k-khmer-xlm-roberta-base'
#trained_model = '/home/ponleur.veng/image-caption/output_training/vit-base-patch16-224-in21k-LaBSE'
# trained_model = '/home/ponleur.veng/image-caption/output_training/deit-base-patch16-224-roberta-base'
# trained_model = '/home/ponleur.veng/image-caption/output_training/vit-base-patch16-224-in21k-roberta-base' 
# trained_model = '/home/ponleur.veng/image-caption/output_training/vit-base-patch16-224-in21k-bert-base-uncased' 

model = VisionEncoderDecoderModel.from_pretrained(trained_model)
# print(model)
tokenizer = AutoTokenizer.from_pretrained(trained_model)
image_processor = AutoImageProcessor.from_pretrained(trained_model)
model.to('cuda')

test_df = pd.read_csv('../flickr8k/test.csv')

# max_length = 128
# num_beams = 4
# gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# image = Image.open(os.path.join('/home/ponleur.veng/image-caption/flickr8k/images',test_df['image'][0])).convert('RGB')
# pixel_values = image_processor(image, return_tensors="pt").pixel_values

# # autoregressively generate caption (uses greedy decoding by default)
# generated_ids = model.generate(pixel_values,**gen_kwargs)
# generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(f"Original Text in Khmer {test_df['khmer'][0]}")
# print(f"Original Text in English {test_df['caption'][0]}")
# print(f'Generated Text {generated_text}')

generated = []
reference = []

metric = datasets.load_metric('rouge','bleu')
# Loop through DataFrame columns using items()
for i in range(len(test_df)):
    # print(f"Index: {i}, A: {test_df.loc[i, 'image']}, B: {test_df.loc[i, 'khmer']}")
    
    max_length = 128
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    image = Image.open(os.path.join('/home/ponleur.veng/image-caption/flickr8k/images',test_df.loc[i, 'image'])).convert('RGB')
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to('cuda')
    generated_ids = model.generate(pixel_values,**gen_kwargs)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    reference.append(test_df.loc[i, 'khmer'])
    generated.append(generated_text)

result = metric.compute(predictions=generated, references=reference)

print(result)


    


