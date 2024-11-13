import pandas as pd
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
from PIL import Image
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
import os 

df = pd.read_csv('../flickr8k/test.csv')
# df = test_df[:5]

# Load the VisionEncoderDecoder model and processor
model_name = '/home/ponleur.veng/image-caption/output_training/vit-base-patch16-224-in21k-xlm-roberta-base'  # Replace with your chosen model
image_processor = AutoImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name).to('cuda')

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to('cuda')

    output_ids = model.generate(pixel_values, max_length=50, num_beams=5, no_repeat_ngram_size=2)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Group captions by image_id
ground_truth_dict = df.groupby('image')['khmer'].apply(list).to_dict()

# Generate captions for each unique image in the DataFrame
unique_image_ids = df['image'].unique()
generated_dict = {}
for image_id in unique_image_ids:
    image_path = f"/home/ponleur.veng/image-caption/flickr8k/images/{image_id}"  # Replace with your actual image path logic
    generated_caption = generate_caption(image_path)
    generated_dict[image_id] = [generated_caption]

# Evaluate with CIDEr
cider_scorer = Cider()
cider_score, _ = cider_scorer.compute_score(ground_truth_dict, generated_dict)
print(f"CIDEr Score: {cider_score}")

rouge_scorer = Rouge()
rouge_score, _ = rouge_scorer.compute_score(ground_truth_dict, generated_dict)
print(f"Rouge Score: {rouge_score}")

bleu_scorer = Bleu()
bleu_score, _ =bleu_scorer.compute_score(ground_truth_dict, generated_dict)
print(f"Bleu Score: {bleu_score}")
# Evaluate with SPICE
# spice_scorer = Spice()
# spice_score, _ = spice_scorer.compute_score(ground_truth_dict, generated_dict)
# print(f"SPICE Score: {spice_score}")



#vit+labse
# CIDEr Score: 0.8376913469722171
# Rouge Score: 0.4027789652924447
# {'testlen': 58361, 'reflen': 60584, 'guess': [58361, 52900, 47439, 41978], 'correct': [26608, 9795, 4010, 1577]}
# ratio: 0.9633071438003274
# Bleu Score: [0.43888123328337453, 0.2796898082963505, 0.18532750443352206, 0.12317477331640402]
#vit+xlm
#CIDEr Score: 0.19636272641838393                                                                                        Rouge Score: 0.2551032666223029                                                                                         {'testlen': 82100, 'reflen': 63884, 'guess': [82100, 76639, 71178, 65717], 'correct': [20685, 2833, 441, 46]}           ratio: 1.2851418195479105                                                                                               #Bleu Score: [0.25194884287454017, 0.09650605363568877, 0.038642686117072875, 0.014176548435260113] 
#vit+khmer-xlm-roberta-base
#CIDEr Score: 0.4038115190267081
#Rouge Score: 0.3285747431357339
#{'testlen': 53863, 'reflen': 59632, 'guess': [53863, 48402, 42941, 37480], 'correct': [21277, 5543, 1687, 476]}
#ratio: 0.9032566407297944
#Bleu Score: [0.3548989522591908, 0.19108895056713893, 0.1088260786687214, 0.06192596651054998]
