import pandas as pd
import pytesseract
from PIL import Image
import time
import os
import json
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
Purpose of Code: Create a dataset that can be used to train an OCR model from your data.

Prerequisites: 
1. CSV (ocr_out.csv) with the following column headers: [min_x, min_y, width, height,text] with data from 
OCR. 
2. JPG of each file the OCR Scanned in following format: (scanned_{page_number}.jpg) where page_number is document 
number (used for multiple document OCR scans)

Post running: You may parse the JSON to correct any discrepancy between the documents and the OCR Scan. Then run 
through pytorch/tensorflow to train model. Code was written to work with docTR OCR but should work with most other 
OCR.

Note: This is only for text recognition, not text detection.
'''
# ~~~ STEP 1 ~~~
# Extract text box

# Open CSV File from scanned OCR
# CSV File needs to
with open('ocr_out.csv') as current_csv:
    # Read into Pandas DB
    full_df = pd.read_csv(current_csv)

    # Find last page for enumeration
    last_page = full_df.page_num.iloc[-1]

    # Enumerate through all pages
    for page_num in range(1, last_page + 1):
        print(f'Curr page:{page_num}')

        # Remove any page that isnt current
        page_df = full_df.drop(full_df[full_df.page_num != page_num].index)

        # If page is empty, skip
        if page_df.empty:
            continue
        else:
            # Convert into numpy (runs magnitudes faster)
            np_df = page_df[['min_x', 'min_y', 'width', 'height', 'text']].to_numpy()

            # Open image of full page
            img = Image.open(f'Original Images/scanned_{page_num}.jpg')

            # Enumerate through each line in the database
            for (line_num, line) in enumerate(np_df, 1):
                # Get length and width of each text box
                new_len = line[2]
                new_wid = line[3]
                print(f'len:{new_len} ~ width:{new_wid}')

                # Get coordinates for each point of text box
                min_x = line[0]
                min_y = line[1]
                max_x = line[0] + new_len
                max_y = line[1] + new_wid

                # Crop it and save it

                new_img = img.crop(tuple([min_x, min_y, max_x, max_y]))
                new_img = new_img.resize((new_len, new_wid))
                new_img.save(rf'Test Images/page_{page_num}_word_{line_num}.jpg')

                # Append to JSON with File Name:Word
                with open('Jsons/training.json') as f:
                    data = json.load(f)

                new_json = {f"page_{page_num}_word_{line_num}.jpg": line[-1]}
                data.update(new_json)

                with open('Jsons/training.json', "w") as f:
                    json.dump(data, f)

# ~~~ STEP 2 ~~~
# Scan with secondary OCR (this takes the longest)
# Scans each text box using a different OCR (I chose pytesseract)
# Saves Confidence and Text output into a 2nd json (OCR_checker.json)

# Put your directory here for the text boxes (If you dont put full directories for this part, it can come up with an
# error occasionally)
directory = 'C:\Python Projects\OCRModelTrainFormating\Test Images'

# Need to download tesseract and put your dirrectory here
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set timer
start_time = time.time()
full_dict = {"filename": {"text": "text", "conf": "conf"}}

# iterate through all text boxes
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):

        # Open Image file
        with Image.open(f'Test Images/{filename}') as im:
            # Pytesseract
            data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DATAFRAME)
            data = data.dropna()

            # Get text
            text_df = data['text'].astype('str')
            text = ''.join(text_df.to_list())

            # Get confidence
            conf_data = data['conf']
            conf = conf_data.mean()

            # If scan doesn't work with standard configuration, it rescans with config meant for single word scans
            if data.empty:
                data = pytesseract.image_to_data(im, config='--psm 10', output_type=pytesseract.Output.DATAFRAME)
                data = data.dropna()

                # Get text
                text_df = data['text'].astype('str')
                text = ''.join(text_df.to_list())

                # Get confidence
                conf_data = data['conf']
                conf = float(conf_data.mean())

            print(f'{filename}:{text}-{conf}')

            # Add to dict only if text is found in image
            if text == '\n' or not text or text is None:
                continue
            else:
                full_dict.update({filename: {'text': text, 'conf': conf}})

# Save all to JSON
with open('C:\Python Projects\OCRModelTrainFormating\Jsons\OCR_checker.json', "w") as f:
    json.dump(full_dict, f)

# Get time elapsed
print(f'Time Elapsed = {time.time() - start_time}')

# ~~~ STEP 3 ~~~
# Manual Confirmation
# Manual data confirmation, checks both the initial scan (from docTR) against the secondary scan (pytesseract)
# and if there is low confidence or a discrepancy, a human inputs the data. This can then be used to train
# the OCR model

# If the title of the plot matches the word hit enter
# If the word is not a good sample to train the model on, hit the space bar then hit enter
# If the title doesn't match the word, enter the correct word

# Format JSON for saving later
final_check_dict = {"Filename": {"text": "text", "conf": "conf"}}

# Load json file from OCR_checker (pytesseract)
with open('Jsons/OCR_checker.json') as f:
    ocr_checker = json.load(f)

# Load file from training dataset (docTR)
with open('Jsons/training.json') as f:
    training = json.load(f)
    filename_training = training.keys()

# Get length of dataset to display how much is completed
tot = len(training)
i = 0

# Iterate through values in OCR_checker json
for filename_ocr, values_ocr in ocr_checker.items():

    text_ocr = values_ocr["text"]
    conf_ocr = values_ocr["conf"]

    # Get text from initial dataset
    if filename_ocr in filename_training:
        text_training = training[filename_ocr]
        word = ''

        # If there is low confidence or a discrepancy, display text box for manual data entry
        if text_training != text_ocr or conf_ocr < 85:

            # Show image
            imgplot = plt.imshow(mpimg.imread(f"C:/Python Projects/OCRModelTrainFormating/Test Images/{filename_ocr}"))
            plt.ion()
            plt.title(text_training)
            plt.show()

            # Get actual input for text in image from human
            word = input(f"Expected: {text_training} ~ Actual: ")

            # If there is an input that is not a space, we update the word with the new word
            if not word == " " and word:
                final_check_dict.update({filename_ocr: {'text': word, 'conf': conf_ocr}})
            # If there is no input, we keep the initial word
            else:
                final_check_dict.update({filename_ocr: {'text': text_training, 'conf': conf_ocr}})

            plt.close()

        # IF there is high confidence and no discrepancy, we keep the initial word
        else:
            final_check_dict.update({filename_ocr: {'text': text_training, 'conf': conf_ocr}})

        # Only save if we want to keep the dataset (no space-bar entry)
        if not word == " ":
            source = f"C:/Python Projects/OCRModelTrainFormating/Test Images/{filename_ocr}"
            destination = f"C:/Python Projects/OCRModelTrainFormating/Final Images/{filename_ocr}"
            shutil.copyfile(source, destination)

    i += 1
    print(f'Completed: {i}/{tot}')

# Save Json
with open("C:/Python Projects/OCRModelTrainFormating/Jsons/final_check.json", "w") as f:
    json.dump(final_check_dict, f)
