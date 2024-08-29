import PIL
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import time
import os, glob
import json
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from alive_progress import alive_bar
import csv

'''
Purpose of Code: Create a dataset that can be used to train an OCR model from your data.

Prerequisites: 
1. CSV (ocr_out.csv) with the following column headers: [min_x, min_y, width, height,text] with data from 
OCR. 
2. JPG of each file the OCR Scanned in following format: (scanned_{page_number}.jpg) where page_number is document 
number (used for multiple document OCR scans)

**Large datasets will have to run code twice, second time you run it press continue to access manual editing**
This is done to prevent code from running waiting for an input. Datasets with >1000 words

Post running: You may run through pytorch/tensorflow to train model. Code was written to work with docTR OCR but should 
work with most other OCR.

Code will automatically stop between the text extraction and re-scanning process and the manual process 
(Between Step 2 and 3)

Note: This is only for text recognition, not text detection.
'''

# Put your directory here for the project OCRModelTrainFormating (If you don't put full directories for this part,
# it can come up with an error occasionally)
directory = 'C:\Python Projects\OCRModelTrainFormating'
if not directory:
    raise ValueError('Please enter a valid directory')

# Need to download tesseract and put your directory here
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if not pytesseract.pytesseract.tesseract_cmd:
    raise ValueError('Please enter a valid tesseract directory')

# Check previous runs history
with open(f'{directory}/data/history.json', 'r') as f:
    history = json.load(f)
    step = history['Step']

    # Ask for continue where we left off or start from beginning
    if step > 1:
        c = input(f'Welcome to OCRModelTrainFormating\n Would you like to continue where you left off? (Y/N): ')
        # Set step to 1 (beginning) if we are not continuing
        if c.lower() != "y":
            step = 1
            # Remove Test files and final images
            print('% Removing Old Files %')
            test_files = glob.glob(f'{directory}/Test Images/*')
            final_files = glob.glob(f'{directory}/Final Images/*')

            all_files = test_files + final_files
            for f in all_files:
                # Dont remove text file
                if ".txt" not in f:
                    os.remove(f)
                    # Bug that requires sleep otherwise it won't delete files
                    time.sleep(.1)

            # Erase Json history
            with open(f'{directory}/data/history.json', 'w') as f:
                json.dump("{}", f)

# ~~~ STEP 1 ~~~
# Extract text box
print("~Starting Step 1~")
# Open CSV File from scanned OCR
# CSV File needs to
if step == 1:

    with open(f'{directory}/data/ocr_out.csv') as current_csv:
        # Read into Pandas DB
        full_df = pd.read_csv(current_csv)

    # Find last page for enumeration
    last_page = full_df.page_num.iloc[-1]
    training_data = []

    # Progress Bar
    with alive_bar(int(last_page + 1), force_tty=True) as bar:
        # Enumerate through all pages
        for page_num in range(1, last_page + 1):
            # Remove any page that isn't current
            page_df = full_df.drop(full_df[full_df.page_num != page_num].index)

            # If page is empty, skip
            if page_df.empty:
                continue
            else:
                # Convert into numpy (runs magnitudes faster)
                np_df = page_df[['min_x', 'min_y', 'width', 'height', 'conf', 'text']].to_numpy()

                # Open image of full page
                img = Image.open(f'{directory}/Original Images/scanned_{page_num}.jpg')

                # Enumerate through each line in the database
                for (line_num, line) in enumerate(np_df, 1):
                    # Get length and width of each text box
                    new_len = line[2]
                    new_wid = line[3]

                    # Get coordinates for each point of text box
                    min_x = line[0]
                    min_y = line[1]
                    max_x = line[0] + new_len
                    max_y = line[1] + new_wid

                    # Crop it and save it

                    new_img = img.crop(tuple([min_x, min_y, max_x, max_y]))
                    new_img = new_img.resize((new_len, new_wid))
                    new_img.save(rf'{directory}/Test Images/page_{page_num}_word_{line_num}.jpg')

                    # Append data to dict list
                    new_line = {"filename": f"page_{page_num}_word_{line_num}.jpg", "text": line[-1], "conf": line[-2]}
                    training_data.append(new_line)

            bar()

    with open(f'{directory}/data/training_data.csv', "w", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'text', 'conf'])
        writer.writeheader()
        writer.writerows(training_data)

    # Save step in history
    with open(f'{directory}/data/history.json', 'w') as f:
        json.dump({"Step": 2}, f)
    print('~Finished Step 1~\n~Beginning Step 2~')
    step = 2

# ~~~ STEP 2 ~~~
# Scan with secondary OCR (this takes the longest)
# Scans each text box using a different OCR (I chose pytesseract)
# Saves Confidence and Text output into a 2nd csv (ocr_data.csv)
if step == 2:
    # Set timer
    start_time = time.time()
    ocr_data = []

    # iterate through all text boxes
    for filename in os.listdir(f'{directory}/Test Images'):
        f = os.path.join(f'{directory}/Test Images', filename)
        if os.path.isfile(f):

            # Open Image file
            try:
                with Image.open(f'{directory}/Test Images/{filename}') as im:
                    # Pytesseract
                    data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DATAFRAME)
                    data = data.dropna()

                    # Get text
                    text_df = data['text'].astype('str')
                    text = ''.join(text_df.to_list())

                    # Get confidence
                    conf_data = data['conf']
                    conf = conf_data.mean()

                    # If scan doesn't work with standard configuration, it rescans with config meant for single word
                    # scans
                    if data.empty:
                        data = pytesseract.image_to_data(im, config='--psm 10',
                                                         output_type=pytesseract.Output.DATAFRAME)
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
                        ocr_data.append({"filename": filename, "text": text, "conf": conf})

            # Exceptions for textfile and DS_Store when opening file
            except PIL.UnidentifiedImageError:
                if '.txt' in filename or ".DS_Store" in filename:
                    continue
                else:
                    raise PIL.UnidentifiedImageError(f'Cannot identify image in file: {f}')

    # Save ocr data to csv file
    with open(f'{directory}/data/ocr_data.csv', "w", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'text', 'conf'])
        writer.writeheader()
        writer.writerows(ocr_data)

    # Get time elapsed
    print(f'Time Elapsed = {time.time() - start_time}')
    print("~Finished Step 2~\n~Beginning Step 3~")

    # Large files stop here rerun and enter "y" when prompted to continue to manual confirmation
    if len(os.listdir(f'{directory}/Test Images')) < 5000:
        step = 3

    # Save step into history Json
    with open(f'{directory}/data/history.json', 'w') as f:
        json.dump({"Step": 3}, f)

# ~~~ STEP 3 ~~~
# Manual Confirmation
# Manual data confirmation, checks both the initial scan (from docTR) against the secondary scan (pytesseract)
# and if there is low confidence or a discrepancy, a human inputs the data. This can then be used to train
# the OCR model

# HOW TO USE:
# If the title of the plot matches the word hit enter
# If the word is not a good sample to train the model on, hit the space bar then hit enter
# If the title doesn't match the word, enter the correct word
# Format JSON for saving later
if step == 3:
    final_data = []

    # Load file from OCR_checker (pytesseract) and convert to numpy (faster)
    ocr_data = pd.read_csv(f'{directory}/data/ocr_data.csv')
    ocr_np = ocr_data[['filename', 'text', 'conf']].to_numpy()

    # Load file from training dataset (docTR)  and convert to numpy (faster)
    training_data = pd.read_csv(f'{directory}/data/training_data.csv')
    training_np = training_data[['filename', 'text', 'conf']].to_numpy()

    # Get length of dataset to display how much is completed
    tot = len(training_data)
    i = 0

    # Iterate through values in OCR_checker csv

    for filename_ocr, text_ocr, conf_ocr in ocr_np:
        # Get text from initial dataset
        i = np.where(training_np[:, 0] == filename_ocr)[0][0]
        text_training = training_np[i][1]
        conf_training = training_np[i][2]

        word = ''

        try:
            # If there is low confidence or a discrepancy, display text box for manual data entry
            if float(conf_training) < 50 or (int(conf_training) < 80 and not text_training == text_ocr) or (
                    float(conf_training) < 90 and text_training != text_ocr) or (
                    float(conf_training) >= 90 and text_training != text_ocr and float(conf_ocr) > 90):

                # Show image
                imgplot = plt.imshow(mpimg.imread(f"{directory}/Test Images/{filename_ocr}"))
                plt.rc('axes')
                plt.ion()

                # Set a proportional font size so it doesn't go off the screen
                font = abs(55 - len(text_training))
                plt.title(text_training, fontsize=font)
                plt.tight_layout()

                plt.show()

                # Get actual input for text in image from human
                word = input(f"Expected: {text_training} ~ Actual: ")

                # If there is an input that is not a space, we update the word with the new word
                if not word == " " and word:
                    final_data.append({"filename": filename_ocr, "text": text_training})
                # If there is no input, we keep the initial word
                else:
                    final_data.append({"filename": filename_ocr, "text": text_training})

                plt.close()

            # IF there is high confidence and no discrepancy, we keep the initial word
            else:
                final_data.append({"filename": filename_ocr, "text": text_training})
        except ValueError:
            continue

        # Only save if we want to keep the dataset (no space-bar entry)
        if not word == " ":
            source = f"{directory}/Test Images/{filename_ocr}"
            destination = f"{directory}/Final Images/{filename_ocr}"
            shutil.copyfile(source, destination)

        i += 1
        print(f'Completed: {i}/{tot}')

    # Save csv
    with open(f'{directory}/data/final_data.csv', "w") as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'text'])
        writer.writeheader()
        writer.writerows(final_data)

    print("~Completed Entire Process!~")
