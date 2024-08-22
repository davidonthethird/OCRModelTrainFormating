import pandas as pd
from PIL import Image
import json

'''
Purpose of Code: To extract each scanned word into a json with following format {"Filename":"Text"} and export jpg 
of each word as scanned by OCR 

Prerequisites: 
1. CSV (ocr_out.csv) with the following column headers: [min_x, min_y, width, height,text] with data from 
OCR. 
2. JPG of each file the OCR Scanned in following format: (scanned_{page_number}.jpg) where page_number is document 
number (used for multiple document OCR scans)

Post running: You may parse the JSON to correct any discrepenceis between the documents and the OCR Scan. Then run 
through pytorch/tensorflow to train model. Code was written to work with docTR OCR but should work with most other 
OCR.
'''

#Open CSV File from scanned OCR
#CSV File needs to
with open('ocr_out.csv') as current_csv:

    #Read into Pandas DB
    full_df = pd.read_csv(current_csv)

    #Find last page for enumeration
    last_page = full_df.page_num.iloc[-1]

    #Enumerate through all pages
    for page_num in range(1,last_page+1):
        print(f'Curr page:{page_num}')

        #Remove any page that isnt current
        page_df = full_df.drop(full_df[full_df.page_num != page_num].index)

        #If page is empty, skip
        if page_df.empty:
            continue
        else:
            #Convert into numpy
            np_df = page_df[['min_x', 'min_y', 'width', 'height', 'text']].to_numpy()

            # Open image of full page
            img = Image.open(f'my_project/files/images/scanned_{page_num}.jpg')

            #Enumerate through each line in the database
            for (line_num,line) in enumerate(np_df,1):

                #Get length and width of each text box
                new_len = line[4]
                new_wid = line[5]
                print(f'len:{new_len} ~ width:{new_wid}')

                # Get coordinates for each point of text box
                min_x=line[2]
                min_y=line[3]
                max_x = line[2]+new_len
                max_y = line[3]+new_wid

                #Crop it and save it

                new_img = img.crop(tuple([min_x, min_y, max_x, max_y]))
                new_img = new_img.resize((new_len, new_wid))
                new_img.save(rf'./my_project/files/images/training/page_{page_num}_word_{line_num}.jpg')

                #Append to JSON with File Name:Word
                with open('my_project/files/images/training/training.json') as f:
                    data=json.load(f)

                new_json={f"page_{page_num}_word_{line_num}.jpg":line[-1]}
                data.update(new_json)

                with open('my_project/files/images/training/training.json',"w") as f:
                    json.dump(data,f)


