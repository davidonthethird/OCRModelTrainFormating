Export Text Boxes and JSON of words from OCR to correct and train models

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
