**Export Text Boxes and JSON of words from OCR to correct and train models**

Purpose of Code: Create a dataset that can be used to train an OCR model from your data.

Prerequisites: 
1. CSV (ocr_out.csv) with the following column headers: [min_x, min_y, width, height,text] with data from 
OCR. 
2. JPG of each file the OCR Scanned in following format: (scanned_{page_number}.jpg) where page_number is document 
number (used for multiple document OCR scans)

Was written to work with docTR primarily but should work with any other OCR

