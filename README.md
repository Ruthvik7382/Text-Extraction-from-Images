# Text-Extraction-from-Images
Mounting Google Drive: The code begins by mounting Google Drive to access files stored there.

Image Processing: I used cv2 (OpenCV) for image processing tasks, which includes reading images from a specified directory.

Text Extraction: The core of the code involves using pytesseract for Optical Character Recognition (OCR) to extract text from images. It includes steps for converting images to grayscale and thresholding before feeding them to pytesseract.

Tesseract Configuration: It configures pytesseract with custom options for better OCR performance.

Data Extraction and Parsing: The code extracts detailed information like the level, page number, block number, etc., from the OCR results. It also includes code for parsing the extracted text and handling it (e.g., storing in a list).

File Writing: The extracted text is written to a text file (result_text.txt).

Function for Text Extraction: A function named text_extraction is defined for encapsulating the OCR process, suggesting a modular approach.
