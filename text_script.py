import re

def process_line(line):
    # Replace all-uppercase word sequences with the desired format
    line = re.sub(r'\b([A-Z][A-Z\s,]*[A-Z])\b', r'<title> \1 <eotitle> <sos>', line)
    
    # Remove commas from the <title> and <eotitle> tokens
    line = re.sub(r'(?<=<title>)(.*?)(?= <eotitle>)', lambda m: m.group(0).replace(',', ''), line)
    
    return line

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            outfile.write(process_line(line))

input_file = 'kids_stories_val.txt'
output_file = 'kids_stories_val_new.txt'

process_file(input_file, output_file)