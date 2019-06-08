# -*- coding: utf-8 -*-

'''
The 2012 informatics for integrating biology and the bedside (i2b2)
project temporal relations challenge corpus (in short, i2b2 temporal
corpus) (29, 30): It contains 310 de-identified discharge summaries
of more than 178,000 tokens, with annotations of clinically signifi-
cant events, temporal expressions and temporal relations in clinical
narratives. On average, each discharge summary in the corpus con-
tains 86.6 events, 12.4 temporal expressions, and 176 raw temporal
relations. In this corpus, 8 kinds of temporal relations between
events and temporal expressions are defined: BEFORE, AFTER,
SIMULTANEOUS, OVERLAP, BEGUN_BY, ENDED_BY,
DURING, BEFORE_OVERLAP. The entire annotations are available at
http://i2b2.org/NLP/DataSets.


This script make the 2012 informatics for integrating biology and the bedside (i2b2) project temporal relations challenge corpus (in short, i2b2 temporal corpus)  into BERT input file format.
guid\ttarget\tcontext\tlabel\n

For example:

1_TL0	Admission 09/29/1993	Admission Date : 09/29/1993 Discharge Date : 10/04/1993 HISTORY OF PRESENT ILLNESS : The patient is a 28-year-old woman who is HIV positive for two years . She presented with left upper quadrant pain as well as nausea and vomiting which is a long-standing complaint . She was diagnosed in 1991 during the birth of her child . She claims she does not know why she is HIV positive . She is from Maryland , apparently had no blood transfusions before the birth of her children so it is presumed heterosexual transmission . At that time , she also had cat scratch fever and she had resection of an abscess in the left lower extremity . She has not used any anti retroviral therapy since then , because of pancytopenia and vomiting on DDI . She has complaints of nausea and vomiting as well as left upper quadrant pain on and off getting progressively worse over the past month . She has had similar pain intermittently for last year . She described the pain as a burning pain which is positional , worse when she walks or does any type of exercise . She has no relief from antacids or H2 blockers . In 10/92 , she had a CT scan which showed fatty infiltration of her liver diffusely with a 1 cm cyst in the right lobe of the liver . She had a normal pancreas at that time , however , hyperdense kidneys . Her alkaline phosphatase was slightly elevated but otherwise relatively normal . Her amylase was mildly elevated but has been down since then . The patient has had progressive failure to thrive and steady weight loss . She was brought in for an esophagogastroduodenoscopy on 9/26 but she basically was not sufficiently sedated and readmitted at this time for a GI work-up as well as an evaluation of new abscess in her left lower calf and right medial lower extremity quadriceps muscle . She was also admitted to be connected up with social services for HIV patients . HOSPITAL COURSE : The patient was admitted and many cultures were sent which were all negative . She did not have any of her pain in the hospital . On the third hospital day , she did have some pain and was treated with Percocet . She went for a debridement of her left calf lesion on 10/2/93 and was started empirically on IV ceftriaxone which was changed to po doxycycline on the day of discharge . A follow-up CT scan was done which did not show any evidence for splenomegaly or hepatomegaly . The 1 cm cyst which was seen in 10/92 was still present . There was a question of a cyst in her kidney with a stone right below the cyst , although this did not seem to be clinically significant .	SIMULTANEOUS


'''

import xml.etree.ElementTree as ET
import random
import os

TRAINNING_DATA_DIR = "./corpus/i2b2/2012-07-15.original-annotation.release/"
TEST_DATA_DIR = "./corpus/i2b2/ground_truth/merged_xml/"
SAVE_DIR = "./corpus/i2b2/"

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                # L.append(os.path.join(root, file))
                L.append(file)
    return L


def data_process(inDIR, outFile):
    fileList = file_name(inDIR)
    print(len(fileList))
    lableType = set()
    outFile = open(outFile, "w")
    for f in fileList:
        print(f, end=' ')
        linkNO = 0
        inFile = open(inDIR + f, "r")
        xmlString = ""
        for lines in inFile.readlines():
            xmlString += lines.replace(" & ", " ").replace("&", " and ")
        inFile.close()

        parser = ET.XMLParser(encoding="utf-8")
        root = ET.fromstring(xmlString, parser=parser)
        # tree = ET.parse(inDIR + f)
        # root = tree.getroot()
        text = root.find("TEXT").text.replace("\n", " ").strip()
        # print(text)
        tags = root.find("TAGS")
        for tlink in tags.findall("TLINK"):
            id = f[:-4] +"_"+ str(tlink.attrib['id'] )
            target = tlink.attrib['fromText'] + " " + tlink.attrib['toText']
            label = tlink.attrib['type'].upper()
            if label == '':
                continue
            lableType.add(label)
            # print(id + "\t"+target + "\t" + label)
            outFile.write(id + "\t" + target + "\t" + text + "\t"+ label+"\n")
            linkNO += 1
        print("linkNO = " + str(linkNO))
    print("*"*80)
    
if __name__ == '__main__':
    data_process( TRAINNING_DATA_DIR , SAVE_DIR + "train.txt")
    data_process( TEST_DATA_DIR, SAVE_DIR + "test.txt")
    