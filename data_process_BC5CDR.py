# -*- coding: utf-8 -*-

'''
BioCreative V chemical-disease relation (CDR) corpus (in short,
BC5CDR corpus) (13, 14, 16, 34): It consists of 1,500 PubMed arti-
cles with 4,409 annotated chemicals, 5,818 diseases, and 3,116
chemical-disease interactions. The relation task data is publicly available through
BioCreative V at https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/.

This script make BioCreative V Chemical Disease Relation (CDR) corpus into BERT input file format.
guid\ttarget\tcontext\tlabel\n

For example:

227508_1	alpha-methyldopa hypotensive	naloxone reverses the antihypertensive effect of clonidine .  in unanesthetized ,  spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine ,  5 to 20 micrograms/kg ,  was inhibited or reversed by nalozone ,  0 . 2 to 2 mg/kg .  the hypotensive effect of 100 mg/kg alpha-methyldopa was also partially reversed by naloxone .  naloxone alone did not affect either blood pressure or heart rate .  in brain membranes from spontaneously hypertensive rats clonidine ,  10 ( -8 )  to 10 ( -5 )  m ,  did not influence stereoselective binding of [3h]-naloxone  ( 8 nm )  ,  and naloxone ,  10 ( -8 )  to 10 ( -4 )  m ,  did not influence clonidine-suppressible binding of [3h]-dihydroergocryptine  ( 1 nm )  .  these findings indicate that in spontaneously hypertensive rats the effects of central alpha-adrenoceptor stimulation involve activation of opiate receptors .  as naloxone and clonidine do not appear to interact with the same receptor site ,  the observed functional antagonism suggests the release of an endogenous opiate by clonidine or alpha-methyldopa and the possible role of the opiate in the central control of sympathetic tone .	1
'''

import xml.etree.ElementTree as ET
import random

CORPUS_DIR = "./corpus/BC5CDR/CDR_Data/CDR.Corpus.v010516/"
SAVE_DIR = "./corpus/BC5CDR/"

def randomSelectCID(cList, dList):
    c = cList[random.randint(0, len(cList) - 1)]
    d = dList[random.randint(0, len(dList) - 1)]
    return c + " " + d

def data_process(inFile, outFile):
    tree = ET.parse(inFile)
    root = tree.getroot()
    docNO = len(root) - 3
    print("docNO = " + str(docNO))  # 500
    docNO = 0

    outFile = open(outFile, "w")
    errorNO = 0
    for child in root[3:]:
        docNO += 1
        id = child[0].text
        print(id)
        text = ""
        chemical, desease = {}, {}
        for passage in child.findall("passage"):
            text += str(passage[2].text).lower() + " "
            for annotation in passage.findall("annotation"):
                flag = ""
                s = str(annotation.find("text").text).lower()
                # print(s)
                for infon in annotation.findall("infon"):
                    if infon.attrib['key'] == "type":
                        if infon.text == "Chemical":
                            flag = "Chemical"
                        elif infon.text == "Disease":
                            flag = "Disease"
                    if infon.attrib['key'] == "MESH":
                        for k in str(infon.text).split("|"):
                            if flag == "Chemical":
                                chemical[k] = s
                            elif flag == "Disease":
                                desease[k] = s

        symbols = [".", ":",",", "%", ")", "(" ]
        for symbol in symbols:
            text = text.replace(symbol, " "+ symbol+" ")
        text= text.rstrip()
        print(chemical)
        print(desease)
        print("*" * 80)
        # print(text)

        relationNO = 0
        cidDict = {}
        cidList = []
        for relation in child.findall("relation"):
            relationNO += 1
            c = chemical[relation[1].text]
            d = desease[relation[2].text]
            outFile.write(id + "_" + str(relationNO) + "\t" + c + " " + d + "\t" + text + "\t1\n")
            cidDict[c] = d
            cidList.append(c + " " + d)

        negRelationNO = relationNO
        maxTryNO = 100
        tryNO = 0
        for i in range(relationNO):
            negRelationNO += 1
            # print(chemical.values())
            # print(list(chemical.values()))
            cid = randomSelectCID(list(chemical.values()), list(desease.values()))
            while cid in cidList and tryNO < maxTryNO:
                tryNO += 1
                cid = randomSelectCID(list(chemical.values()), list(desease.values()))
            if tryNO < maxTryNO:
                outFile.write(id + "_" + str(negRelationNO) + "\t" + cid + "\t" + text + "\t0\n")
            else:
                print("\n\n\n" + str(id))

    print("docNO = " + str(docNO))
    outFile.close()

if __name__ == '__main__':
    data_process( CORPUS_DIR + "CDR_TrainingSet.BioC.xml", SAVE_DIR + "train.txt")
    data_process( CORPUS_DIR + "CDR_DevelopmentSet.BioC.xml", SAVE_DIR + "dev.txt")
    data_process( CORPUS_DIR + "CDR_TestSet.BioC.xml", SAVE_DIR + "test.txt")
    