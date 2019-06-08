# -*- coding: utf-8 -*-

'''
Traditional Chinese medicine (TCM) literature corpus (in short,
TCM corpus) (32): The abstracts of all 106,150 papers published in
the 114 most popular Chinese TCM journals between 2011 to 2016
are collected. 3024 herbs, 4957 formulae, 1126 syndromes, and 1650
diseases are found. 5 types of relations are annotated. 
The entire dataset is available online at http://arnetminer.org/TCMRelExtr.

This script make TCM literature corpus into BERT input file format.
guid\ttarget\tcontext\tlabel\n

For example:

1889	四君子汤 糖尿病	黄芪 四君子汤 加减 治疗 气阴 两虚型 妊娠 糖尿病 30 例 临床 观察   目的 : 观察 黄芪 四君子汤 加减 治疗 气阴 两虚型 妊娠 糖尿病 的 临床 疗效 。 方法 : 将 60 例本 病患者 随机 分为 两组 , 治疗 组 30 例 予以 黄芪 四君子汤 加减 治疗 , 对照组 30 例 予以 饮食疗法 。 治疗 结束 后 进行 空腹 血糖 、 餐后 2   h 血糖 及 中医 证候 疗效 比较 。 结果 : 治疗 后 7   d 治疗 组 空腹 血糖 、 餐后 2   h 血糖 较 治疗 前 下降 , 差异 具有 统计学 意义 ( P < 0.05 ) ; 对照组 空腹 血糖 较 治疗 前 下降 , 差异 无 统计学 意义 ( P > 0.05 ) , 餐后 2   h 血糖 较 治疗 前 下降 , 差异 有 统计学 意义 ( P < 0.05 ) ; 治疗 后 空腹 血糖 、 餐后 2   h 血糖 两组 差值 比较 差异 有 统计学 意义 ( P < 0.01 ) ; 治疗 组总 有效率 93.3% , 对照组 总 有效率 76.7% , 治疗 组 中医 证候 疗效 优于 对照组 , 差异 有 统计学 意义 ( P < 0.05 ) 。 结论 : 黄芪 四君子汤 加减 治疗 气阴 两虚型 妊娠 糖尿病 有 较 好 疗效 , 可 有效 控制 患者 血糖 , 改善 患者 的 临床 症状 。	1

"Jieba"  Chinese text segmentation from https://github.com/fxsjy/jieba is needed.

Then, Five fold.
'''

import pandas as pd
import jieba
import numpy as np

CORPUS_DIR = "./corpus/TCM/TCMRelationExtraction/"
SAVE_DIR = "./corpus/TCM/"

jieba.load_userdict(SAVE_DIR+"dict.txt")
literature = pd.read_csv(
    CORPUS_DIR + "tcm_literature.txt", sep='\t', index_col="ID")
# print(literature[:3])


def data_process(in_file, relation_type, entity_name1, entity_name2,  text_encoding="utf-16LE"):
    """Make TCM corpus into BERT format and 5 fold.
        Args:
            in_file: string. The full/relative path of the TCM literature corpus file will be processed.
            relation_type: string. One of five relation types in TCM literature corpus, e.g. 
                            FD (formula-disease), SD (syndrome-disease), HD (herb-syndrom), 
                            FS (formula-syndrome), HS (herb-syndrome).
            entity_name1: string. The first entity is one column in TCM literature corpus, e.g. FormulaName, SyndromeName, HerbName. 
            entity_name2: string. The second entity is another column in TCM literature corpus, e.g. DiseaseName, SyndromeName. 
            text_encoding: string. The encoding of TCM literature corpus files, "utf-16LE" or "utf-8"
    """
    # Step 1: Make traditional Chinese medicine (TCM) literature corpus into BERT input file format.
    # guid\ttarget\tcontext\tlabel\n

    df = pd.read_csv(in_file, sep='\t', index_col="ID", encoding=text_encoding)
    # print(df[:3])
    data = df.loc[df['Label'] != "?"]
    # print(data)  #[788 rows x 5 columns]
    resultList = []

    for i, r in data.iterrows():
        entity1 = r[entity_name1]
        entity2 = r[entity_name2]
        label = r["Label"]
        # print(formula, disease)
        lineNO = 0
        for i_literature, r_literature in literature.iterrows():
            # print(r["Title"], r["Abstract"])
            s = str(r_literature["Title"] + " " + r_literature["Abstract"])
            if s.find(entity1) >= 0 and s.find(entity2) >= 0:
                s = " ".join(jieba.cut(s, cut_all=False))
                resultList.append(str(i) + "\t" + entity1 + " " +
                                  entity2 + "\t" + s + "\t" + label + "\n")
                lineNO += 1
                if lineNO > 1:
                    print(relation_type + " " + str(i) +
                          " " + entity1 + "\t" + entity2)
    print(in_file + " finished!")

    # Step 2: Five fold the output file of step 1.
    k = 5
    ids = np.arange(len(resultList) - 1)
    # print(ids)
    np.random.shuffle(ids)
    # print(ids)
    chunk_num = int((len(resultList) - 1)/k)

    for i in range(k):
        print(relation_type + " fold " + str(i))
        trainID = []
        testID = ids[i*chunk_num: (i+1) * chunk_num]
        for id in ids:
            if id not in testID:
                trainID.append(id)
        # print(trainID)
        # print(testID)
        # print(len(trainID))
        # print(len(testID))
        outFile_training = open(
            SAVE_DIR + relation_type + "_fold" + str(i+1) + "_training.txt", "w")
        outFile_test = open(SAVE_DIR + relation_type +
                            "_fold" + str(i+1) + "_test.txt", "w")
        for id in trainID:
            outFile_training.write(resultList[id])
        for id in testID:
            outFile_test.write(resultList[id])


if __name__ == '__main__':
    data_process(CORPUS_DIR + "candicate-relation_formula2disease.txt",
                 "FD", "FormulaName", "DiseaseName")
    data_process(CORPUS_DIR + "candicate-relation_syndrome2disease.txt",
                 "SD", "SyndromeName", "DiseaseName")
    data_process(CORPUS_DIR + "candicate-relation_herb2disease.txt",
                 "HD", "HerbName", "DiseaseName")
    data_process(CORPUS_DIR + "candicate-relation_formula2syndrome.txt",
                 "FS", "FormulaName", "SyndromeName", "utf-8")
    data_process(CORPUS_DIR + "candicate-relation_herb2syndrome.txt",
                 "HS", "HerbName", "SyndromeName")
