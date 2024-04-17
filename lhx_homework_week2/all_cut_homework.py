#分词方法：最大正向切分的第一种实现方式

import re
import time

#加载词典
def load_word_dict(path):
    max_word_length = 0
    word_dict = {}  #用set也是可以的。用list会很慢
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            word_dict[word] = 0
            max_word_length = max(max_word_length, len(word))
    return word_dict, max_word_length


def full_segmentation(dict, corpus, start=0, segments=[]):
    if start == len(corpus):
        print("/".join(segments))
        return
    for end in range(start + 1, len(corpus) + 1):
        if corpus[start:end] in dict:
            full_segmentation(dict, corpus, end, segments + [corpus[start:end]])

#cut_method是切割函数
#output_path是输出路径
def main(full_segmentation, input_path, output_path):
    word_dict, max_word_length = load_word_dict("dict.txt")
    writer = open(output_path, "w", encoding="utf8")
    start_time = time.time()
    with open(input_path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            segments = []
            full_segmentation(word_dict, line, 0, segments)
            writer.write(" / ".join(segments) + "\n")
    writer.close()
    print("耗时：", time.time() - start_time)
    return

main(full_segmentation, "corpus.txt", "cut_method1_output.txt")


string = "测试字符串"
word_dict, max_len = load_word_dict("dict.txt")
# print(cut_method1(string, word_dict, max_len))

main(full_segmentation, "corpus.txt", "cut_method1_output.txt")
