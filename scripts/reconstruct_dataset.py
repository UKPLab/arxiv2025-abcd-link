import json
import pandas as pd


def reconstruct_dataset(spiced, news_he_sentence_boundaries):
    reversed_docs = ['246', '132', '118', '294', '333', '180', '360', '243']    
    output = {}

    for idx, data in news_he_sentence_boundaries.items():
        output[idx] = {"doc1": {}, "doc2": {}}
        if idx in reversed_docs:
            spiced_text1 = spiced.loc[int(idx)].to_dict()["text_2"]
            spiced_text2 = spiced.loc[int(idx)].to_dict()["text_1"]
        else:
            spiced_text1 = spiced.loc[int(idx)].to_dict()["text_1"]
            spiced_text2 = spiced.loc[int(idx)].to_dict()["text_2"]
        
        for sent_idx, sent_boundaries in data["doc1"].items():
            output[idx]["doc1"][sent_idx] = spiced_text1[sent_boundaries[0]:sent_boundaries[1]]
        for sent_idx, sent_boundaries in data["doc2"].items():
            output[idx]["doc2"][sent_idx] = spiced_text2[sent_boundaries[0]:sent_boundaries[1]]
    
    return output


if __name__ == "__main__":
    with open("./datasets/news_he/spiced.csv", "r") as f:
        spiced = pd.read_csv(f)

    with open("./datasets/news_he/sentence_boundaries.json", "r") as f:
        sentence_boundaries = json.load(f)

    reconstructed_dataset = reconstruct_dataset(spiced, sentence_boundaries)

    with open("./datasets/news_he/docs.json", "w") as f:
        json.dump(reconstructed_dataset, f, indent=4, ensure_ascii=False)