# Dataset structure

Each dataset folder (except for news_he, see below) has the following two JSON files:

- **docs.json** contains the raw documents split into sentences where the keys are the sentence IDs and the values the sentence text.

```text
{
    "DOCUMENT_ID_XYZ" : {
        "SENTENCE_INDEX_0" : "SENTENCE_TEXT",
        "SENTENCE_INDEX_1" : "SENTENCE_TEXT",
        ...
    },
    "DOCUMENT_ID_ABC" : {
        ...
    },
    ...
}
```

- ***_links.json** contains the mapping of sentences between the two related documents in the dataset (aka links). The source and target document ID are used to retrieve the sentence IDs and raw text from **docs.json**

```text
{
    "LINK_ID_XYZ" : {
        "doc1" : "ID_OF_SOURCE_DOCUMENT",
        "doc2" : "ID_OF_TARGET_DOCUMENT",
        "links" : {
            "SOURCE_SENTENCE_ID_A" : [
                "TARGET_SENTENCE_ID_X",
                "TARGET_SENTENCE_ID_Y",
            ],
            "SOURCE_SENTENCE_ID_B" : [
                "TARGET_SENTENCE_ID_XX",
                "TARGET_SENTENCE_ID_YY",
            ],
            ...
        }
    },
    ...
}
```

## NEWS_HE and REVIEWS_HE

In these two folders, you can also find the results of the manual annotations in the **annotations.csv** file. It has the following structure:

- document_id : the link id from **docs.json**
- source_id : the id of the source document
- target_id : the id of the target document
- method : the method used for suggesting this link for annotation (can be either llm, retrieval, both, or random)
- annotator_1 : the annotation by annotator 1
- annotator_2 : the annotation by annotator 2
- source_sentence : the raw text of the source sentence
- target_sentence : the raw text of the target sentence
- agreement : boolean value indicating if both annotators agreed
- both_link : boolean value indicating if both annotators annotated the link as true

## License

- news_ecb: CC BY 4.0
- news_synth: CC BY 4.0
- reviews_f1000: CC BY 4.0
- reviews_he: CC BY 4.0
- reviews_synth: CC BY 4.0

## Note on NEWS_HE

The **news_he** dataset uses documents from the [SPICED](https://zenodo.org/records/8044777) dataset, which has to be downloaded and processed manually due to licensing issues. Please check the [instructions in the main README](https://github.com/UKPLab/arxiv2025-ABCD-Link/blob/main/README.md#reconstruct-dataset).
