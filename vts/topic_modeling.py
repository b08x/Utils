import os
import sys
import progressbar
import spacy
from gensim import corpora
from gensim.models import LdaMulticore

parent_directory = os.path.abspath('..')

sys.path.append(parent_directory)


from vts import (
    audio_processing,
    metadata_generation,
    segment_analysis,
    topic_modeling,
    transcription,
    utils,
)

# Load spaCy model (move this to a setup function or main if possible)
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    print("Preprocessing text...")
    doc = nlp(text)
    subjects = []

    for sent in doc.sents:
        for token in sent:
            if "subj" in token.dep_:
                if token.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    subject = get_compound_subject(token)
                    subjects.append(subject)

    cleaned_subjects = [
        [
            token.lemma_.lower()
            for token in nlp(subject)
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        for subject in subjects
    ]

    cleaned_subjects = [
        list(s) for s in set(tuple(sub) for sub in cleaned_subjects) if s
    ]

    print(
        f"Text preprocessing complete. Extracted {len(cleaned_subjects)} unique subjects."
    )
    return cleaned_subjects


def get_compound_subject(token):
    subject = [token.text]
    for left_token in token.lefts:
        if left_token.dep_ == "compound":
            subject.insert(0, left_token.text)
    for right_token in token.rights:
        if right_token.dep_ == "compound":
            subject.append(right_token.text)
    return " ".join(subject)


def perform_topic_modeling(subjects, num_topics=5):
    print(f"Performing topic modeling with {num_topics} topics...")
    dictionary = corpora.Dictionary(subjects)
    corpus = [dictionary.doc2bow(subject) for subject in subjects]
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        chunksize=100,
        passes=10,
        per_word_topics=True,
    )
    print("Topic modeling complete.")
    return lda_model, corpus, dictionary


def identify_segments(transcript, lda_model, dictionary, num_topics):
    print("Identifying segments based on topics...")
    segments = []
    current_segment = {"start": 0, "end": 0, "content": "", "topic": None}

    for sentence in progressbar.progressbar(transcript):
        subjects = preprocess_text(sentence["content"])
        if not subjects:
            continue

        bow = dictionary.doc2bow([token for subject in subjects for token in subject])
        topic_dist = lda_model.get_document_topics(bow)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0] if topic_dist else None

        if dominant_topic != current_segment["topic"]:
            if current_segment["content"]:
                current_segment["end"] = sentence["start"]
                segments.append(current_segment)
            current_segment = {
                "start": sentence["start"],
                "end": sentence["end"],
                "content": sentence["content"],
                "topic": dominant_topic,
            }
        else:
            current_segment["end"] = sentence["end"]
            current_segment["content"] += " " + sentence["content"]

    if current_segment["content"]:
        segments.append(current_segment)

    print(f"Identified {len(segments)} segments.")
    return segments