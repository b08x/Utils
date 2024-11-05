import spacy
from gensim import corpora
from gensim.models import LdaMulticore
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

from vts.config import Settings
from vts.models import Topic, Segment

logger = logging.getLogger(__name__)

class TopicAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text: str) -> List[List[str]]:
        doc = self.nlp(text)
        subjects = []

        for sent in doc.sents:
            for token in sent:
                if "subj" in token.dep_:
                    subject = self._get_compound_subject(token)
                    subjects.append(subject)

        return [
            [token.lemma_.lower() for token in self.nlp(subject)
             if not token.is_stop and not token.is_punct and token.is_alpha]
            for subject in subjects
        ]

    def _get_compound_subject(self, token) -> str:
        subject = [token.text]
        for left_token in token.lefts:
            if left_token.dep_ == "compound":
                subject.insert(0, left_token.text)
        for right_token in token.rights:
            if right_token.dep_ == "compound":
                subject.append(right_token.text)
        return " ".join(subject)

    def build_topic_model(
        self, 
        subjects: List[List[str]]
    ) -> Tuple[LdaMulticore, corpora.Dictionary]:
        dictionary = corpora.Dictionary(subjects)
        corpus = [dictionary.doc2bow(subject) for subject in subjects]

        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.settings.DEFAULT_NUM_TOPICS,
            random_state=100,
            chunksize=100,
            passes=10,
            per_word_topics=True
        )

        return model, dictionary

    def get_dominant_topic(
        self, 
        text: str, 
        model: LdaMulticore, 
        dictionary: corpora.Dictionary
    ) -> Tuple[Optional[int], float]:
        subjects = self.preprocess_text(text)
        if not subjects:
            return None, 0.0

        bow = dictionary.doc2bow([token for subject in subjects for token in subject])
        topic_dist = model.get_document_topics(bow)
        
        if not topic_dist:
            return None, 0.0
            
        return max(topic_dist, key=lambda x: x[1])

    def generate_topic_summary(
        self, 
        model: LdaMulticore, 
        segments: List[Segment]
    ) -> List[Topic]:
        topics = []
        for topic_id in range(model.num_topics):
            keywords = [word for word, _ in model.show_topic(topic_id, topn=10)]
            topic_segments = [s for s in segments if s.topic_id == topic_id]
            weight = len(topic_segments) / len(segments) if segments else 0.0
            
            topics.append(Topic(
                id=topic_id,
                keywords=keywords,
                weight=weight,
                segments=topic_segments
            ))
    
        return topics

    def analyze_coherence(
        self, 
        model: LdaMulticore, 
        dictionary: corpora.Dictionary, 
        texts: List[List[str]]
    ) -> float:
        """Analyze topic model coherence"""
        from gensim.models.coherencemodel import CoherenceModel
        
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()