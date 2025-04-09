import re
from typing import List, Dict, Any
import spacy
from transformers import pipeline

class ObjectExtractor:
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the object extractor with either spaCy or a transformer-based NER model.
        
        Args:
            use_spacy (bool): If True, use spaCy for NER. If False, use a transformer model.
        """
        self.use_spacy = use_spacy
        if use_spacy:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.ner = pipeline("ner", model="dslim/bert-base-NER")
            
        # Common object-related words to filter out
        self.object_indicators = {
            'a', 'an', 'the', 'this', 'that', 'these', 'those',
            'is', 'are', 'was', 'were', 'be', 'being', 'been',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'and', 'or', 'but', 'nor', 'so', 'yet',
            'there', 'here', 'where', 'when', 'why', 'how',
            'it', 'its', 'it\'s', 'they', 'their', 'there',
            'can', 'could', 'will', 'would', 'shall', 'should',
            'may', 'might', 'must', 'need', 'used'
        }
        
    def extract_objects_from_text(self, text: str) -> List[str]:
        """
        Extract object names from a text description.
        
        Args:
            text (str): The text description to extract objects from.
            
        Returns:
            List[str]: List of extracted object names.
        """
        # Clean and normalize text
        text = text.lower().strip()
        
        if self.use_spacy:
            # Use spaCy for Named Entity Recognition
            doc = self.nlp(text)
            objects = []
            
            # Extract noun phrases and named entities
            for chunk in doc.noun_chunks:
                # Filter out common words and indicators
                if not any(word in chunk.text.lower() for word in self.object_indicators):
                    objects.append(chunk.text.strip())
                    
            for ent in doc.ents:
                if ent.label_ in ['PRODUCT', 'ORG', 'PERSON', 'GPE']:
                    objects.append(ent.text.strip())
                    
        else:
            # Use transformer-based NER
            ner_results = self.ner(text)
            objects = []
            
            current_entity = []
            for token in ner_results:
                if token['entity'].startswith('B-'):
                    if current_entity:
                        objects.append(' '.join(current_entity))
                    current_entity = [token['word']]
                elif token['entity'].startswith('I-'):
                    current_entity.append(token['word'])
                    
            if current_entity:
                objects.append(' '.join(current_entity))
        
        # Remove duplicates and clean up
        objects = list(set(objects))
        objects = [obj.strip() for obj in objects if obj.strip()]
        
        return objects
    
    def extract_objects_from_vlm_description(self, vlm_description: Dict[str, Any]) -> List[str]:
        """
        Extract objects from a VLM description dictionary.
        
        Args:
            vlm_description (Dict[str, Any]): The VLM description dictionary containing text.
            
        Returns:
            List[str]: List of extracted object names.
        """
        if not isinstance(vlm_description, dict) or 'text' not in vlm_description:
            return []
            
        text = vlm_description['text']
        return self.extract_objects_from_text(text)
    
    def extract_objects_from_batch(self, descriptions: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Extract objects from a batch of VLM descriptions.
        
        Args:
            descriptions (List[Dict[str, Any]]): List of VLM description dictionaries.
            
        Returns:
            List[List[str]]: List of lists containing extracted object names for each description.
        """
        return [self.extract_objects_from_vlm_description(desc) for desc in descriptions] 