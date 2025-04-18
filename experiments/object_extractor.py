import re
from typing import List, Dict, Any, Tuple, Set

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from difflib import SequenceMatcher
from nltk.corpus import wordnet
import nltk
from huggingface_hub import login
import torch
from sentence_transformers import SentenceTransformer, util

login("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")
import transformers
transformers.set_seed(42)

class ObjectExtractor:
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the object extractor with either spaCy or a transformer-based NER model.
        
        Args:
            use_spacy (bool): If True, use spaCy for NER. If False, use a transformer model.
        """
        self.use_spacy = use_spacy
        if use_spacy:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        else:
            model_id = "meta-llama/Llama-3.2-1B-Instruct"
            self.ner = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print("in object extractor ************ #######")
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")
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
        
        # Common object synonyms mapping (one-way)
        self.raw_synonyms = {
            'sofa': ['couch', 'settee'],
            'refrigerator': ['fridge', 'icebox'],
            'tv': ['television', 'televisions'],
            'remote': ['remote control'],
            'cellphone': ['mobile phone', 'phone', 'cellular phone'],
            'laptop': ['computer', 'notebook'],
            'cabinet': ['cupboard', 'closet'],
            'faucet': ['tap', 'spigot'],
            'sink': ['basin', 'washbasin'],
            'bed': ['mattress'],
            'pillow': ['cushion'],
            'blanket': ['comforter', 'quilt'],
            'chair': ['seat'],
            'table': ['desk'],
            'book': ['books'],
            'apple': ['apples'],
            'banana': ['bananas'],
            'orange': ['oranges'],
            'cup': ['mug', 'glass'],
            'plate': ['dish'],
            'bowl': ['container'],
            'fork': ['forks'],
            'spoon': ['spoons'],
            'knife': ['knives'],
            'door': ['doors'],
            'window': ['windows'],
            'wall': ['walls'],
            'floor': ['ground'],
            'ceiling': ['roof'],
            'light': ['lamp', 'lighting'],
            'switch': ['button', 'control'],
            'clock': ['timepiece'],
            'picture': ['photo', 'photograph', 'image'],
            'painting': ['art', 'artwork'],
            'mirror': ['looking glass'],
            'rug': ['carpet', 'mat'],
            'curtain': ['drape', 'drapery'],
            'shower': ['shower stall'],
            'bathtub': ['tub', 'bath'],
            'toilet': ['commode', 'throne'],
            'towel': ['washcloth'],
            'soap': ['bar soap'],
            'toothbrush': ['brush'],
            'toothpaste': ['paste'],
            'comb': ['hairbrush'],
            'hair dryer': ['dryer', 'blow dryer'],
            'trash can': ['garbage can', 'wastebasket', 'bin'],
            'basket': ['container'],
            'box': ['container'],
            'drawer': ['drawers'],
            'shelf': ['shelves'],
            'counter': ['countertop'],
            'stove': ['oven', 'range'],
            'microwave': ['microwave oven'],
            'dishwasher': ['dish washing machine'],
            'washer': ['washing machine'],
            'dryer': ['clothes dryer'],
            'vacuum': ['vacuum cleaner'],
            'broom': ['sweeper'],
            'mop': ['mop'],
            'dustpan': ['pan'],
            'sponge': ['sponge'],
            'cloth': ['rag', 'towel'],
            'brush': ['brush'],
            'scissors': ['scissor'],
            'pen': ['pencil'],
            'paper': ['papers'],
            'notebook': ['notepad'],
            'folder': ['file'],
            'backpack': ['bag', 'rucksack'],
            'purse': ['handbag', 'bag'],
            'wallet': ['billfold'],
            'key': ['keys'],
            'phone': ['telephone', 'cellphone', 'mobile'],
            'charger': ['power adapter'],
            'battery': ['batteries'],
            'headphones': ['earphones', 'earbuds'],
            'speaker': ['speakers'],
            'camera': ['cameras'],
            'printer': ['printers'],
            'scanner': ['scanners'],
            'monitor': ['screen', 'display'],
            'keyboard': ['keyboards'],
            'mouse': ['mice'],
            'mousepad': ['mouse pad'],
            'desk': ['table'],
            'chair': ['seat'],
            'bed': ['mattress'],
            'pillow': ['cushion'],
            'blanket': ['comforter', 'quilt'],
            'sheet': ['bed sheet'],
            'curtain': ['drape', 'drapery'],
            'rug': ['carpet', 'mat'],
            'mirror': ['looking glass'],
            'picture': ['photo', 'photograph', 'image'],
            'painting': ['art', 'artwork'],
            'clock': ['timepiece'],
            'light': ['lamp', 'lighting'],
            'switch': ['button', 'control'],
            'door': ['doors'],
            'window': ['windows'],
            'wall': ['walls'],
            'floor': ['ground'],
            'ceiling': ['roof'],
            'trash can': ['garbage can', 'wastebasket', 'bin', 'garbagecan'],
            'basket': ['container'],
            'box': ['container'],
            'drawer': ['drawers'],
            'shelf': ['shelves'],
            'counter': ['countertop'],
            'stove': ['oven', 'range'],
            'microwave': ['microwave oven'],
            'dishwasher': ['dish washing machine'],
            'washer': ['washing machine'],
            'dryer': ['clothes dryer'],
            'vacuum': ['vacuum cleaner'],
            'broom': ['sweeper'],
            'mop': ['mop'],
            'dustpan': ['pan'],
            'sponge': ['sponge'],
            'cloth': ['rag', 'towel'],
            'brush': ['brush'],
            'scissors': ['scissor'],
            'pen': ['pencil'],
            'paper': ['papers'],
            'notebook': ['notepad'],
            'folder': ['file'],
            'backpack': ['bag', 'rucksack'],
            'purse': ['handbag', 'bag'],
            'wallet': ['billfold'],
            'key': ['keys'],
            'phone': ['telephone', 'cellphone', 'mobile'],
            'charger': ['power adapter'],
            'battery': ['batteries'],
            'headphones': ['earphones', 'earbuds'],
            'speaker': ['speakers'],
            'camera': ['cameras'],
            'printer': ['printers'],
            'scanner': ['scanners'],
            'monitor': ['screen', 'display'],
            'keyboard': ['keyboards'],
            'mouse': ['mice'],
            'mousepad': ['mouse pad'],
        }
        
        # Create bidirectional synonym mapping
        self.common_synonyms = {}
        for key, values in self.raw_synonyms.items():
            # Add the main key and its synonyms
            if key not in self.common_synonyms:
                self.common_synonyms[key] = set()
            self.common_synonyms[key].update(values)
            
            # Add each synonym as a key with the main key and other synonyms
            for value in values:
                if value not in self.common_synonyms:
                    self.common_synonyms[value] = set()
                self.common_synonyms[value].add(key)
                # Add other synonyms to this synonym
                self.common_synonyms[value].update([v for v in values if v != value])
        
    def get_wordnet_synonyms(self, word: str) -> Set[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word (str): The word to get synonyms for
            
        Returns:
            Set[str]: Set of synonyms including the original word
        """
        word_lower = word.lower()
        synonyms = {word_lower}
        
        # Add common synonyms from our mapping
        if word_lower in self.common_synonyms:
            synonyms.update(self.common_synonyms[word_lower])
        
        # Add WordNet synonyms
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
                
        return synonyms
    
    def are_synonyms(self, word1: str, word2: str) -> bool:
        """
        Check if two words are synonyms.
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            
        Returns:
            bool: True if words are synonyms, False otherwise
        """
        word1_lower = word1.lower()
        word2_lower = word2.lower()
        
        # Direct match
        if word1_lower == word2_lower:
            return True
            
        # Check if either word is in the other's synonym set
        if word1_lower in self.common_synonyms and word2_lower in self.common_synonyms[word1_lower]:
            return True
            
        if word2_lower in self.common_synonyms and word1_lower in self.common_synonyms[word2_lower]:
            return True
            
        # Check WordNet synonyms
        word1_synonyms = self.get_wordnet_synonyms(word1)
        word2_synonyms = self.get_wordnet_synonyms(word2)
        
        return (word1_lower in word2_synonyms or 
                word2_lower in word1_synonyms or
                bool(word1_synonyms.intersection(word2_synonyms)))
    
    def calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using SequenceMatcher and synonym matching.
        
        Args:
            str1 (str): First string
            str2 (str): Second string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # First check for exact synonym match
        if self.are_synonyms(str1, str2):
            return 1.0
            
        # If no synonym match, use string similarity
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
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

            objects = [token.text for token in doc if token.pos_ == "NOUN"]
            breakpoint()
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
            messages = [
                        {"role": "system", "content": "You are a text processor. Produce an inclusive, comma-separated list of all object names including furniture or surfaces mentioned in the given text by the user. without extra words!"},
                        {"role": "user", "content": text},
                    ]
            ner_results = self.ner(messages, max_new_tokens=256, temperature =0.1)[0]["generated_text"][-1]['content']
            objects = []
            items = ner_results.split(",")

            # Optionally strip whitespace
            objects = [item.strip() for item in items]
        
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
        
    def find_best_match(self, obj: str, target_list: List[str], threshold: float = 0.8) -> Tuple[str, float]:
        """
        Find the best matching object in the target list.
        
        Args:
            obj (str): Object to match
            target_list (List[str]): List of target objects
            threshold (float): Minimum similarity threshold
            
        Returns:
            Tuple[str, float]: Best matching object and its similarity score
        """
        best_match = None
        best_score = 0.0
        
        for target in target_list:
            score = self.calculate_string_similarity(obj, target)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = target
                
        return best_match, best_score
    
    def calculate_overlap_score(self, source_objects: List[str], vlm_objects: List[str], 
                              threshold: float = 0.8) -> Tuple[float, Dict[str, str]]:
        """
        Calculate how many objects from the source list exist in the VLM list.
        
        Args:
            source_objects (List[str]): List of source objects
            vlm_objects (List[str]): List of VLM-detected objects
            threshold (float): Minimum similarity threshold for considering a match
            
        Returns:
            Tuple[float, Dict[str, str]]: 
                - Overlap score between 0 and 1
                - Dictionary mapping source objects to their best matches
        """
        if not source_objects:
            return 0.0, {}
            
        matches = {}
        total_score = 0.0
        
        for source_obj in source_objects:
            best_match, score = self.find_best_match(source_obj, vlm_objects, threshold)
            if best_match:
                matches[source_obj] = best_match
                total_score += score
                
        overlap_score = total_score / len(source_objects)
        return overlap_score, matches
    
    def calculate_overlap_score_transformer(self, source_objects, vlm_objects, threshold=0.4):
        # Encode both lists in batch
        source_objects = list(set(source_objects))
        vlm_objects = list(set(vlm_objects))
        if len(source_objects) == 0:
            return None, []
        elif len(source_objects)> 0 and len(vlm_objects) == 0:
            return 0, []
        source_embs = self.sim_model.encode(source_objects, convert_to_tensor=True, show_progress_bar=False)
        vlm_embs = self.sim_model.encode(vlm_objects, convert_to_tensor=True, show_progress_bar=False)

        # Compute cosine similarities for all pairs
        cosine_scores = util.cos_sim(source_embs, vlm_embs.to(source_embs.device))  # shape: [len(source), len(vlm)]

        matches = {}
        total_score = 0.0

        for i, source_obj in enumerate(source_objects):
            scores = cosine_scores[i]
            best_idx = torch.argmax(scores).item()
            best_score = scores[best_idx].item()
            if best_score >= threshold:
                matches[source_obj] = vlm_objects[best_idx]
                total_score += best_score

        overlap_score = total_score / len(source_objects)
        return overlap_score, list(matches.keys())


    def calculate_batch_overlap_scores(self, source_objects: List[List[str]], 
                                     vlm_objects: List[List[str]], 
                                     threshold: float = 0.8) -> List[Tuple[float, Dict[str, str]]]:
        """
        Calculate overlap scores for batches of object lists.
        
        Args:
            source_objects (List[List[str]]): List of source object lists
            vlm_objects (List[List[str]]): List of VLM object lists
            threshold (float): Minimum similarity threshold
            
        Returns:
            List[Tuple[float, Dict[str, str]]]: List of (score, matches) tuples
        """
        if len(source_objects) != len(vlm_objects):
            raise ValueError("Source and VLM object lists must have the same length")
            
        return [self.calculate_overlap_score_transformer(src, vlm, threshold) 
                for src, vlm in zip(source_objects, vlm_objects)] 