import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import warnings

class ZeroShotCLIPClassifier:
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize CLIP model for zero-shot classification
        
        Args:
            model_name: CLIP model variant ('ViT-B/32', 'ViT-B/16', 'RN50', etc.)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model_name = model_name
        
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image).unsqueeze(0).to(self.device) #preprocess img (resize to 224x224 and normalize)
    
    def get_text_embeddings(self, class_names: List[str]) -> torch.Tensor:
        #class name embedding
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)

        return text_features

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Compute image embeddings for a list of PIL images
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            Tensor of shape (N, embedding_dim)
        """
        # Preprocess all images and stack them into a batch
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
        
        return image_features
    
    def predict(self, image: Image.Image, class_names: List[str]) -> Dict:
        """
        Predict class probabilities for given image and classes
        
        Args:
            image: PIL Image object
            class_names: List of candidate class names
            
        Returns:
            Dictionary with predictions and probabilities
        """

        # Preprocess inputs
        image_input = self.preprocess_image(image)
        text_features = self.get_text_embeddings(class_names)
        
        # Calculate similarity
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity (cosine similarity)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probabilities = similarity[0].cpu().numpy()
        
        # Get top predictions
        results = []
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            results.append({
                'class': class_name,
                'probability': float(prob),
                'rank': i + 1
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'top_prediction': results[0]['class'],
            'top_probability': results[0]['probability'],
            'all_predictions': results,
            'model_used': f'CLIP {self.model_name}'
        }
    
    def few_shot_predict(self, 
                        image: Image.Image, 
                        examples: Dict[str, List[Image.Image]], 
                        class_names: List[str],
                        method: str = "prototype") -> Dict:
        """
        Few-shot prediction using example images
        
        Args:
            image: Target image to classify
            examples: Dictionary {class_name: [list of example images]}
            class_names: Available class names
            method: Few-shot method - "prototype" or "similarity"
            
        Returns:
            Dictionary with predictions
        """
        if method == "prototype":
            return self._prototype_few_shot(image, examples, class_names)
        elif method == "similarity":
            return self._similarity_few_shot(image, examples, class_names)
        else:
            raise ValueError("Method must be 'prototype' or 'similarity'")
    
    def _prototype_few_shot(self, 
                          image: Image.Image, 
                          examples: Dict[str, List[Image.Image]], 
                          class_names: List[str]) -> Dict:
        """
        Prototype-based few-shot learning
        Creates class prototypes from example images and compares target image to prototypes
        """
        # Get embeddings for all example images
        class_prototypes = {}
        
        for class_name in class_names:
            if class_name in examples and len(examples[class_name]) > 0:
                # Get embeddings for all examples of this class
                example_embeddings = self.get_image_embeddings(examples[class_name])
                
                # Create prototype by averaging embeddings
                prototype = example_embeddings.mean(dim=0)
                prototype = prototype / prototype.norm(dim=-1, keepdim=True)
                class_prototypes[class_name] = prototype
        
        # If no examples provided for some classes, use zero-shot for those
        if len(class_prototypes) < len(class_names):
            # Get zero-shot predictions for all classes
            zero_shot_results = self.predict(image, class_names)
            zero_shot_probs = {pred['class']: pred['probability'] 
                             for pred in zero_shot_results['all_predictions']}
        
        # Get target image embedding
        target_embedding = self.get_image_embeddings([image])[0]
        target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        
        # Calculate similarities to prototypes
        results = []
        for class_name in class_names:
            if class_name in class_prototypes:
                # Use few-shot prototype similarity
                similarity = (100.0 * target_embedding @ class_prototypes[class_name]).item()
                probability = torch.sigmoid(torch.tensor(similarity / 100.0)).item()
            else:
                # Fall back to zero-shot for classes without examples
                probability = zero_shot_probs[class_name]
            
            results.append({
                'class': class_name,
                'probability': probability,
                'method': 'few-shot' if class_name in class_prototypes else 'zero-shot'
            })
        
        # Normalize probabilities to sum to 1
        total_prob = sum(r['probability'] for r in results)
        for result in results:
            result['probability'] /= total_prob
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'top_prediction': results[0]['class'],
            'top_probability': results[0]['probability'],
            'all_predictions': results,
            'model_used': f'CLIP {self.model_name} (Few-Shot)',
            'method_used': 'prototype'
        }
    
    def _similarity_few_shot(self, 
                           image: Image.Image, 
                           examples: Dict[str, List[Image.Image]], 
                           class_names: List[str]) -> Dict:
        """
        Similarity-based few-shot learning
        Compares target image to each example image and aggregates similarities
        """
        # Get target image embedding
        target_embedding = self.get_image_embeddings([image])[0]
        target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        
        # Calculate average similarity to examples for each class
        class_similarities = {}
        
        for class_name in class_names:
            if class_name in examples and len(examples[class_name]) > 0:
                # Get embeddings for all examples of this class
                example_embeddings = self.get_image_embeddings(examples[class_name])
                example_embeddings = example_embeddings / example_embeddings.norm(dim=-1, keepdim=True)
                
                # Calculate similarities to all examples
                similarities = (100.0 * target_embedding @ example_embeddings.T)
                avg_similarity = similarities.mean().item()
                
                # Convert to probability using softmax over classes
                class_similarities[class_name] = avg_similarity
        
        # If no examples for some classes, use zero-shot
        if len(class_similarities) < len(class_names):
            zero_shot_results = self.predict(image, class_names)
            zero_shot_probs = {pred['class']: pred['probability'] 
                             for pred in zero_shot_results['all_predictions']}
        
        # Combine similarities and convert to probabilities
        results = []
        for class_name in class_names:
            if class_name in class_similarities:
                # Use few-shot similarity
                similarity = class_similarities[class_name]
                probability = torch.sigmoid(torch.tensor(similarity / 100.0)).item()
            else:
                # Fall back to zero-shot
                probability = zero_shot_probs[class_name]
            
            results.append({
                'class': class_name,
                'probability': probability,
                'method': 'few-shot' if class_name in class_similarities else 'zero-shot'
            })
        
        # Normalize probabilities
        total_prob = sum(r['probability'] for r in results)
        for result in results:
            result['probability'] /= total_prob
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'top_prediction': results[0]['class'],
            'top_probability': results[0]['probability'],
            'all_predictions': results,
            'model_used': f'CLIP {self.model_name} (Few-Shot)',
            'method_used': 'similarity'
        }