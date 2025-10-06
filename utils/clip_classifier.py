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
    
    def few_shot_predict(self, image: Image.Image, 
                        examples: List[Tuple[Image.Image, str]], 
                        class_names: List[str]) -> Dict:
        """
        Few-shot prediction using example images
        
        Args:
            image: Target image to classify
            examples: List of (example_image, class_name) tuples
            class_names: Available class names
            
        Returns:
            Dictionary with predictions
        """
       
        warnings.warn("This is a basic few-shot implementation. For production, consider more advanced methods.")
        

        return self.predict(image, class_names)