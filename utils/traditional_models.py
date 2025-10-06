import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, List

class TraditionalClassifier:
    def __init__(self, model_type: str = "resnet50"):
        """
        Initialize traditional pre-trained classifier
        
        Args:
            model_type: 'resnet50' or 'efficientnet_b0'
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        
        if model_type == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.num_classes = 1000
        elif model_type == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=True)
            self.num_classes = 1000
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load ImageNet class labels
        self.imagenet_labels = self._load_imagenet_labels()
    
    def _load_imagenet_labels(self) -> List[str]:
        """
        Load ImageNet class labels
        """

        try:
            import requests
            response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
            labels = response.text.split('\n')
            return [label.strip() for label in labels if label.strip()]
        except:
            # Fallback to generic labels
            return [f"class_{i}" for i in range(1000)]
    
    def predict(self, image: Image.Image, top_k: int = 5) -> Dict:
        """
        Predict using traditional model
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions
        """
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            results.append({
                'class': self.imagenet_labels[idx.item()],
                'probability': prob.item(),
                'rank': i + 1
            })
        
        return {
            'top_prediction': results[0]['class'],
            'top_probability': results[0]['probability'],
            'all_predictions': results,
            'model_used': f'{self.model_type} (ImageNet)'
        }