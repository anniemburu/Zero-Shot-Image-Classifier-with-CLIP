import streamlit as st
import torch
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from utils.clip_classifier import ZeroShotCLIPClassifier
from utils.traditional_models import TraditionalClassifier
import numpy as np
import io
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="Zero-Shot Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #0d0c0c;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .example-image {
        border: 2px solid #ddd;
        border-radius: 5px;
        margin: 5px;
        padding: 5px;
    }
    .few-shot-active {
        border-color: #ff7f0e;
        background-color: #fff7e6;
    }
    .model-info {
        background-color: #0d0c0c;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    
    #Initialize session state variables
   
    if 'clip_models' not in st.session_state:
        st.session_state.clip_models = {}
    if 'traditional_models' not in st.session_state:
        st.session_state.traditional_models = {}
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'few_shot_examples' not in st.session_state:
        st.session_state.few_shot_examples = {}
    if 'few_shot_method' not in st.session_state:
        st.session_state.few_shot_method = "prototype"
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "ViT-B/32"

def load_clip_model(model_name: str) -> ZeroShotCLIPClassifier:
    # Load a specific CLIP model
    if model_name not in st.session_state.clip_models:
        with st.spinner(f"Loading {model_name} model... This may take a moment."):
            st.session_state.clip_models[model_name] = ZeroShotCLIPClassifier(model_name)

    return st.session_state.clip_models[model_name]

def load_traditional_models():
    #Load traditional models
    traditional_models = ['resnet50', 'efficientnet_b0']
    for model_name in traditional_models:
        if model_name not in st.session_state.traditional_models:
            with st.spinner(f"Loading {model_name}..."):
                st.session_state.traditional_models[model_name] = TraditionalClassifier(model_name)

def get_model_info(model_name: str) -> Dict:
    #Get information about different CLIP models

    model_info = {
        "ViT-B/32": {
            "description": "Vision Transformer Base (32x32 patches)",
            "speed": "Fastest",
            "accuracy": "Good",
            "size": "Medium",
            "params": "~150M",
            "best_for": "General purpose, quick inference"
        },
        "ViT-B/16": {
            "description": "Vision Transformer Base (16x16 patches)",
            "speed": "Medium",
            "accuracy": "Better",
            "size": "Large",
            "params": "~150M",
            "best_for": "Higher accuracy requirements"
        },
        "RN50": {
            "description": "ResNet-50 Backbone",
            "speed": "Fast",
            "accuracy": "Good",
            "size": "Medium",
            "params": "~100M",
            "best_for": "Balanced speed and accuracy"
        }
    }
    return model_info.get(model_name, {})

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ Zero-Shot Image Classifier with CLIP</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    available_models = ["ViT-B/32", "ViT-B/16", "RN50"]
    
    # Model selection with info
    selected_model = st.sidebar.selectbox(
        "CLIP Model Variant",
        available_models,
        index=available_models.index(st.session_state.selected_model),
        help="Select CLIP model architecture"
    )
    
    # Update selected model in session state
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # Clear results when model changes
        if 'last_results' in st.session_state:
            del st.session_state.last_results
    
    # Show model information
    model_info = get_model_info(selected_model)
    if model_info:
        st.sidebar.markdown(f"""
        <div class="model-info">
            <strong>Model Info:</strong><br>
            • {model_info['description']}<br>
            • Speed: {model_info['speed']}<br>
            • Accuracy: {model_info['accuracy']}<br>
            • Best for: {model_info['best_for']}
        </div>
        """, unsafe_allow_html=True)
    
    # Classification options
    st.sidebar.subheader("Classification Options")
    custom_classes = st.sidebar.text_area(
        "Custom Classes (comma-separated)",
        "cat, dog, car, tree, person, building, bird, flower, food, mountain",
        help="Enter classes you want to classify against"
    )
    
    # Parse classes
    class_names = [cls.strip() for cls in custom_classes.split(',') if cls.strip()]
    
    # Performance tips
    st.sidebar.subheader("💡 Performance Tips")
    st.sidebar.info(
        "• **ViT-B/32**: Fastest, good for general use\n"
        "• **ViT-B/16**: Slower but more accurate\n"
        "• **RN50**: Good balance, familiar architecture"
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📤 Upload Image</div>', unsafe_allow_html=True)
        
        # Show current model
        st.info(f"**Selected Model:** {st.session_state.selected_model}")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload an image to classify"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.info(f"Image size: {image.size}, Mode: {image.mode}")
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Classification Results</div>', unsafe_allow_html=True)
        
        # Show current model in results section too
        st.write(f"**Using Model:** `{st.session_state.selected_model}`")
        
        if st.session_state.uploaded_image is not None and class_names:
            # Load selected model
            clip_model = load_clip_model(st.session_state.selected_model)
            load_traditional_models()
            
            # Classify button
            if st.button("🚀 Classify Image", type="primary"):
                with st.spinner(f"Classifying image with {st.session_state.selected_model}..."):
                    # Determine if we're doing few-shot or zero-shot
                    has_few_shot_examples = any(
                        len(st.session_state.few_shot_examples.get(cls, [])) > 0 
                        for cls in class_names
                    )
                    
                    if has_few_shot_examples:
                        # Few-shot classification
                        results = clip_model.few_shot_predict(
                            st.session_state.uploaded_image,
                            st.session_state.few_shot_examples,
                            class_names,
                            method=st.session_state.few_shot_method
                        )
                        
                        # Display few-shot results
                        st.markdown("### 🔬 Few-Shot CLIP Results")
                        display_few_shot_results(results)
                        
                    else:
                        # Zero-shot classification
                        results = clip_model.predict(
                            st.session_state.uploaded_image, 
                            class_names
                        )
                        
                        # Display CLIP results
                        st.markdown("### CLIP Zero-Shot Results")
                        display_prediction_results(results)
                    
                    # Store results for visualization
                    st.session_state.last_results = results
                    st.session_state.last_class_names = class_names
                    st.session_state.last_has_few_shot = has_few_shot_examples
                    
                    # Traditional models comparison
                    st.markdown("### Traditional Models Comparison")
                    compare_traditional_models(st.session_state.uploaded_image)
            
            # Show visualization if results exist
            if 'last_results' in st.session_state:
                st.markdown("### 📊 Results Visualization")
                create_comparison_visualization(
                    st.session_state.last_results, 
                    st.session_state.last_class_names,
                    st.session_state.last_has_few_shot
                )

def display_prediction_results(results: dict):
    """Display prediction results in a nice format"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🏆 Top Prediction</h3>
            <h4>{results['top_prediction']}</h4>
            <p>Confidence: {results['top_probability']:.2%}</p>
            <p><small>Model: {results['model_used']}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**All Predictions:**")
        for pred in results['all_predictions'][:5]:  # Show top 5
            st.progress(pred['probability'])
            st.write(f"{pred['class']}: {pred['probability']:.2%}")

def display_few_shot_results(results: dict):
    """Display few-shot prediction results with method information"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🔬 Few-Shot Prediction</h3>
            <h4>{results['top_prediction']}</h4>
            <p>Confidence: {results['top_probability']:.2%}</p>
            <p><small>Method: {results['method_used']}</small></p>
            <p><small>Model: {results['model_used']}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**All Predictions (Few-Shot vs Zero-Shot):**")
        for pred in results['all_predictions'][:5]:
            method_color = "🟠" if pred.get('method') == 'few-shot' else "🔵"
            method_text = " (Few-Shot)" if pred.get('method') == 'few-shot' else " (Zero-Shot)"
            st.progress(pred['probability'])
            st.write(f"{method_color} {pred['class']}: {pred['probability']:.2%}{method_text}")

def compare_traditional_models(image):
    """Compare with traditional models"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ResNet-50")
        with st.spinner("Running ResNet..."):
            resnet_results = st.session_state.traditional_models['resnet50'].predict(image)
            st.write(f"**Top Prediction:** {resnet_results['top_prediction']}")
            st.write(f"**Confidence:** {resnet_results['top_probability']:.2%}")
    
    with col2:
        st.markdown("#### EfficientNet-B0")
        with st.spinner("Running EfficientNet..."):
            efficient_results = st.session_state.traditional_models['efficientnet_b0'].predict(image)
            st.write(f"**Top Prediction:** {efficient_results['top_prediction']}")
            st.write(f"**Confidence:** {efficient_results['top_probability']:.2%}")

def create_comparison_visualization(results, class_names, has_few_shot=False):
    """Create visualization of classification results"""
    # Prepare data for plotting
    classes = [pred['class'] for pred in results['all_predictions']]
    probabilities = [pred['probability'] for pred in results['all_predictions']]
    
    # Determine colors based on method (for few-shot) or use single color
    if has_few_shot:
        colors = ['#ff7f0e' if pred.get('method') == 'few-shot' else '#1f77b4' 
                  for pred in results['all_predictions']]
        color_discrete_map = {'#ff7f0e': 'Few-Shot', '#1f77b4': 'Zero-Shot'}
    else:
        colors = '#1f77b4'
        color_discrete_map = None
    
    # color map for each method
    colors_map = {
        "Few-Shot": "orange",
        "Zero-Shot": "royalblue"
    }

    # color list
    bar_colors = [colors_map.get(c, "gray") for c in colors]
    
    fig = px.bar(
        x=probabilities,
        y=classes,
        orientation='h',
        title=f"Classification Probabilities - {results['model_used']}",
        labels={'x': 'Probability', 'y': 'Class'}
    )

    fig.update_traces(marker_color=bar_colors)
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        showlegend=has_few_shot
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_advanced_features():
    with st.expander("🔬 Advanced Few-Shot Learning", expanded=False):
        st.markdown("### Few-Shot Learning Configuration")
        
        st.info("""
        **Few-shot learning** allows the model to learn from a small number of examples. 
        Upload example images for each class to improve classification accuracy!
        """)
        
        # Show current model in advanced section
        st.write(f"**Current Model:** `{st.session_state.selected_model}`")
        
        # Few-shot method selection
        st.session_state.few_shot_method = st.selectbox(
            "Few-Shot Method",
            ["prototype", "similarity"],
            help="Prototype: Average of example embeddings. Similarity: Compare to each example."
        )
        
        st.write("### Upload Example Images for Each Class")
        
        # Get classes from main section
        if st.session_state.uploaded_image is not None:
            # Try to get classes from the main input
            custom_classes = st.session_state.get('class_names', ["cat", "dog", "car", "tree"])
        else:
            custom_classes = ["cat", "dog", "car", "tree"]
        
        # Create file uploaders for each class
        for class_name in custom_classes:
            st.write(f"**{class_name.capitalize()} Examples:**")
            
            # File uploader for this class
            example_files = st.file_uploader(
                f"Upload {class_name} examples",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                accept_multiple_files=True,
                key=f"few_shot_{class_name}_{st.session_state.selected_model}"  # Include model in key to reset on change
            )
            
            if example_files:
                # Store images in session state
                if class_name not in st.session_state.few_shot_examples:
                    st.session_state.few_shot_examples[class_name] = []
                
                # Clear existing examples and add new ones
                st.session_state.few_shot_examples[class_name] = [Image.open(file) for file in example_files]
                
                # Display example images
                if st.session_state.few_shot_examples[class_name]:
                    cols = st.columns(min(3, len(st.session_state.few_shot_examples[class_name])))
                    for idx, example_img in enumerate(st.session_state.few_shot_examples[class_name][:3]):
                        with cols[idx]:
                            st.image(example_img, width=100, caption=f"Example {idx+1}")
                    
                    if len(st.session_state.few_shot_examples[class_name]) > 3:
                        st.info(f"... and {len(st.session_state.few_shot_examples[class_name]) - 3} more examples")
            
            else:
                # Clear examples if no files uploaded
                if class_name in st.session_state.few_shot_examples:
                    st.session_state.few_shot_examples[class_name] = []
        
        # Show current few-shot status
        st.markdown("### Current Few-Shot Setup")
        few_shot_classes = [cls for cls in custom_classes 
                          if cls in st.session_state.few_shot_examples 
                          and len(st.session_state.few_shot_examples[cls]) > 0]
        
        if few_shot_classes:
            st.success(f"✅ Few-shot learning enabled for: {', '.join(few_shot_classes)}")
            st.write(f"Using **{st.session_state.few_shot_method}** method with **{st.session_state.selected_model}** model")
        else:
            st.warning("⚠️ No few-shot examples uploaded. Using zero-shot classification.")

def show_model_comparison():
    """Show model comparison in an expander"""
    with st.expander("🔍 Model Comparison Guide", expanded=False):
        st.markdown("### CLIP Model Comparison")
        
        comparison_data = {
            "Model": ["ViT-B/32", "ViT-B/16", "RN50"],
            "Speed": ["⚡⚡⚡ Fastest", "⚡⚡ Medium", "⚡⚡⚡ Fast"],
            "Accuracy": ["🎯 Good", "🎯🎯 Better", "🎯 Good"],
            "Best For": ["Quick prototyping", "High accuracy needs", "Balanced use"],
            "Parameters": ["~150M", "~150M", "~100M"]
        }
        
        # Create a nice comparison table
        for i, model in enumerate(comparison_data["Model"]):
            with st.container():
                cols = st.columns([1, 2, 2, 3, 2])
                with cols[0]:
                    st.write(f"**{model}**")
                with cols[1]:
                    st.write(comparison_data["Speed"][i])
                with cols[2]:
                    st.write(comparison_data["Accuracy"][i])
                with cols[3]:
                    st.write(comparison_data["Best For"][i])
                with cols[4]:
                    st.write(comparison_data["Parameters"][i])
                st.markdown("---")

if __name__ == "__main__":
    main()
    show_advanced_features()
    show_model_comparison()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using CLIP, PyTorch, and Streamlit | "
        "Model Selection & Few-Shot Learning"
    )
    st.markdown("Made by Anne Mburu 🌹")