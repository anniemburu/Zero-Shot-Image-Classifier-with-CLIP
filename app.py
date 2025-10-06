import streamlit as st
import torch
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from utils.clip_classifier import ZeroShotCLIPClassifier
from utils.traditional_models import TraditionalClassifier
import numpy as np
import io

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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    #Initialize session state variables

    if 'clip_model' not in st.session_state:
        st.session_state.clip_model = None
    if 'traditional_models' not in st.session_state:
        st.session_state.traditional_models = {}
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

def load_models():
    #Load models

    with st.spinner("Loading CLIP model... This may take a moment. Have some icecream 🍦"):
        if st.session_state.clip_model is None:
            st.session_state.clip_model = ZeroShotCLIPClassifier()
    
    traditional_models = ['resnet50', 'efficientnet_b0']
    for model_name in traditional_models:
        if model_name not in st.session_state.traditional_models:
            with st.spinner(f"Loading {model_name}..."):
                st.session_state.traditional_models[model_name] = TraditionalClassifier(model_name)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ Zero-Shot Image Classifier with CLIP</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    clip_model_variant = st.sidebar.selectbox(
        "CLIP Model Variant",
        ["ViT-B/32", "ViT-B/16", "RN50"],
        help="Select CLIP model architecture"
    )
    
    # Classification options
    st.sidebar.subheader("Classification Options")
    custom_classes = st.sidebar.text_area(
        "Custom Classes (comma-separated)",
        "cat, dog, car, tree, person, building, bird, flower, food, mountain",
        help="Enter classes you want to classify against"
    )
    
    # Parse classes
    class_names = [cls.strip() for cls in custom_classes.split(',') if cls.strip()]
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📤 Upload Image</div>', unsafe_allow_html=True)
        
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
            
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"Image size: {image.size}, Mode: {image.mode}")
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Classification Results</div>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_image is not None and class_names:
            
            load_models() # Load models if not already loaded
            
            # Classify button
            if st.button("🚀 Classify Image", type="primary"):
                with st.spinner("Classifying image..."):
                    # CLIP prediction
                    clip_results = st.session_state.clip_model.predict(
                        st.session_state.uploaded_image, 
                        class_names
                    )
                    
                    # Display CLIP results
                    st.markdown("### CLIP Zero-Shot Results")
                    display_prediction_results(clip_results)
                    
                    # Traditional models comparison
                    st.markdown("### Traditional Models Comparison")
                    compare_traditional_models(st.session_state.uploaded_image)
                    
                    # Visualization
                    st.markdown("### 📊 Results Visualization")
                    create_comparison_visualization(clip_results, class_names)

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

def create_comparison_visualization(clip_results, class_names):
    """Create visualization of classification results"""
    # Prepare data for plotting
    classes = [pred['class'] for pred in clip_results['all_predictions']]
    probabilities = [pred['probability'] for pred in clip_results['all_predictions']]
    
    # Create bar chart
    fig = px.bar(
        x=probabilities,
        y=classes,
        orientation='h',
        title="CLIP Classification Probabilities",
        labels={'x': 'Probability', 'y': 'Class'},
        color=probabilities,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Advanced features section
def show_advanced_features():
    """Show advanced features in an expander"""
    with st.expander("🔬 Advanced Features"):
        st.markdown("### Few-Shot Learning Demo")
        
        st.info("""
        **Few-shot learning** allows the model to learn from a small number of examples. 
        While CLIP is inherently zero-shot, we can demonstrate the concept using example-based classification.
        """)
        
        # Example of few-shot interface
        if st.session_state.uploaded_image is not None:
            st.write("Upload example images for few-shot learning (conceptual demo):")
            
            example_files = st.file_uploader(
                "Upload example images for few-shot learning",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="few_shot_examples"
            )
            
            if example_files and len(example_files) > 0:
                st.write(f"Uploaded {len(example_files)} example images")
                # In a real implementation, you would process these examples
                # for few-shot learning

if __name__ == "__main__":
    main()
    show_advanced_features()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using CLIP, PyTorch, and Streamlit | "
        "[Report Issues](https://github.com/your-username/clip-classifier/issues)"
    )

    st.markdown("Created by **Anne Mburu**")