# 🔬 AI Skin Lesion Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)

An advanced AI-powered medical imaging system for automated skin lesion classification using EfficientNet-B7 architecture. Designed for educational and research purposes with medical-grade performance metrics and explainable AI features.

![System Status](https://img.shields.io/badge/Status-Production-success)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Information](#dataset-information)
- [Classification Categories](#classification-categories)
- [Technical Details](#technical-details)
- [Explainable AI](#explainable-ai)
- [Medical Disclaimer](#medical-disclaimer)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

The AI Skin Lesion Analyzer is a state-of-the-art deep learning system that classifies dermatological images into seven distinct lesion types. Built on the EfficientNet-B7 architecture and trained on the HAM10000 dataset, this system achieves medical-grade accuracy with explainable AI features for clinical transparency.

### Mission

To democratize access to advanced dermatological screening tools through AI, enabling early detection of potentially malignant skin lesions while maintaining the highest standards of accuracy, interpretability, and medical safety.

### Key Highlights

- **88.28%** Overall Classification Accuracy
- **85.2%** Sensitivity for Malignant Lesion Detection
- **93.4%** AUC Score for Malignant vs Benign Classification
- **Real-time** Analysis with Grad-CAM Visualization
- **Medical-grade** Risk Assessment System

## ✨ Key Features

### 🎯 Core Capabilities

- **Advanced AI Classification**: EfficientNet-B7 architecture with 66M parameters
- **Seven-Class Detection**: Comprehensive lesion type identification
- **Risk-Based Assessment**: Dynamic confidence thresholds based on medical risk levels
- **Grad-CAM Visualization**: Explainable AI showing model attention regions
- **Medical Recommendations**: Automated clinical guidance based on predictions
- **Comprehensive Analytics**: Multi-tab dashboard with detailed insights

### 📊 Advanced Analytics

- **Confidence Analysis**: Detailed probability distributions and entropy calculations
- **Clinical Reports**: Downloadable PDF and JSON reports with full analysis
- **Performance Metrics**: Real-time display of sensitivity, specificity, and AUC scores
- **Differential Diagnosis**: Top-3 predictions with risk stratification
- **Interactive Visualizations**: Plotly-powered charts and graphs

### 🔬 Medical Features

- **ABCDE Rule Integration**: Melanoma detection guidelines
- **Risk Stratification**: Three-tier risk classification (HIGH/MEDIUM/LOW)
- **Threshold-Based Confidence**: Medical-grade confidence levels per risk category
- **Educational Resources**: Prevention tips and warning signs
- **Professional Disclaimer**: Clear medical guidance and limitations

## 📈 Model Performance

### Overall Metrics

| Metric | Score | Improvement vs Baseline |
|--------|-------|------------------------|
| **Accuracy** | 88.28% | +4.48% |
| **F1-Score (Macro)** | 81.54% | +3.54% |
| **Precision (Macro)** | 82.70% | +3.20% |
| **Recall (Macro)** | 81.16% | +3.36% |

### Malignant Detection Performance

| Metric | Value | Clinical Significance |
|--------|-------|---------------------|
| **Sensitivity** | 85.2% | High true positive rate for dangerous lesions |
| **Specificity** | 91.7% | Low false positive rate for benign lesions |
| **NPV (Negative Predictive Value)** | 96.8% | High confidence when predicting benign |
| **AUC (Malignant Detection)** | 93.4% | Excellent discrimination ability |

### Per-Class Performance

| Lesion Type | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Melanocytic nevi (nv) | 92% | 89% | 90% |
| Melanoma (mel) | 75% | 71% | 73% |
| Benign keratosis (bkl) | 88% | 82% | 85% |
| Basal cell carcinoma (bcc) | 79% | 76% | 77% |
| Actinic keratoses (akiec) | 74% | 69% | 71% |
| Vascular lesions (vasc) | 85% | 81% | 83% |
| Dermatofibroma (df) | 91% | 87% | 89% |

## 🏗️ System Architecture

```
AI Skin Lesion Analyzer
│
├── Frontend (Streamlit)
│   ├── Image Upload Interface
│   ├── Real-time Analysis Dashboard
│   ├── Multi-tab Analytics
│   └── Report Generation
│
├── AI Pipeline
│   ├── EfficientNet-B7 Model (66M params)
│   ├── Image Preprocessing
│   ├── Inference Engine
│   └── Grad-CAM Generator
│
├── Risk Assessment System
│   ├── Confidence Threshold Optimizer
│   ├── Medical Recommendation Engine
│   └── Risk Stratification Module
│
└── Analytics & Reporting
    ├── Performance Metrics Dashboard
    ├── Clinical Report Generator (PDF/JSON)
    └── Interactive Visualizations
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, recommended for faster inference)
- 8GB+ RAM
- Modern web browser

### Step 1: Clone the Repository

```bash
git clone https://github.com/AtamerErkal/skin-lesion-classifier.git
cd skin-lesion-classifier
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Dependencies

```txt
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.5.0
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
plotly>=5.17.0
scikit-learn>=1.3.0
fpdf>=1.7.2
```

### Step 4: Download Model File

The trained model file `best_model.pth` should be placed in the root directory. Due to file size limitations:

**Option A: Download from Release**
- Download `best_model.pth` from the [Releases](https://github.com/AtamerErkal/skin-lesion-classifier/releases) page

**Option B: Git LFS (Recommended for developers)**
```bash
git lfs install
git lfs pull
```

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`

### 🌐 Live Demo

Try the deployed application: [**AI Skin Lesion Analyzer**](YOUR_STREAMLIT_CLOUD_LINK)

### Quick Start Guide

1. **Upload Image**
   - Click "Browse files" or drag-and-drop
   - Supported formats: JPG, JPEG, PNG
   - Recommended: Clear, well-lit dermatoscopic images

2. **View Analysis**
   - Original image and Grad-CAM visualization
   - AI prediction with confidence score
   - Risk assessment and medical recommendations

3. **Explore Analytics**
   - Tab 1: All prediction probabilities
   - Tab 2: Confidence analysis dashboard
   - Tab 3: Medical performance metrics
   - Tab 4: Clinical report (downloadable)
   - Tab 5: AI model insights

4. **Download Reports**
   - PDF clinical report
   - JSON data export
   - Original and Grad-CAM images

## 📊 Dataset Information

### HAM10000 Dataset

**Source**: Harvard Dataverse - Human Against Machine with 10,000 training images

**Characteristics**:
- **Total Images**: 10,015 dermatoscopic images
- **Collection Period**: 20 years of data
- **Image Types**: Dermatoscopic (clinical close-up with magnification)
- **Demographics**: Various age groups and skin types
- **Quality**: Medical-grade photography with standardized protocols

**Class Distribution**:
| Class | Count | Percentage |
|-------|-------|------------|
| nv (Melanocytic nevi) | 6,705 | 66.9% |
| mel (Melanoma) | 1,113 | 11.1% |
| bkl (Benign keratosis) | 1,099 | 11.0% |
| bcc (Basal cell carcinoma) | 514 | 5.1% |
| akiec (Actinic keratoses) | 327 | 3.3% |
| vasc (Vascular lesions) | 142 | 1.4% |
| df (Dermatofibroma) | 115 | 1.1% |

**Data Challenges**:
- Severe class imbalance (addressed via weighted loss and augmentation)
- Visual similarity between benign and malignant lesions
- Varying image quality and lighting conditions

## 🏥 Classification Categories

### 1. Melanocytic Nevi (nv) - BENIGN
**Common moles, typically harmless pigmented skin lesions**
- Most common type (~67% of cases)
- Risk Level: LOW
- Confidence Threshold: 80%
- Recommendation: Regular monitoring, annual dermatology check-ups

### 2. Melanoma (mel) - MALIGNANT
**Most dangerous form of skin cancer**
- Life-threatening if not detected early
- Risk Level: HIGH
- Confidence Threshold: 60%
- Recommendation: URGENT dermatologist consultation required

### 3. Benign Keratosis-like Lesions (bkl) - BENIGN
**Seborrheic keratosis, solar lentigo, lichen planus**
- Non-cancerous growths
- Risk Level: MEDIUM
- Confidence Threshold: 70%
- Recommendation: Monitor for changes, professional evaluation advised

### 4. Basal Cell Carcinoma (bcc) - MALIGNANT
**Most common form of skin cancer, slow-growing**
- Treatable with early detection
- Risk Level: HIGH
- Confidence Threshold: 60%
- Recommendation: Dermatologist appointment within 1-2 weeks

### 5. Actinic Keratoses (akiec) - PRE-MALIGNANT
**Precancerous lesions caused by sun exposure**
- Can progress to squamous cell carcinoma
- Risk Level: HIGH
- Confidence Threshold: 60%
- Recommendation: Professional evaluation within one month

### 6. Vascular Lesions (vasc) - BENIGN
**Angiomas, angiokeratomas, pyogenic granulomas**
- Blood vessel growths, usually harmless
- Risk Level: LOW
- Confidence Threshold: 80%
- Recommendation: Routine monitoring

### 7. Dermatofibroma (df) - BENIGN
**Fibrous tissue growth in the skin**
- Harmless, firm nodules
- Risk Level: LOW
- Confidence Threshold: 80%
- Recommendation: No treatment needed unless symptomatic

## 🔧 Technical Details

### Model Architecture

**EfficientNet-B7 Specifications**:
- **Parameters**: 66 million trainable parameters
- **Depth**: 45 layers with compound scaling
- **Width**: 2.0x width scaling factor
- **Resolution**: 600x600 native, 224x224 used for efficiency
- **Input Normalization**: ImageNet statistics

**Architecture Advantages**:
- Compound scaling balances depth, width, and resolution
- Mobile inverted bottleneck convolutions (MBConv)
- Squeeze-and-excitation blocks for channel attention
- Transfer learning from ImageNet pre-training

### Training Configuration

```python
TRAINING_PARAMS = {
    'epochs': 20,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'optimizer': 'AdamW',
    'scheduler': 'ReduceLROnPlateau',
    'loss_function': 'CrossEntropyLoss (weighted)',
    'weight_decay': 0.01,
    'dropout': 0.2
}
```

**Data Augmentation Pipeline**:
- Random rotation (±30°)
- Random horizontal and vertical flips
- Color jitter (brightness, contrast, saturation)
- Random erasing (cutout augmentation)
- Gaussian blur

**Training Strategy**:
- Transfer learning with ImageNet weights
- Progressive unfreezing of layers
- Class-weighted loss to handle imbalance
- Stratified K-fold cross-validation
- Early stopping with patience=5

### Preprocessing Pipeline

```python
PREPROCESSING = {
    'resize': 256,
    'center_crop': 224,
    'normalization_mean': [0.485, 0.456, 0.406],
    'normalization_std': [0.229, 0.224, 0.225]
}
```

### Risk-Based Confidence Thresholds

The system uses medical risk-appropriate confidence thresholds:

| Risk Level | Classes | Threshold | Rationale |
|-----------|---------|-----------|-----------|
| HIGH | mel, bcc, akiec | 60% | Lower threshold for dangerous lesions (prioritize sensitivity) |
| MEDIUM | bkl | 70% | Balanced threshold for benign monitoring |
| LOW | nv, vasc, df | 80% | Higher threshold for common benign lesions |

## 🔬 Explainable AI

### Grad-CAM (Gradient-weighted Class Activation Mapping)

**How It Works**:
1. Forward pass through the network
2. Compute gradients of target class with respect to feature maps
3. Weight feature maps by gradient importance
4. Generate heatmap showing pixel-level importance
5. Overlay heatmap on original image

**Interpretation Guide**:
- **Red/Yellow regions**: Highest importance for classification
- **Green regions**: Moderate importance
- **Blue/Cool regions**: Lower importance
- **Medical relevance**: Should focus on lesion borders, texture, color variations

**Clinical Value**:
- Validates model is using medically relevant features
- Builds trust with healthcare professionals
- Enables error analysis and model debugging
- Educational tool for understanding AI decision-making

## 🛡️ Medical Disclaimer

**CRITICAL NOTICE**: This AI system is designed exclusively for **educational and research purposes**. 

### Important Limitations

- ❌ NOT FDA/CE approved for clinical use
- ❌ NOT a substitute for professional medical diagnosis
- ❌ NOT for making treatment decisions
- ❌ NOT for unsupervised use by patients

### Proper Usage

- ✅ Educational demonstrations
- ✅ Research and development
- ✅ Healthcare professional assistance (non-diagnostic)
- ✅ Second opinion tool with physician oversight

### Medical Guidance

- **Always consult qualified dermatologists** for any suspicious skin lesions
- **Early detection saves lives** - when in doubt, see a doctor
- **Self-diagnosis is dangerous** - AI cannot replace clinical expertise
- **Regular skin examinations** by professionals are essential

## 📚 Educational Resources

### ABCDE Rule for Melanoma Detection

- **A - Asymmetry**: One half doesn't match the other
- **B - Border**: Irregular, ragged, notched, or blurred edges
- **C - Color**: Non-uniform color with multiple shades
- **D - Diameter**: Larger than 6mm (pencil eraser size)
- **E - Evolving**: Changes in size, shape, or color over time

### Prevention Tips

- Use broad-spectrum SPF 30+ sunscreen daily
- Seek shade during peak sun hours (10 AM - 4 PM)
- Wear protective clothing and wide-brimmed hats
- Avoid tanning beds and sun lamps
- Perform monthly self-skin examinations
- Schedule annual dermatologist check-ups

### Warning Signs

- Sore that doesn't heal within 2-3 weeks
- Pigment spreading beyond lesion border
- Redness or swelling around lesion
- Change in sensation (itching, tenderness, pain)
- Surface changes (scaling, oozing, bleeding)

## 🤝 Contributing

Contributions are welcome! This is an open-source educational project.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/ImprovementName`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/ImprovementName`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation
- Maintain medical accuracy and safety

### Areas for Contribution

- Model architecture improvements
- New visualization techniques
- Additional medical resources
- Performance optimization
- Bug fixes and error handling
- Documentation enhancements

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Additional Terms**:
- For educational and research use only
- Not for commercial medical applications
- Medical disclaimer must be retained
- Attribution required for derivative works

## 🔒 Privacy & Security

### Data Handling

- **No data storage**: Uploaded images are processed in memory only
- **No user tracking**: No personal information collected
- **Session-based**: Data cleared when browser is closed
- **Local processing**: Model runs locally, not sent to external servers

### Security Best Practices

- Images should be anonymized (no patient identifiers)
- Use in controlled educational environments
- Follow institutional data protection policies
- Comply with GDPR, HIPAA, and local regulations

## 📞 Contact

**Project Maintainer**: Atamer Erkal

- 📧 Email: atamererkal.eu@gmail.com
- 💼 LinkedIn: [Atamer Erkal](https://www.linkedin.com/in/atamererkal/)
- 🐙 GitHub: [@AtamerErkal](https://github.com/AtamerErkal)
- 🔗 Project: [https://github.com/AtamerErkal/skin-lesion-classifier](https://github.com/AtamerErkal/skin-lesion-classifier)

## 🙏 Acknowledgments

### Dataset & Research

- **HAM10000 Dataset**: Tschandl P., Rosendahl C., Kittler H. (2018)
- **Harvard Dataverse**: For hosting the dataset
- **International Skin Imaging Collaboration (ISIC)**

### Technology & Frameworks

- **PyTorch & Torchvision**: Deep learning framework
- **Streamlit**: Web application framework
- **EfficientNet**: Model architecture by Google Research
- **Plotly**: Interactive visualizations
- **OpenCV**: Computer vision tools

### Medical Resources

- **Skin Cancer Foundation**
- **American Academy of Dermatology**
- **Deutsche Krebsforschungszentrum (DKFZ)**
- **National Cancer Institute**

## 🗓️ Version History

- **v2.1** (Current) - Enhanced EfficientNet-B7 with Grad-CAM XAI
  - Improved accuracy from 83.8% to 88.28%
  - Advanced risk-based confidence thresholds
  - Clinical report generation (PDF/JSON)
  - Comprehensive analytics dashboard

- **v2.0** - EfficientNet-B7 upgrade
  - Migration from EfficientNet-B0
  - Enhanced model architecture
  - Improved malignant detection

- **v1.0** - Initial release
  - EfficientNet-B0 baseline
  - Basic classification interface
  - Grad-CAM visualization

## 📊 Future Roadmap

- [ ] Multi-model ensemble for improved accuracy
- [ ] Mobile app version (iOS/Android)
- [ ] Real-time video analysis
- [ ] Integration with dermoscopy devices
- [ ] Multi-language support
- [ ] Extended dataset training (ISIC archives)
- [ ] Federated learning for privacy-preserving training
- [ ] Clinical validation studies

---

**🔬 AI Skin Lesion Analyzer** | Medical-Grade Deep Learning | Educational Research Tool

*Built with ❤️ for advancing dermatological AI research and education*

**Version**: EfficientNet-B7-Enhanced-v2.1 | **Status**: Production | **License**: MIT (Educational Use)

---

### 📖 Citation

If you use this project in your research or education, please cite:

```bibtex
@software{erkal2025skinlesion,
  author = {Erkal, Atamer},
  title = {AI Skin Lesion Analyzer: Educational Deep Learning System},
  year = {2025},
  url = {https://github.com/AtamerErkal/skin-lesion-classifier},
  version = {2.1}
}
```
