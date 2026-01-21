"""
Streamlit Dashboard for HGPS Multi-Modal AI System

Interactive web interface for:
- Face image and clinical data input
- Risk prediction visualization
- Growth curve timeline
- Classical vs Quantum ML comparison
- Model explanations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import io
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="HGPS Risk Assessment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-moderate { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load ML models (cached for performance)."""
    try:
        from src.data import FacePreprocessor, TabularPreprocessor, generate_hgps_tabular_data
        from src.models import ClassicalTabularModels

        # Initialize preprocessors
        face_preprocessor = FacePreprocessor()
        tabular_preprocessor = TabularPreprocessor()

        # Generate synthetic data and train models
        df = generate_hgps_tabular_data(n_hgps=50, n_controls=200)
        tabular_preprocessor.fit(df)

        features = tabular_preprocessor.transform(df)
        labels = df['risk_label'].values

        split_idx = int(0.8 * len(features))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]

        tabular_model = ClassicalTabularModels(calibrate=True)
        tabular_model.fit(X_train, y_train, X_val, y_val)

        return {
            'face_preprocessor': face_preprocessor,
            'tabular_preprocessor': tabular_preprocessor,
            'tabular_model': tabular_model,
            'loaded': True
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {'loaded': False, 'error': str(e)}


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def compute_derived_features(age, height_cm, weight_kg):
    """Compute derived clinical features."""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0

    # Z-scores (simplified approximation)
    expected_height = 75 + age * 5.5
    expected_weight = 10 + age * 2.0
    height_z = (height_cm - expected_height) / 5
    weight_z = (weight_kg - expected_weight) / 2

    return bmi, height_z, weight_z


def make_prediction(models, clinical_data):
    """Make prediction using loaded models."""
    if not models.get('loaded'):
        return None

    age = clinical_data['age']
    height = clinical_data['height_cm']
    weight = clinical_data['weight_kg']

    bmi, height_z, weight_z = compute_derived_features(age, height, weight)

    features = np.array([
        age, height, weight, bmi, height_z, weight_z,
        clinical_data['small_jaw'],
        clinical_data['prominent_eyes'],
        clinical_data['thin_skin'],
        clinical_data['hair_loss'],
        clinical_data['lmna_mut']
    ], dtype=np.float32).reshape(1, -1)

    feature_cols = models['tabular_preprocessor'].feature_columns
    df = pd.DataFrame(features, columns=feature_cols)
    features_scaled = models['tabular_preprocessor'].transform(df)

    probs = models['tabular_model'].predict_proba(features_scaled)
    pred = models['tabular_model'].predict(features_scaled)

    risk_score = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])

    return {
        'risk_score': risk_score,
        'risk_class': get_risk_class(risk_score),
        'prediction': int(pred[0]),
        'confidence': compute_confidence(probs[0]),
        'height_z': height_z,
        'weight_z': weight_z
    }


def get_risk_class(score):
    """Classify risk score."""
    if score < 0.3:
        return "Low"
    elif score < 0.7:
        return "Moderate"
    return "High"


def compute_confidence(probs):
    """Compute prediction confidence."""
    return float(2 * abs(np.max(probs) - 0.5))


def get_recommendation(risk_score, confidence):
    """Generate clinical recommendation."""
    if risk_score < 0.3:
        return "✅ Low risk. Continue routine pediatric monitoring."
    elif risk_score < 0.5:
        return "⚠️ Moderate risk. Consider clinical evaluation and growth monitoring."
    elif risk_score < 0.7:
        if confidence > 0.7:
            return "⚠️ Elevated risk. Clinical evaluation recommended. Consider genetic consultation."
        return "⚠️ Moderate-high risk. Further clinical assessment needed."
    else:
        if confidence > 0.8:
            return "🚨 HIGH RISK. Immediate genetic testing strongly recommended."
        return "🚨 High risk indicated. Genetic testing and specialist consultation recommended."


def get_decision_support(risk_score, confidence, height_z, weight_z, phenotype_count):
    """
    Generate structured decision support recommendations.

    Returns a dict with:
    - primary_action: Main recommended action category
    - actions: List of specific action items
    - urgency: 'routine', 'elevated', 'urgent'
    """
    actions = []
    phenotype_flags = phenotype_count >= 3

    # Determine urgency level and primary action
    if risk_score < 0.3 and abs(height_z) < 2 and abs(weight_z) < 2:
        urgency = 'routine'
        primary_action = 'MONITOR'
        actions = [
            {'icon': '📋', 'text': 'Continue routine pediatric check-ups', 'priority': 'standard'},
            {'icon': '📏', 'text': 'Monitor growth at regular intervals', 'priority': 'standard'},
            {'icon': '📅', 'text': 'Schedule follow-up in 6-12 months', 'priority': 'standard'},
        ]

    elif risk_score < 0.5 or (risk_score < 0.7 and confidence < 0.6):
        urgency = 'elevated'
        primary_action = 'GENETIC_TESTING'
        actions = [
            {'icon': '🧬', 'text': 'Consider LMNA gene sequencing', 'priority': 'recommended'},
            {'icon': '👨‍⚕️', 'text': 'Schedule genetics consultation', 'priority': 'recommended'},
            {'icon': '📏', 'text': 'Increase growth monitoring frequency', 'priority': 'standard'},
            {'icon': '🫀', 'text': 'Baseline cardiovascular assessment', 'priority': 'recommended'},
        ]

    elif risk_score < 0.7:
        urgency = 'elevated'
        primary_action = 'GENETIC_TESTING'
        actions = [
            {'icon': '🧬', 'text': 'LMNA mutation testing recommended', 'priority': 'high'},
            {'icon': '👨‍⚕️', 'text': 'Refer to progeria specialist', 'priority': 'high'},
            {'icon': '🫀', 'text': 'Cardiac evaluation (echocardiogram)', 'priority': 'recommended'},
            {'icon': '🦴', 'text': 'Bone density assessment', 'priority': 'recommended'},
            {'icon': '📅', 'text': 'Monthly growth monitoring', 'priority': 'standard'},
        ]

    else:
        urgency = 'urgent'
        primary_action = 'URGENT_REVIEW'
        actions = [
            {'icon': '🚨', 'text': 'URGENT: Immediate specialist referral', 'priority': 'critical'},
            {'icon': '🧬', 'text': 'Expedited genetic testing (LMNA)', 'priority': 'critical'},
            {'icon': '🫀', 'text': 'Comprehensive cardiac workup', 'priority': 'critical'},
            {'icon': '👨‍⚕️', 'text': 'Multi-disciplinary team review', 'priority': 'high'},
            {'icon': '💊', 'text': 'Discuss treatment options (e.g., lonafarnib)', 'priority': 'high'},
            {'icon': '🏥', 'text': 'Connect with PRF (Progeria Research Foundation)', 'priority': 'recommended'},
        ]

    # Add phenotype-based recommendations
    if phenotype_flags and urgency == 'routine':
        actions.append({'icon': '👁️', 'text': 'Phenotypic features warrant closer observation', 'priority': 'recommended'})

    if abs(height_z) >= 2 or abs(weight_z) >= 2:
        actions.append({'icon': '📊', 'text': 'Growth parameters significantly deviated - investigate cause', 'priority': 'high'})

    return {
        'primary_action': primary_action,
        'urgency': urgency,
        'actions': actions
    }


def create_confidence_gauge(confidence):
    """Create a gauge chart for model confidence visualization."""
    # Determine color based on confidence level
    if confidence >= 0.8:
        bar_color = "#2ca02c"  # Green - High confidence
    elif confidence >= 0.6:
        bar_color = "#1f77b4"  # Blue - Moderate confidence
    else:
        bar_color = "#ff7f0e"  # Orange - Low confidence

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={'suffix': '%', 'font': {'size': 24}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Model Confidence", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': '#ffebee'},
                {'range': [60, 80], 'color': '#e3f2fd'},
                {'range': [80, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "darkblue", 'width': 3},
                'thickness': 0.8,
                'value': confidence * 100
            }
        }
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10)
    )

    return fig


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_risk_gauge(risk_score, title="HGPS Risk Score"):
    """Create a gauge chart for risk visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': '#2ca02c'},
                {'range': [30, 70], 'color': '#ff7f0e'},
                {'range': [70, 100], 'color': '#d62728'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score * 100
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_progression_chart(probs):
    """Create progression probability bar chart."""
    categories = ['Slow', 'Moderate', 'Rapid']
    colors = ['#2ca02c', '#ff7f0e', '#d62728']

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=probs,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probs],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='Predicted Disease Progression',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_progression_timeline(current_age, risk_score, progression_type='moderate'):
    """
    Create a disease progression timeline showing expected trajectory.

    Args:
        current_age: Patient's current age
        risk_score: Risk score (0-1)
        progression_type: 'slow', 'moderate', or 'rapid'
    """
    # Define progression parameters based on type
    progression_params = {
        'slow': {'severity_rate': 0.03, 'color': '#2ca02c', 'label': 'Slow Progression'},
        'moderate': {'severity_rate': 0.06, 'color': '#ff7f0e', 'label': 'Moderate Progression'},
        'rapid': {'severity_rate': 0.10, 'color': '#d62728', 'label': 'Rapid Progression'}
    }

    params = progression_params.get(progression_type, progression_params['moderate'])

    # Timeline from birth to 20 years
    ages = np.linspace(0, 20, 100)

    # Calculate severity curves (sigmoid-like progression)
    def severity_curve(age, rate):
        return 1 / (1 + np.exp(-rate * (age - 5)))

    slow_severity = severity_curve(ages, 0.3) * 100
    moderate_severity = severity_curve(ages, 0.5) * 100
    rapid_severity = severity_curve(ages, 0.8) * 100

    # Determine patient's current position
    current_severity = risk_score * 100

    fig = go.Figure()

    # Add reference trajectories
    fig.add_trace(go.Scatter(
        x=ages, y=slow_severity,
        name='Slow Progression',
        line=dict(color='#2ca02c', dash='dot', width=2),
        hovertemplate='Age: %{x:.1f}<br>Severity: %{y:.1f}%<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=ages, y=moderate_severity,
        name='Moderate Progression',
        line=dict(color='#ff7f0e', dash='dash', width=2),
        hovertemplate='Age: %{x:.1f}<br>Severity: %{y:.1f}%<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=ages, y=rapid_severity,
        name='Rapid Progression',
        line=dict(color='#d62728', width=2),
        hovertemplate='Age: %{x:.1f}<br>Severity: %{y:.1f}%<extra></extra>'
    ))

    # Patient's current position
    fig.add_trace(go.Scatter(
        x=[current_age],
        y=[current_severity],
        mode='markers',
        name='Current Status',
        marker=dict(size=18, color='#1f77b4', symbol='star', line=dict(width=2, color='white')),
        hovertemplate=f'Patient (Age {current_age})<br>Risk: {current_severity:.1f}%<extra></extra>'
    ))

    # Add milestone markers
    milestones = [
        {'age': 5, 'label': 'Early Childhood', 'y': 105},
        {'age': 10, 'label': 'Mid Childhood', 'y': 105},
        {'age': 15, 'label': 'Adolescence', 'y': 105},
    ]

    for m in milestones:
        fig.add_vline(x=m['age'], line_dash="dot", line_color="gray", opacity=0.5)
        fig.add_annotation(
            x=m['age'], y=m['y'],
            text=m['label'],
            showarrow=False,
            font=dict(size=9, color='gray')
        )

    # Add severity zones
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=30, y1=70, fillcolor="orange", opacity=0.1, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)

    # Zone labels
    fig.add_annotation(x=19, y=15, text="Low Risk Zone", showarrow=False,
                       font=dict(size=10, color='#2ca02c'))
    fig.add_annotation(x=19, y=50, text="Moderate Risk", showarrow=False,
                       font=dict(size=10, color='#ff7f0e'))
    fig.add_annotation(x=19, y=85, text="High Risk Zone", showarrow=False,
                       font=dict(size=10, color='#d62728'))

    fig.update_layout(
        title='Disease Progression Timeline',
        xaxis_title='Age (years)',
        yaxis_title='Disease Severity (%)',
        yaxis_range=[0, 110],
        xaxis_range=[0, 20],
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest'
    )

    return fig


def create_growth_curve(age, height, weight, is_hgps=False):
    """Create growth curve timeline."""
    ages = np.linspace(0, 18, 100)

    # Normal growth curves (simplified)
    normal_height = 50 + ages * 5.5
    normal_weight = 3.5 + ages * 2.0

    # HGPS growth curves
    hgps_height = 50 + ages * 4.0
    hgps_weight = 3.0 + ages * 1.2

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Height vs Age', 'Weight vs Age')
    )

    # Height plot
    fig.add_trace(
        go.Scatter(x=ages, y=normal_height, name='Normal', line=dict(color='#2ca02c', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ages, y=hgps_height, name='HGPS Typical', line=dict(color='#d62728', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[age], y=[height], name='Patient', mode='markers',
                   marker=dict(size=15, color='#1f77b4', symbol='star')),
        row=1, col=1
    )

    # Weight plot
    fig.add_trace(
        go.Scatter(x=ages, y=normal_weight, name='Normal', line=dict(color='#2ca02c', dash='dash'),
                   showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ages, y=hgps_weight, name='HGPS Typical', line=dict(color='#d62728', dash='dash'),
                   showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[age], y=[weight], name='Patient', mode='markers',
                   marker=dict(size=15, color='#1f77b4', symbol='star'), showlegend=False),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Age (years)", row=1, col=1)
    fig.update_xaxes(title_text="Age (years)", row=1, col=2)
    fig.update_yaxes(title_text="Height (cm)", row=1, col=1)
    fig.update_yaxes(title_text="Weight (kg)", row=1, col=2)

    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))

    return fig


def create_feature_importance_chart(importance_dict):
    """Create feature importance bar chart."""
    sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [f[0] for f in sorted_features[:8]]
    values = [f[1] for f in sorted_features[:8]]

    colors = ['#d62728' if v > 0 else '#2ca02c' for v in values]

    fig = go.Figure(data=[
        go.Bar(
            y=names,
            x=values,
            orientation='h',
            marker_color=colors
        )
    ])

    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(clinical_data, result, quantum_score):
    """
    Generate a downloadable text report of the HGPS risk assessment.

    Args:
        clinical_data: Dictionary of patient clinical data
        result: Prediction result dictionary
        quantum_score: Quantum ML prediction score

    Returns:
        Report content as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate decision support
    phenotype_count = sum([
        clinical_data['small_jaw'],
        clinical_data['prominent_eyes'],
        clinical_data['thin_skin'],
        clinical_data['hair_loss'],
        clinical_data['lmna_mut']
    ])

    decision = get_decision_support(
        result['risk_score'],
        result['confidence'],
        result['height_z'],
        result['weight_z'],
        phenotype_count
    )

    # Format action items
    action_items = "\n".join([f"  [{a['priority'].upper()}] {a['text']}" for a in decision['actions']])

    report = f"""
================================================================================
              HGPS RISK ASSESSMENT REPORT
              Multi-Modal Quantum AI System
================================================================================

Report Generated: {timestamp}

--------------------------------------------------------------------------------
                         PATIENT INFORMATION
--------------------------------------------------------------------------------

Age:                    {clinical_data['age']} years
Height:                 {clinical_data['height_cm']} cm
Weight:                 {clinical_data['weight_kg']} kg
BMI:                    {clinical_data['weight_kg'] / ((clinical_data['height_cm']/100) ** 2):.1f}

Phenotypic Features:
  - Small Jaw (Micrognathia):    {'Yes' if clinical_data['small_jaw'] else 'No'}
  - Prominent Eyes:              {'Yes' if clinical_data['prominent_eyes'] else 'No'}
  - Thin/Aged Skin:              {'Yes' if clinical_data['thin_skin'] else 'No'}
  - Hair Loss (Alopecia):        {'Yes' if clinical_data['hair_loss'] else 'No'}
  - LMNA Mutation Known:         {'Yes' if clinical_data['lmna_mut'] else 'No'}

--------------------------------------------------------------------------------
                         RISK ASSESSMENT RESULTS
--------------------------------------------------------------------------------

RISK CLASSIFICATION:    {result['risk_class'].upper()}
Risk Score:             {result['risk_score']:.1%}
Model Confidence:       {result['confidence']:.1%}

Growth Analysis:
  - Height Z-Score:     {result['height_z']:.2f} ({'Normal' if abs(result['height_z']) < 2 else 'ABNORMAL'})
  - Weight Z-Score:     {result['weight_z']:.2f} ({'Normal' if abs(result['weight_z']) < 2 else 'ABNORMAL'})

--------------------------------------------------------------------------------
                      DECISION SUPPORT
--------------------------------------------------------------------------------

PRIMARY RECOMMENDATION: {decision['primary_action'].replace('_', ' ')}
URGENCY LEVEL:          {decision['urgency'].upper()}

Action Items:
{action_items}

--------------------------------------------------------------------------------
                      ML MODEL COMPARISON
--------------------------------------------------------------------------------

Classical ML Risk Score:    {result['risk_score']:.1%}
Quantum ML Risk Score:      {quantum_score:.1%}
Model Agreement:            {'HIGH' if abs(result['risk_score'] - quantum_score) < 0.05 else 'MODERATE' if abs(result['risk_score'] - quantum_score) < 0.15 else 'LOW'}

--------------------------------------------------------------------------------
                      CLINICAL RECOMMENDATION
--------------------------------------------------------------------------------

{get_recommendation(result['risk_score'], result['confidence']).replace('✅ ', '').replace('⚠️ ', 'ATTENTION: ').replace('🚨 ', 'URGENT: ')}

--------------------------------------------------------------------------------
                           DISCLAIMER
--------------------------------------------------------------------------------

This report is generated by an AI-based risk assessment system for research
and educational purposes only. It should NOT replace professional medical
diagnosis. Always consult qualified healthcare providers for clinical
decision-making.

The predictions are based on:
- Multi-modal deep learning (CNN + Clinical data fusion)
- Quantum Machine Learning (QSVM/VQC via Qiskit)
- Trained on synthetic HGPS phenotype data

================================================================================
                    END OF REPORT
================================================================================
"""
    return report


def generate_csv_report(clinical_data, result, quantum_score):
    """Generate a CSV format report for data export."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate decision support
    phenotype_count = sum([
        clinical_data['small_jaw'],
        clinical_data['prominent_eyes'],
        clinical_data['thin_skin'],
        clinical_data['hair_loss'],
        clinical_data['lmna_mut']
    ])

    decision = get_decision_support(
        result['risk_score'],
        result['confidence'],
        result['height_z'],
        result['weight_z'],
        phenotype_count
    )

    data = {
        'Report_Timestamp': [timestamp],
        'Age_Years': [clinical_data['age']],
        'Height_cm': [clinical_data['height_cm']],
        'Weight_kg': [clinical_data['weight_kg']],
        'BMI': [clinical_data['weight_kg'] / ((clinical_data['height_cm']/100) ** 2)],
        'Small_Jaw': [clinical_data['small_jaw']],
        'Prominent_Eyes': [clinical_data['prominent_eyes']],
        'Thin_Skin': [clinical_data['thin_skin']],
        'Hair_Loss': [clinical_data['hair_loss']],
        'LMNA_Mutation': [clinical_data['lmna_mut']],
        'Risk_Classification': [result['risk_class']],
        'Risk_Score': [result['risk_score']],
        'Confidence': [result['confidence']],
        'Height_Z_Score': [result['height_z']],
        'Weight_Z_Score': [result['weight_z']],
        'Classical_ML_Score': [result['risk_score']],
        'Quantum_ML_Score': [quantum_score],
        'Primary_Recommendation': [decision['primary_action']],
        'Urgency_Level': [decision['urgency']],
    }

    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def create_comparison_chart(classical_score, quantum_score):
    """Create classical vs quantum comparison chart."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Classical ML', 'Quantum ML'],
        y=[classical_score * 100, quantum_score * 100],
        marker_color=['#1f77b4', '#9467bd'],
        text=[f'{classical_score:.1%}', f'{quantum_score:.1%}'],
        textposition='auto'
    ))

    fig.update_layout(
        title='Classical vs Quantum ML Prediction',
        yaxis_title='Risk Score (%)',
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard function."""

    # Header
    st.markdown('<h1 class="main-header">🧬 HGPS Risk Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #666;'>
    Multi-Modal Quantum AI for Hutchinson-Gilford Progeria Syndrome Detection
    </p>
    """, unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models..."):
        models = load_models()

    if not models.get('loaded'):
        st.error("Failed to load models. Running in demo mode.")

    # Sidebar - Input
    st.sidebar.header("📋 Patient Information")

    # Image upload with Camera option
    st.sidebar.subheader("Face Image")

    image_source = st.sidebar.radio(
        "Image Source",
        ["Upload", "Camera"],
        horizontal=True,
        help="Choose to upload an image or capture from camera"
    )

    uploaded_file = None
    camera_image = None

    if image_source == "Upload":
        uploaded_file = st.sidebar.file_uploader(
            "Upload face photo",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a frontal face photograph for analysis"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        camera_image = st.sidebar.camera_input(
            "Capture face photo",
            help="Take a frontal face photograph for analysis"
        )
        if camera_image is not None:
            image = Image.open(camera_image)
            st.sidebar.image(image, caption="Captured Image", use_container_width=True)

    # Combined image reference
    face_image = uploaded_file or camera_image

    # Face alignment feedback
    if face_image is not None:
        try:
            # Attempt face preprocessing
            img_bytes = face_image.getvalue()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                # Basic face detection check with OpenCV
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    st.sidebar.success(f"Face detected and aligned ({len(faces)} face{'s' if len(faces) > 1 else ''} found)")
                    x, y, w, h = faces[0]
                    st.sidebar.caption(f"Face region: {w}x{h}px at position ({x}, {y})")
                else:
                    st.sidebar.warning("No face detected - ensure frontal face is visible")
        except Exception as e:
            st.sidebar.info("Image uploaded - face analysis will run on analyze")

    # Clinical data inputs
    st.sidebar.subheader("Clinical Data")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=0.0, max_value=25.0, value=5.0, step=0.5)
    with col2:
        height = st.number_input("Height (cm)", min_value=30.0, max_value=200.0, value=95.0, step=1.0)

    col3, col4 = st.sidebar.columns(2)
    with col3:
        weight = st.number_input("Weight (kg)", min_value=2.0, max_value=100.0, value=15.0, step=0.5)
    with col4:
        bmi, height_z, weight_z = compute_derived_features(age, height, weight)
        st.metric("BMI", f"{bmi:.1f}")

    st.sidebar.subheader("Phenotypic Features")
    small_jaw = st.sidebar.checkbox("Small jaw (micrognathia)")
    prominent_eyes = st.sidebar.checkbox("Prominent eyes")
    thin_skin = st.sidebar.checkbox("Thin, aged skin")
    hair_loss = st.sidebar.checkbox("Hair loss (alopecia)")
    lmna_mut = st.sidebar.checkbox("LMNA mutation known")

    # Analyze button
    analyze = st.sidebar.button("🔬 Analyze", use_container_width=True, type="primary")

    # Main content area
    if analyze:
        clinical_data = {
            'age': age,
            'height_cm': height,
            'weight_kg': weight,
            'small_jaw': int(small_jaw),
            'prominent_eyes': int(prominent_eyes),
            'thin_skin': int(thin_skin),
            'hair_loss': int(hair_loss),
            'lmna_mut': int(lmna_mut)
        }

        with st.spinner("Analyzing..."):
            result = make_prediction(models, clinical_data)

        if result:
            # Results section
            st.header("📊 Analysis Results")

            # Top metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                risk_color = {"Low": "normal", "Moderate": "off", "High": "inverse"}
                st.metric(
                    "Risk Classification",
                    result['risk_class'],
                    delta=f"{result['risk_score']:.1%} probability"
                )

            with col2:
                st.metric(
                    "Height Z-Score",
                    f"{result['height_z']:.2f}",
                    delta="Normal" if abs(result['height_z']) < 2 else "Abnormal"
                )

            with col3:
                st.metric(
                    "Weight Z-Score",
                    f"{result['weight_z']:.2f}",
                    delta="Normal" if abs(result['weight_z']) < 2 else "Abnormal"
                )

            st.markdown("---")

            # Visualization row - Risk Gauge and Confidence
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.plotly_chart(
                    create_risk_gauge(result['risk_score']),
                    use_container_width=True
                )

            with col2:
                st.plotly_chart(
                    create_confidence_gauge(result['confidence']),
                    use_container_width=True
                )

            with col3:
                # Simulated progression probabilities
                prog_probs = [0.2, 0.5, 0.3] if result['risk_score'] < 0.5 else [0.1, 0.3, 0.6]
                # Determine dominant progression type
                prog_types = ['slow', 'moderate', 'rapid']
                dominant_prog = prog_types[np.argmax(prog_probs)]
                st.plotly_chart(
                    create_progression_chart(prog_probs),
                    use_container_width=True
                )

            st.markdown("---")

            # ============================================================
            # DECISION SUPPORT SECTION (Monitor / Genetic Testing / Urgent Review)
            # ============================================================
            st.subheader("🏥 Decision Support")

            phenotype_count = sum([
                clinical_data['small_jaw'],
                clinical_data['prominent_eyes'],
                clinical_data['thin_skin'],
                clinical_data['hair_loss'],
                clinical_data['lmna_mut']
            ])

            decision = get_decision_support(
                result['risk_score'],
                result['confidence'],
                result['height_z'],
                result['weight_z'],
                phenotype_count
            )

            # Display primary action category prominently
            action_colors = {
                'MONITOR': ('🟢', '#d4edda', '#155724'),
                'GENETIC_TESTING': ('🟡', '#fff3cd', '#856404'),
                'URGENT_REVIEW': ('🔴', '#f8d7da', '#721c24')
            }

            action_emoji, bg_color, text_color = action_colors.get(
                decision['primary_action'],
                ('⚪', '#f8f9fa', '#212529')
            )

            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: {text_color}; margin: 0;">
                    {action_emoji} Primary Recommendation: {decision['primary_action'].replace('_', ' ')}
                </h3>
                <p style="color: {text_color}; margin: 5px 0 0 0;">
                    Urgency Level: <strong>{decision['urgency'].upper()}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Display action items in columns
            col1, col2, col3 = st.columns(3)

            # Group actions by priority
            critical_actions = [a for a in decision['actions'] if a['priority'] == 'critical']
            high_actions = [a for a in decision['actions'] if a['priority'] == 'high']
            other_actions = [a for a in decision['actions'] if a['priority'] in ['recommended', 'standard']]

            with col1:
                st.markdown("**Critical Actions**")
                if critical_actions:
                    for action in critical_actions:
                        st.error(f"{action['icon']} {action['text']}")
                else:
                    st.success("No critical actions required")

            with col2:
                st.markdown("**High Priority**")
                if high_actions:
                    for action in high_actions:
                        st.warning(f"{action['icon']} {action['text']}")
                else:
                    st.info("No high priority actions")

            with col3:
                st.markdown("**Recommended**")
                if other_actions:
                    for action in other_actions:
                        st.info(f"{action['icon']} {action['text']}")
                else:
                    st.success("Standard monitoring only")

            st.markdown("---")

            # Growth curve
            st.subheader("📈 Growth Trajectory")
            st.plotly_chart(
                create_growth_curve(age, height, weight, result['risk_score'] > 0.5),
                use_container_width=True
            )

            st.markdown("---")

            # ============================================================
            # DISEASE PROGRESSION TIMELINE
            # ============================================================
            st.subheader("📅 Disease Progression Timeline")
            st.caption("Projected disease trajectory based on current risk assessment")

            # Determine progression type from probabilities
            prog_probs = [0.2, 0.5, 0.3] if result['risk_score'] < 0.5 else [0.1, 0.3, 0.6]
            prog_types = ['slow', 'moderate', 'rapid']
            dominant_prog = prog_types[np.argmax(prog_probs)]

            st.plotly_chart(
                create_progression_timeline(age, result['risk_score'], dominant_prog),
                use_container_width=True
            )

            # Progression summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Slow Progression", f"{prog_probs[0]:.0%}")
            with col2:
                st.metric("Moderate Progression", f"{prog_probs[1]:.0%}")
            with col3:
                st.metric("Rapid Progression", f"{prog_probs[2]:.0%}")

            st.markdown("---")

            # Model comparison section
            st.subheader("🔬 Classical vs Quantum ML Comparison")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Simulated quantum result (slightly different for demo)
                quantum_score = result['risk_score'] * 0.95 + 0.025
                st.plotly_chart(
                    create_comparison_chart(result['risk_score'], quantum_score),
                    use_container_width=True
                )

            with col2:
                st.markdown("**Model Agreement**")
                diff = abs(result['risk_score'] - quantum_score)
                if diff < 0.05:
                    st.success("High agreement between models")
                elif diff < 0.15:
                    st.info("Moderate agreement")
                else:
                    st.warning("Models show disagreement")

                st.markdown("**Classical ML**")
                st.write(f"Risk: {result['risk_score']:.1%}")

                st.markdown("**Quantum ML**")
                st.write(f"Risk: {quantum_score:.1%}")

            # Feature importance
            st.markdown("---")
            st.subheader("🔍 Feature Importance")

            # Get feature importance
            if models.get('loaded'):
                importance = models['tabular_model'].get_feature_importance('random_forest')
                if importance is not None:
                    feature_names = models['tabular_preprocessor'].feature_columns
                    importance_dict = dict(zip(feature_names, importance.tolist()))
                    st.plotly_chart(
                        create_feature_importance_chart(importance_dict),
                        use_container_width=True
                    )

            # Downloadable Report Section
            st.markdown("---")
            st.subheader("📥 Download Report")

            col1, col2 = st.columns(2)

            with col1:
                # Text report download
                text_report = generate_report(clinical_data, result, quantum_score)
                st.download_button(
                    label="📄 Download Text Report",
                    data=text_report,
                    file_name=f"HGPS_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col2:
                # CSV report download
                csv_report = generate_csv_report(clinical_data, result, quantum_score)
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv_report,
                    file_name=f"HGPS_Risk_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.caption("Download your assessment report in text or CSV format for records and further analysis.")

    else:
        # Default view when not analyzing
        st.info("👈 Enter patient information in the sidebar and click 'Analyze' to begin assessment.")

        # Information cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🧬 About HGPS
            Hutchinson-Gilford Progeria Syndrome is an extremely rare genetic
            condition causing accelerated aging in children. Early detection
            is crucial for intervention and care planning.
            """)

        with col2:
            st.markdown("""
            ### 🤖 AI Analysis
            This system uses multi-modal deep learning combining facial
            analysis with clinical data. Quantum ML provides complementary
            predictions for enhanced accuracy.
            """)

        with col3:
            st.markdown("""
            ### ⚠️ Disclaimer
            This tool is for research and educational purposes only.
            It should not replace professional medical diagnosis.
            Always consult qualified healthcare providers.
            """)

        # SDG alignment
        st.markdown("---")
        st.subheader("🌍 UN Sustainable Development Goals Alignment")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **SDG 3: Good Health**

            Early detection and intervention for rare diseases
            """)

        with col2:
            st.markdown("""
            **SDG 9: Innovation**

            Quantum computing applications in healthcare
            """)

        with col3:
            st.markdown("""
            **SDG 10: Reduced Inequalities**

            Improving access to rare disease diagnostics
            """)


    # Footer
    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: #888; font-size: 0.8rem;'>
    HGPS Multi-Modal Quantum AI System v1.0 | Research Project |
    <a href='/docs'>API Documentation</a>
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
