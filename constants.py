"""
Constants for the Risk-Aware Active Learning framework.
Defines clinical risk categories for all 7 HAM10000 diagnostic classes.
"""

# ---------------------------------------------------------------------------
# Risk Mapping — maps each diagnosis code to its clinical risk category
# ---------------------------------------------------------------------------
risk_mapping = {
    'mel': 'high_risk',    # Melanoma
    'bcc': 'high_risk',    # Basal cell carcinoma
    'akiec': 'high_risk',  # Actinic keratoses
    'nv': 'low_risk',      # Nevi (moles)
    'bkl': 'low_risk',     # Benign keratosis
    'df': 'low_risk',      # Dermatofibroma
    'vasc': 'low_risk'     # Vascular lesions
}

# ---------------------------------------------------------------------------
# Convenience sets for quick lookup
# ---------------------------------------------------------------------------
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
HIGH_RISK_CLASSES = {'mel', 'bcc', 'akiec'}
LOW_RISK_CLASSES = {'nv', 'bkl', 'df', 'vasc'}

# Full diagnostic names for display/plotting
DIAGNOSIS_FULL_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}
