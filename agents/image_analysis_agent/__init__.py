"""
ClinicaGraph AI - Multimodal Image Analysis Agent
Coordinates vision classification, neuro-radiology (Brain MRI),
pulmonary radiography (Chest X-Ray), and dermatological imaging (Skin Lesion).
"""

from .image_classifier import ImageClassifier
from .chest_xray_agent.covid_chest_xray_inference import ChestXRayClassification
from .brain_tumor_agent.brain_tumor_inference import BrainTumorAgent
from .skin_lesion_agent.skin_lesion_inference import SkinLesionSegmentation


class ImageAnalysisAgent:
    """Orchestrates medical computer vision diagnostic sub-agents."""
    
    def __init__(self, config):
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)
        self.chest_xray_agent = ChestXRayClassification(model_path=config.medical_cv.chest_xray_model_path)
        self.brain_tumor_agent = BrainTumorAgent(model_path=config.medical_cv.brain_tumor_model_path)
        self.skin_lesion_agent = SkinLesionSegmentation(model_path=config.medical_cv.skin_lesion_model_path)
        self.skin_lesion_segmentation_output_path = config.medical_cv.skin_lesion_segmentation_output_path
        self.brain_tumor_output_path = getattr(config.medical_cv, "brain_tumor_output_path", "./uploads/brain_tumor_output/mri_segmentation.png")
    
    def analyze_image(self, image_path: str) -> dict:
        """Classifies image modality (Brain MRI, Chest X-ray, Skin Lesion, or Non-Medical)."""
        return self.image_classifier.classify_image(image_path)
    
    def classify_chest_xray(self, image_path: str) -> str:
        """Inference for pulmonary chest radiography."""
        return self.chest_xray_agent.predict(image_path)
    
    def analyze_brain_mri(self, image_path: str) -> dict:
        """Segment and analyze intracranial lesions in Brain MRI scans."""
        return self.brain_tumor_agent.analyze_mri(image_path)
    
    def segment_skin_lesion(self, image_path: str) -> bool:
        """Segment dermatological lesion and generate overlay visual."""
        return self.skin_lesion_agent.predict(image_path, self.skin_lesion_segmentation_output_path)
