from roboflow import Roboflow
import json

rf = Roboflow(api_key="kKBzrSzGKIeyzfHSloBo")  # Roboflow API key
project = rf.workspace().project("chiari-mri-detection")  # Roboflow project name
model = project.version(3).model  # Roboflow version number


# infer on a local image
def str_infer(filePath):  # Run the defect detection model
    return model.predict(filePath, confidence=50).json()  # Returns the detection data from the image


def img_infer(filePath):
    return model.predict(filePath, confidence=50).plot()

def str_parse(inference):
    if 'chiari' in inference:  # Check if the inference contains chiari
        return 'Chiari'
    if 'syrinx' in inference:
        return 'Syrinx'
    if not 'syrinx' in inference or 'chiari' in inference:
        return 'Nothing Detected'


def dict_parse(inference):
    return str(round(inference.get('predictions')[0].get('confidence'), 2))
