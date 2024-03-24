from roboflow import Roboflow
import json

rf = Roboflow(api_key="kKBzrSzGKIeyzfHSloBo")  # Roboflow API key
project = rf.workspace().project("chiari-mri-detection")  # Roboflow project name
model = project.version(3).model  # Roboflow version number


# infer on a local image
def str_infer(filePath):  # Run the defect detection model
    return model.predict(filePath, confidence=30).json()  # Returns the detection data from the image


def img_infer(filePath):
    return model.predict(filePath, confidence=30).plot()

def str_parse(inference):
    output = ''
    if 'chiari' in inference:  # Check if the inference contains chiari
        output = 'Chiari'
    if 'syrinx' in inference and 'chiari' in inference:
        output = (output + '\n')
    if 'syrinx' in inference:
        output = (output + 'Syrinx')
    if not 'syrinx' in inference and not 'chiari' in inference:
        output =  'Nothing Detected'
    return output


def dict_parse(inference):
    try:
        return str(round(inference.get('predictions')[0].get('confidence'), 2))
    except:
        return 'N/A'
