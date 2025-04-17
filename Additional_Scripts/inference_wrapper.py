from flask import Flask, request, jsonify
from gr00t.eval.service import ExternalRobotInferenceClient
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

# Create your policy client. Adjust host/port as needed.
policy = ExternalRobotInferenceClient(host="localhost", port=5555)

@app.route('/inference', methods=['POST'])
def inference():
    # This example assumes that Unity sends the image as binary (or base64-encoded).
    # Here, we retrieve the image file that Unity posted.
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    try:
        # Open the image using PIL (this gives you a NumPy array as needed by your model)
        img = Image.open(file.stream).convert('RGB')
        # Convert to array if required by your model (or pass as-is if your model accepts a PIL image)
        img_array = np.array(img)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Construct a raw observation dictionary. Adjust keys as your deployment expects.
    raw_obs_dict = {
        "image": img_array.tolist()  # optionally convert to list, if needed
    }
    
    # Call the inference (this might block until the policy returns an action)
    raw_action_chunk = policy.get_action(raw_obs_dict)
    
    # Return the action as JSON
    return jsonify(raw_action_chunk)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555)
