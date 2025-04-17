from flask import Flask, request, jsonify
from gr00t.eval.service import ExternalRobotInferenceClient
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Initialize the GR00T policy client.
policy = ExternalRobotInferenceClient(host="localhost", port=5555)

@app.route('/inference', methods=['POST'])
def inference():
    raw_obs_dict = {}
    
    # Process the image from the uploaded file.
    if 'image' in request.files:
        file = request.files['image']
        try:
            img = Image.open(file.stream).convert('RGB')
            # Convert image to a NumPy array.
            img_array = np.array(img)
            # Set the key that matches the modality configuration.
            raw_obs_dict["video.ego_view"] = img_array.tolist()  # or keep as array if supported
        except Exception as e:
            return jsonify({"error": f"Failed processing image: {str(e)}"}), 500
    else:
        return jsonify({"error": "No image provided"}), 400

    # Process additional state information from form fields.
    try:
        raw_obs_dict["state.left_wheel"] = float(request.form.get("state.left_wheel", "0"))
        raw_obs_dict["state.right_wheel"] = float(request.form.get("state.right_wheel", "0"))
        raw_obs_dict["state.acceleration"] = float(request.form.get("state.acceleration", "0"))
    except Exception as e:
        return jsonify({"error": f"Invalid state parameters: {str(e)}"}), 400

    # Optionally, send previous action information if needed.
    # For example, expect a comma‐separated string of 16 numbers.
    if "action.wheel_commands" in request.form:
        try:
            action_str = request.form.get("action.wheel_commands")
            action_values = [float(x) for x in action_str.split(",")]
            raw_obs_dict["action.wheel_commands"] = action_values
        except Exception as e:
            # In case parsing fails, you can assign a default (e.g., zeros)
            raw_obs_dict["action.wheel_commands"] = [0.0] * 16
    else:
        # You can also provide a default if no previous action is available.
        raw_obs_dict["action.wheel_commands"] = [0.0] * 16

    # Call the inference method: this returns a dictionary.
    try:
        raw_action_chunk = policy.get_action(raw_obs_dict)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # Return the action dictionary (e.g., action.wheel_commands of length 16).
    return jsonify(raw_action_chunk)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555)
