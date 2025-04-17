from flask import Flask, request, jsonify
import io
from PIL import Image

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    print("Received a request at /inference")

    # Print out all form fields
    for key in request.form:
        value = request.form.get(key)
        print(f"Form field {key}: {value}")

    # Process the image if provided.
    if 'image' in request.files:
        file = request.files['image']
        try:
            img = Image.open(file.stream)
            print("Image received: size:", img.size, "mode:", img.mode)
        except Exception as e:
            print("Error processing image:", str(e))
    else:
        print("No image received.")

    # Return a dummy response with a dummy action sequence.
    dummy_action_sequence = [0.5 for _ in range(16)]
    response = {"action_wheel_commands": dummy_action_sequence}
    print("Sending response:", response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6000, debug=True)
