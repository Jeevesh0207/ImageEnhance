from flask import Flask, request, jsonify
from flask import send_file
from flask_cors import CORS,cross_origin
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})



# Function to adjust brightness, contrast, and color saturation
def adjust_image_properties(image, alpha, beta, saturation):
    new_image = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)
    hsv_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = cv2.add(hsv_image[:, :, 1], saturation)
    final_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return final_image

# Function to soften the image using a bilateral filter
def soften_image(image):
    softened_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return softened_image

# Function to detect edges and smooth them
def smooth_edges(image):
    edges = cv2.Canny(image, 100, 200)
    dilated_edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    smooth_edges = cv2.bilateralFilter(dilated_edges, d=9, sigmaColor=75, sigmaSpace=75)
    edge_mask = smooth_edges != 0
    softened_image_with_smooth_edges = soften_image(image)
    softened_image_with_smooth_edges[edge_mask] = image[edge_mask]
    return softened_image_with_smooth_edges

@app.route('/enhance-image', methods=['POST'])
@cross_origin()
def enhance_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)

    # Convert the image to a NumPy array
    image = np.asarray(bytearray(in_memory_file.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Enhance the image
    enhanced_image = adjust_image_properties(image, 1.2, 5, 30)
    smoothed_edges_image = smooth_edges(enhanced_image)

    # Save the processed image to an in-memory file
    _, buffer = cv2.imencode('.jpg', smoothed_edges_image)
    io_buf = io.BytesIO(buffer)

    # Send the enhanced image back to the client
    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True,port=5001)
