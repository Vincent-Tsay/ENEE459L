import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import warnings
from scipy.ndimage import convolve, gaussian_filter
from scipy.signal import convolve2d

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
# each row of box_scores corresponds to [corner, corner, corner, corner, confidence]
# needs to return selected rows
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):

    confidence = box_scores[:,4]
    coordinates = box_scores[:,0:4]
    # coordinates formated as [x1, y1, x2, y2]

    # Sorting the order of indices in ascending order to find best boxes
    indices = np.array(range(0, len(box_scores[:,0])))
    sorted_indices = indices[box_scores[:, 4].argsort()]

    # Getting indices of top candidate_size confidence scores
    limited = list(sorted_indices[-candidate_size:])

    # Indices of picked boxes
    picked = []

    # checking each candidate box, back to front
    while len(limited) > 0:

        if not picked:
            # appends last index in limited, which corresponds to the best candidate
            # guarantees at least one candidate box
            picked.append(limited.pop())

        if len(picked) >= top_k or len(limited) <= 1:
            break
        
        # gets coordinates of next best box
        candidate_coord = coordinates[limited[-1]]

        # keeps iterating over limited until no boxes are removed from
        # limited in comparison to the one currently selected in candidate_coord
        limited_len_changed = True
        while limited_len_changed:

            limited_len_changed = False

            # iterates through coordinates of each box in limited
            # front to back. If iou > threshold, then remove that
            # index from limited
            i = 0
            while i < len(limited)-1:

                comparison_coord = coordinates[limited[i]]

                # need to find the top left corner of intersecting rectangle (x1, y1)
                # need to find the bottom right corner of the intersecting rectangle (x2, y2)

                x1 = max(candidate_coord[0], comparison_coord[0])
                y1 = max(candidate_coord[1], comparison_coord[1])
                x2 = min(candidate_coord[2], comparison_coord[2])
                y2 = min(candidate_coord[3], comparison_coord[3])

                intersection = max(0, x2-x1) * max(0, y2-y1)

                active_area = (candidate_coord[2]-candidate_coord[0])*(candidate_coord[3]-candidate_coord[1])
                remaining_area = (comparison_coord[2]-comparison_coord[0])*(candidate_coord[3]-candidate_coord[1])

                union = active_area + remaining_area - intersection

                iou = intersection/union
                
                if iou > iou_threshold:
                    limited.pop(i) # gets rid of element in limited at index i
                    limited_len_changed = True

                i = i + 1
        
        picked.append(limited.pop())

    return box_scores[picked]
    

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):

    with Image.open(image_path) as input_img:

        if input_img.mode != "RGB":
                print(f"image at {image_path} is not in 3-channel RGB formatting")
                return
        
        original_width, original_height = input_img.size

        # int array of resized image
        img_array = np.array(input_img.resize((320, 240)))
        float_img_array = img_array.astype(np.float32)
        norm_img_array = (float_img_array - 127.0) / 128.0

        # norm_img_array has shape (240, 320, 3) corresponding to (height, width, channels)
        # need (channel, height, width) then (1, channel, height, width) for the explicit batch dimension
        norm_img_array = np.transpose(norm_img_array, axes=(2, 0, 1))

        norm_img_array = np.expand_dims(norm_img_array, axis=0)

        # Running ONNX inference model
        onnx_model_path = "version-RFB-320.onnx"
        session = ort.InferenceSession(onnx_model_path, sess_options = sess_options, providers = available_providers)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # get confidence values and bounding boxes coordinates
        confidence, bounds = session.run(None, {input_name: norm_img_array})
        
        # Removing the explicit batch dimension
        confidence = np.squeeze(confidence, axis = 0)
        bounds = np.squeeze(bounds, axis = 0)

        # Getting only the p values, not the complements
        confidence = confidence[:,0]

        # Masking out detections whose confidence falls below conf_threshold
        mask = confidence >= conf_threshold
        confidence = confidence[mask]
        bounds = bounds[mask]

        # box_scores is formatted to be like [corner, corner, corner, corner, confidence]
        box_scores = np.column_stack((bounds, confidence))

        hard_nms_box_scores = hard_nms(box_scores, 0.25)
        coordinates = hard_nms_box_scores[:, 0:4]
        confidence = hard_nms_box_scores[:,4]

        # rescale coordinates
        coordinates[:, 0] = coordinates[:, 0] * original_width
        coordinates[:, 2] = coordinates[:, 2] * original_width
        coordinates[:, 1] = coordinates[:, 1] * original_height
        coordinates[:, 3] = coordinates[:, 3] * original_height        

        # drawing bounding boxes
        output_img = input_img
        draw = ImageDraw.Draw(output_img)

        for coord in coordinates:

            draw.rectangle(
                coord[0:4].tolist(),
                outline = "red",
                width = 3
            )

        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf", size = 45)
        text_color = (255, 255, 255)

        i = 0
        for conf in confidence:

            draw.text(coordinates[i, 2:4].tolist(), np.array2string(conf), fill = text_color, font = font)
            i = i + 1
        
        output_img.save(output_path)

    return


# Edge detection
def non_maximum_suppression(magnitude, angle):
    
    thinned = np.zeros(magnitude.shape)

    deg = (angle * (180.0/np.pi)) % 180 # converts angles from radians to degrees, and then restricts to [0, 180)

    # steps through pixels
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if deg[i,j] >= 22.5 and deg[i,j] < 67.5: # diagonal direction
                
                if magnitude[i,j] >= magnitude[i+1,j-1] or magnitude[i,j] >= magnitude[i-1,j+1]:
                    thinned[i,j] = magnitude[i,j]

            elif deg[i,j] >= 67.5 and deg[i,j] < 112.5: # vertical direction

                if magnitude[i,j] >= magnitude[i-1,j] or magnitude[i,j] >= magnitude[i+1,j]:
                    thinned[i,j] = magnitude[i,j]
                
            elif deg[i,j] >= 112.5 and deg[i,j] < 157.5: # anti-diagonal direction

                if magnitude[i,j] >= magnitude[i-1,j-1] or magnitude[i,j] >= magnitude[i+1,j+1]:
                    thinned[i,j] = magnitude[i,j]

            else:   # horizontal direction

                if magnitude[i,j] >= magnitude[i,j-1] or magnitude[i,j] >= magnitude[i,j+1]:
                    thinned[i,j] = magnitude[i,j]


    return thinned

def hysteresis_thresholding(img, low_thresh, high_thresh):
    
    STRONG = 255
    WEAK = 50
    SUPPRESSED = 0

    

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):

    with Image.open(image_path) as input_img:

        if input_img.mode != "RGB":
                print(f"image at {image_path} is not in 3-channel RGB formatting")
                return

        # get grayscale version of input image
        input_img = input_img.convert("L")
        img_array = np.asarray(input_img)
        
        # converting uint8 array to fp32 and convolving with Gaussian kernel (smooths)
        img_array = img_array.astype(np.float32)
        img_array = gaussian_filter(img_array, sigma = 1.4)

        x_sobel = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        y_sobel = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        grad_x = convolve2d(img_array, x_sobel, mode = "same", boundary = "symm")
        grad_y = convolve2d(img_array, y_sobel, mode = "same", boundary = "symm")

        mag = np.sqrt(grad_x**2 + grad_y**2)

        # normalizing magnitude matrix in [0, 255]
        mag = (mag / np.max(mag)) * 255.0

        theta = np.atan2(grad_y, grad_x)

        # thinning gradients down to one pixel using normalized magnitute and angle
        mag = non_maximum_suppression(mag, theta)

        # isolating valid topological edges, edge_mask is already uint8
        edge_mask = hysteresis_thresholding(mag, low, high)

        output_img = Image.fromarray(edge_mask)
        
        output_img.save(output_path)

    return




if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")