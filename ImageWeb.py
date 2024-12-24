import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.segmentation import clear_border
from skimage import color
from scipy.ndimage import generic_filter
from skimage.util import random_noise
# Define the custom HTML for the radio buttons
custom_html1= """
<div class="radio">
  <input value="1" name="rating" type="radio" id="rating-1" />
  <label title="1 stars" for="rating-1">
    <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 576 512">
      <path
        d="M316.9 18C311.6 7 300.4 0 288.1 0s-23.4 7-28.8 18L195 150.3 51.4 171.5c-12 1.8-22 10.2-25.7 21.7s-.7 24.2 7.9 32.7L137.8 329 113.2 474.7c-2 12 3 24.2 12.9 31.3s23 8 33.8 2.3l128.3-68.5 128.3 68.5c10.8 5.7 23.9 4.9 33.8-2.3s14.9-19.3 12.9-31.3L438.5 329 542.7 225.9c8.6-8.5 11.7-21.2 7.9-32.7s-13.7-19.9-25.7-21.7L381.2 150.3 316.9 18z"
      ></path>
    </svg>
  </label>

  <input value="2" name="rating" type="radio" id="rating-2" />
  <label title="2 stars" for="rating-2">
    <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 576 512">
      <path
        d="M316.9 18C311.6 7 300.4 0 288.1 0s-23.4 7-28.8 18L195 150.3 51.4 171.5c-12 1.8-22 10.2-25.7 21.7s-.7 24.2 7.9 32.7L137.8 329 113.2 474.7c-2 12 3 24.2 12.9 31.3s23 8 33.8 2.3l128.3-68.5 128.3 68.5c10.8 5.7 23.9 4.9 33.8-2.3s14.9-19.3 12.9-31.3L438.5 329 542.7 225.9c8.6-8.5 11.7-21.2 7.9-32.7s-13.7-19.9-25.7-21.7L381.2 150.3 316.9 18z"
      ></path>
    </svg>
  </label>

  <input value="3" name="rating" type="radio" id="rating-3" />
  <label title="3 stars" for="rating-3">
    <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 576 512">
      <path
        d="M316.9 18C311.6 7 300.4 0 288.1 0s-23.4 7-28.8 18L195 150.3 51.4 171.5c-12 1.8-22 10.2-25.7 21.7s-.7 24.2 7.9 32.7L137.8 329 113.2 474.7c-2 12 3 24.2 12.9 31.3s23 8 33.8 2.3l128.3-68.5 128.3 68.5c10.8 5.7 23.9 4.9 33.8-2.3s14.9-19.3 12.9-31.3L438.5 329 542.7 225.9c8.6-8.5 11.7-21.2 7.9-32.7s-13.7-19.9-25.7-21.7L381.2 150.3 316.9 18z"
      ></path>
    </svg>
  </label>

  <input value="4" name="rating" type="radio" id="rating-4" />
  <label title="4 stars" for="rating-4">
    <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 576 512">
      <path
        d="M316.9 18C311.6 7 300.4 0 288.1 0s-23.4 7-28.8 18L195 150.3 51.4 171.5c-12 1.8-22 10.2-25.7 21.7s-.7 24.2 7.9 32.7L137.8 329 113.2 474.7c-2 12 3 24.2 12.9 31.3s23 8 33.8 2.3l128.3-68.5 128.3 68.5c10.8 5.7 23.9 4.9 33.8-2.3s14.9-19.3 12.9-31.3L438.5 329 542.7 225.9c8.6-8.5 11.7-21.2 7.9-32.7s-13.7-19.9-25.7-21.7L381.2 150.3 316.9 18z"
      ></path>
    </svg>
  </label>

  <input value="5" name="rating" type="radio" id="rating-5" />
  <label title="5 star" for="rating-5">
    <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 576 512">
      <path
        d="M316.9 18C311.6 7 300.4 0 288.1 0s-23.4 7-28.8 18L195 150.3 51.4 171.5c-12 1.8-22 10.2-25.7 21.7s-.7 24.2 7.9 32.7L137.8 329 113.2 474.7c-2 12 3 24.2 12.9 31.3s23 8 33.8 2.3l128.3-68.5 128.3 68.5c10.8 5.7 23.9 4.9 33.8-2.3s14.9-19.3 12.9-31.3L438.5 329 542.7 225.9c8.6-8.5 11.7-21.2 7.9-32.7s-13.7-19.9-25.7-21.7L381.2 150.3 316.9 18z"
      ></path>
    </svg>
  </label>
</div>
"""
custom_css1 = """
.radio {
  display: flex;
  justify-content: center;
  gap: 10px;
}

.radio > input {
  position: absolute;
  appearance: none;
  opacity: 0; /* Hide the actual radio button input */
  width: 0;
  height: 0;
}

.radio > label {
  cursor: pointer;
  font-size: 30px;
  position: relative;
  display: inline-block;
  transition: transform 0.3s ease;
}

.radio > label > svg {
  fill: #666;
  transition: fill 0.3s ease;
}

/* Ensuring the star icons stay stable */
.radio > label:hover {
  transform: scale(1); /* Prevent moving on hover */
}

.radio > label:hover > svg {
  fill: #ff9e0b; /* Highlight the star color */
  filter: drop-shadow(0 0 15px rgba(255, 158, 11, 0.9));
  animation: shimmer 1s ease infinite alternate;
}

.radio > input:checked + label > svg {
  fill: #ff9e0b;
  filter: drop-shadow(0 0 15px rgba(255, 158, 11, 0.9));
  animation: pulse 0.8s infinite alternate;
}

/* Prevents the stars from expanding and shifting */
.radio > label::before,
.radio > label::after {
  content: "";
  position: absolute;
  width: 6px;
  height: 6px;
  background-color: #ff9e0b;
  border-radius: 50%;
  opacity: 0;
  transform: scale(0);
  transition:
    transform 0.4s ease,
    opacity 0.4s ease;
  animation: particle-explosion 1s ease-out;
}

.radio > label:hover::before,
.radio > label:hover::after {
  opacity: 1;
  transform: translateX(-50%) scale(1.5);
}

@keyframes pulse {
  0% {
    transform: scale(1);
  }
  100% {
    transform: scale(1.1);
  }
}

@keyframes particle-explosion {
  0% {
    opacity: 0;
    transform: scale(0.5);
  }
  50% {
    opacity: 1;
    transform: scale(1.2);
  }
  100% {
    opacity: 0;
    transform: scale(0.5);
  }
}

@keyframes shimmer {
  0% {
    filter: drop-shadow(0 0 10px rgba(255, 158, 11, 0.5));
  }
  100% {
    filter: drop-shadow(0 0 20px rgba(255, 158, 11, 1));
  }
}

.radio > input:checked + label:hover,
.radio > input:checked + label:hover ~ label {
  fill: #e58e09;
}

.radio > label:hover,
.radio > label:hover ~ label {
  fill: #ff9e0b;
}

.radio input:checked ~ label svg {
  fill: #ff9e0b;
}

"""
custom_css2="""
@keyframes astronaut {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}
.astronaut {
  width: 250px;
  height: 300px;
  position: absolute;
  z-index: 11;
  top: calc(50% - 150px);
  left: calc(50% - 125px);
  animation: astronaut 5s linear infinite;
}

.schoolbag {
  width: 100px;
  height: 150px;
  position: absolute;
  z-index: 1;
  top: calc(50% - 75px);
  left: calc(50% - 50px);
  background-color:rgb(202, 202, 202);
  border-radius: 50px 50px 0 0 / 30px 30px 0 0;
}

.b-head {
  width: 97px;
  height: 80px;
  position: absolute;
  z-index: 3;
  background: -webkit-linear-gradient(left, #ff9e0b 0%, #ff9e0b 50%, #ff9e0b 50%, #ff9e0b 100%);
  border-radius: 50%;
  top: 34px;
  left: calc(50% - 47.5px);
}

.b-head:after {
  content: "";
  width: 60px;
  height: 50px;
  position: absolute;
  top: calc(50% - 25px);
  left: calc(50% - 30px);
  background: -webkit-linear-gradient(top, #15aece 0%, #15aece 50%, #0391bf 50%, #0391bf 100%);
  border-radius: 15px;
}

.b-head:before {
  content: "";
  width: 12px;
  height: 25px;
  position: absolute;
  top: calc(50% - 12.5px);
  left: -4px;
  background-color: #618095;
  border-radius: 5px;
  box-shadow: 92px 0px 0px #618095;
}

.body {
  width: 85px;
  height: 100px;
  position: absolute;
  z-index: 2;
  background-color:rgb(196, 196, 196);
  border-radius: 40px / 20px;
  top: 105px;
  left: calc(50% - 41px);
  background: -webkit-linear-gradient(left, #ff9e0b 0%, #ff9e0b 50%, #ff9e0b 50%, #ff9e0b 100%);
}

.panel {
  width: 60px;
  height: 40px;
  position: absolute;
  top: 20px;
  left: calc(50% - 30px);
  background-color: #b7cceb;
}

.panel:before {
  content: "";
  width: 30px;
  height: 5px;
  position: absolute;
  top: 9px;
  left: 7px;
  background-color: #ff9e0b;
  box-shadow: 0px 9px 0px #ff9e0b, 0px 18px 0px #ff9e0b;
}

.panel:after {
  content: "";
  width: 8px;
  height: 8px;
  position: absolute;
  top: 9px;
  right: 7px;
  background-color: #ff9e0b;
  border-radius: 50%;
  box-shadow: 0px 14px 0px 2px #ff9e0b;
}

.arm {
  width: 80px;
  height: 30px;
  position: absolute;
  top: 121px;
  z-index: 2;
}

.arm-left {
  left: 30px;
  background-color: #ff9e0b;
  border-radius: 0 0 0 39px;
}

.arm-right {
  right: 30px;
  background-color: #ff9e0b;
  border-radius: 0 0 39px 0;
}

.arm-left:before,
.arm-right:before {
  content: "";
  width: 30px;
  height: 70px;
  position: absolute;
  top: -40px;
}

.arm-left:before {
  border-radius: 50px 50px 0px 120px / 50px 50px 0 110px;
  left: 0;
  background-color: #ff9e0b;
}

.arm-right:before {
  border-radius: 50px 50px 120px 0 / 50px 50px 110px 0;
  right: 0;
  background-color: #ff9e0b;
}

.arm-left:after,
.arm-right:after {
  content: "";
  width: 30px;
  height: 10px;
  position: absolute;
  top: -24px;
}

.arm-left:after {
  background-color:rgb(195, 195, 195);
  left: 0;
}

.arm-right:after {
  right: 0;
  background-color:rgb(195, 195, 195);
}

.leg {
  width: 30px;
  height: 40px;
  position: absolute;
  z-index: 2;
  bottom: 70px;
}

.leg-left {
  left: 76px;
  background-color: #ff9e0b;
  transform: rotate(20deg);
}

.leg-right {
  right: 73px;
  background-color: #ff9e0b;
  transform: rotate(-20deg);
}

.leg-left:before,
.leg-right:before {
  content: "";
  width: 50px;
  height: 25px;
  position: absolute;
  bottom: -26px;
}

.leg-left:before {
  left: -20px;
  background-color: #ff9e0b;
  border-radius: 30px 0 0 0;
  border-bottom: 10px solidrgb(197, 197, 197);
}

.leg-right:before {
  right: -20px;
  background-color: #ff9e0b;
  border-radius: 0 30px 0 0;
  border-bottom: 10px solidrgb(197, 197, 197);
}
"""
custom_html2="""
  <div data-js="astro" class="astronaut">
    <div class="b-head"></div>
    <div class="arm arm-left"></div>
    <div class="arm arm-right"></div>
    <div class="body">
      <div class="panel"></div>
    </div>
    <div class="leg leg-left"></div>
    <div class="leg leg-right"></div>
    <div class="schoolbag"></div>
  </div>
"""
custom_css3="""
b-button {
  color: white;
  text-decoration: none;
  font-size: 25px;
  border: none;
  background: none;
  font-weight: 600;
  font-family: 'Poppins', sans-serif;  
}

b-button::before {
  margin-left: auto;
}

b-button::after, b-button::before {
  content: '';
  width: 0%;
  height: 2px;
  background: #ff9e0b;
  display: block;
  transition: 0.5s;
}

b-button:hover::after, b-button:hover::before {
  width: 100%;
}
"""
custom_html3="""
<b-button>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;About US
</b-button>
"""

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '3em';  // Increased font size
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to our image processing project!!👋💥';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.3s';
                letter.innerText = text[i];

                // Set color to orange
                letter.style.color = 'orange';

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 150);  // Faster letter appearance
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""
combined_css=custom_css1+custom_css2+custom_css3
# Image Processing Functions
def power_law_transform(image, gamma,grayscale):
    if grayscale:
        image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    normalized_img = image / 255.0
    transformed_img = np.power(normalized_img, gamma)
    transformed_img = np.uint8(transformed_img * 255)
    return transformed_img

def histogram_equalization(image,grayscale):    
    if grayscale:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return  cv2.equalizeHist(gray_image)
   # Histogram equalization on the Value channel only affects brightness and contrast, leaving the colors (Hue) unchanged
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # accesses the Value channel (brightness) using hsv_image[:, :, 2]
    # equalizeHist(hsv_image[:, :, 2]) enhances the brightness (contrast) of the Value channel
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
    equalized_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return equalized_image
def Gray_Level_Slicing(image, grayscale, min_val, max_val):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Convert min_val and max_val to integers (in case they are passed as strings)
    min_val = int(min_val)
    max_val = int(max_val)
    
    # Create the output image, where only values between min_val and max_val are set to 255
    output_image = np.zeros_like(image)
    output_image[(image >= min_val) & (image <= max_val)] = 255
    
    return output_image

def linear_negative_transformation(image, grayscale):
    # Convert to grayscale if the flag is set
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Ensure the image is in uint8 format (0-255 range)
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    # Get the maximum intensity value from the image
    max_val = 255  # For 8-bit images, the max value is 255
    
    # Apply the linear negative transformation
    transformed_image = max_val - image
    
    return transformed_image
def log_transformation(image,grayscale):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    max_val = np.max(image)
    # The value of ‘c’ is chosen such that we get the maximum output value corresponding to the bit size used
    C = 255 / np.log(1 + max_val)
    # Converts the image data type to float32 to allow for fractional calculations since logarithmic operations 
    transformed_image = C * np.log(1 + image.astype(np.float32))
    transformed_image = np.uint8(transformed_image)
    return transformed_image

def piecewise_linear_transformation(image,grayscale):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Breakpoints: Divide the pixel intensity range into sections, 0-99: Dark regions, 100-199: Mid-tones, 200-255: Bright regions
    #Slopes: Control how much to stretch or compress the intensity in each section, 0.5: Compress dark pixels (make them closer together), 
    #1.5: Stretch mid-tones (increase contrast), 0.8: Slightly compress bright pixels
    #Intercepts: Add or subtract intensity values to shift brightness in each section, 0: No shift for dark regions, -50: Darken mid-tones slightly
    #100: Brighten bright regions
    breakpoints = [0, 100, 200, 256]  
    slopes = [0.5, 1.5, 0.8]         
    intercepts = [0, -50, 100]  
    transformed_image = np.zeros_like(image, dtype=np.float32)
    for i in range(len(breakpoints) - 1):
        mask = (image >= breakpoints[i]) & (image < breakpoints[i + 1])
        # s=slopes[i]⋅r+intercepts[i]
        transformed_image[mask] = slopes[i] * image[mask] + intercepts[i]
    transformed_image = np.clip(transformed_image, 0, 255)
    return np.uint8(transformed_image)

def bit_plane_slicing(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bit_planes = []
    for i in range(8):
        bit_plane = (image >> i) & 1  # Extract bit-plane i
        bit_plane = bit_plane * 255    # Scale it to 0-255
        bit_planes.append(bit_plane)
    
    return bit_planes

def generate_bit_plane_plot(image):
    bit_planes = bit_plane_slicing(image)
    
    # Create the plot using matplotlib
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    for i in range(8):
        plt.subplot(3, 3, i + 2)
        plt.imshow(bit_planes[i], cmap='gray')
        plt.title(f'Bit-Plane {i}')
        plt.axis('off')

    plt.tight_layout()
    
    # Save the plot to a temporary file
    plot_filename = "Image Project/bit_plane_plot.png"
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename
def sharp_thresholding(image, grayscale, threshold):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Convert threshold value to integer
    threshold = int(threshold)
    
    threshold_value_computed, sharp_thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return sharp_thresholded_image

def soft_thresholding(image, grayscale, threshold):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Convert threshold value to float
    threshold = float(threshold)
    
    # Apply soft thresholding
    transformed_image = np.copy(image).astype(np.float32)
    transformed_image[transformed_image > threshold] -= threshold
    transformed_image[transformed_image <= threshold] = 0

    # Ensure the pixel values are in the valid range (0-255) after thresholding
    transformed_image = np.clip(transformed_image, 0, 255)

    # Convert the image back to uint8 format
    transformed_image = transformed_image.astype(np.uint8)

    return transformed_image

def diffenece_theresholds(image,grayscale,threshold_value):
    difference = np.abs(sharp_thresholding(image,grayscale,threshold_value) - soft_thresholding(image,grayscale,threshold_value))
    return difference


def thresholding(image, grayscale, threshold_type, threshold_value):

    if threshold_type == "Sharp":
        return sharp_thresholding(image, grayscale, threshold_value)
    elif threshold_type == "Soft":
        return soft_thresholding(image, grayscale, threshold_value)
    elif threshold_type =="Difference":
        return diffenece_theresholds(image, grayscale, threshold_value)
        
    else:
        raise gr.Error("Please select a Threshold type")
def arithmetic_operations(image1, image2, operation_type):
    # Resize image2 to match image1's dimensions if necessary
    if image2 is not None:
        image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    else:
        image2_resized = None

    # Perform the selected arithmetic operation
    if operation_type == "Addition":
        result_image = cv2.add(image1, image2_resized)
    elif operation_type == "Subtraction":
        result_image = cv2.subtract(image1, image2_resized)
    elif operation_type == "Multiplication":
        result_image = cv2.multiply(image1, image2_resized, scale=1/255)
    elif operation_type == "Division":
        result_image = cv2.divide(image1, image2_resized, scale=255)
    else:
        raise gr.Error("Please select an operation type")

    return result_image

def Generic_filter(image,grayscale):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    filtered_image = generic_filter(image, np.std, size=(3,3))
    if grayscale:
        return filtered_image
    RGB=cv2.cvtColor(filtered_image,cv2.COLOR_GRAY2RGB)
    return filtered_image
def Gaussian_filter(image,Sigma,grayscale):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    smoothed_image = gaussian_filter(image, sigma=Sigma)
    if grayscale:
        return smoothed_image
    RGB=cv2.cvtColor(smoothed_image,cv2.COLOR_GRAY2RGB)
    return RGB

def salt_and_pepper_noise(image,grayscale,noise_type):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    salt_paper = random_noise(image, mode='s&p')
    salt_paper = np.array(255 * salt_paper, dtype='uint8') 
    if noise_type =="Salt & Pepper":
        return salt_paper
    if noise_type =="Denoised Image":
        denoised_image = cv2.medianBlur(salt_paper, 3)
        return denoised_image
    else:
        raise gr.Error("Please select a noise type")
    
def add_nosie(image,grayscale,noise_type,reduction,reduction_type):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if noise_type == "additive":
        gauss = np.random.normal(0, 25, image.shape).astype(np.float32)
        noisy = cv2.add(image.astype(np.float32), gauss)
        if reduction:
            return noise_reduction(image,grayscale,reduction_type)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "multiplicative":
        speckle = np.random.normal(0, 0.2, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) * (1 + speckle)
        if reduction:
            return noise_reduction(image,grayscale,reduction_type)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "impulse":
        noisy = image.copy()
        prob = 0.05  
        random_matrix = np.random.rand(*image.shape)
        noisy[random_matrix < (prob / 2)] = 0   
        noisy[random_matrix > 1 - (prob / 2)] = 255  
        if reduction:
            return noise_reduction(image,grayscale,reduction_type)
        return noisy
    elif noise_type == "quantization":
        levels = 16 
        factor = 256 // levels
        noisy = (image // factor) * factor
        if reduction:
            return noise_reduction(image,grayscale,reduction_type)
        return noisy
    else:
        raise gr.Error("Please select a noise type")
    
def noise_reduction(image,grayscale,reduction_type):
    if reduction_type == "Gaussian":
        return cv2.GaussianBlur(image, (5, 5), sigmaX=1)
    elif reduction_type == "Median":
        return cv2.medianBlur(image, 5)
    elif reduction_type == "Mean":
        kernel = np.ones((5, 5), np.float32) / 25
        return cv2.filter2D(image, -1, kernel)
    else:
        raise gr.Error("Please select a noise reduction type")
    
def progressive_image_transmission(image, grayscale,see_difference, output_path=None):
    if output_path is None:
        output_path = "progressive_image.jpg"
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    is_written = cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1])
    if grayscale:
        progressive_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    else:
        progressive_image = cv2.imread(output_path)
    if see_difference:
        difference = cv2.absdiff(image, progressive_image)
        plt.imshow(difference, cmap="hot")
        plt.axis('off')  
        plt.tight_layout()
        cmap_output_path = "cmap_difference.png"
        plt.savefig(cmap_output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        color_diff = cv2.imread(cmap_output_path)
        return color_diff
    return progressive_image
import cv2

def lossy_compression(image, grayscale, lossy_quality, see_difference2):
    lossy_quality = int(lossy_quality)
    if lossy_quality < 0 or lossy_quality > 100:
        return image
    original_image = image.copy()
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lossy_encode_param = [cv2.IMWRITE_JPEG_QUALITY, lossy_quality]
    result, lossy_compressed_image = cv2.imencode('.jpg', image, lossy_encode_param)
    lossy_compressed_image = cv2.imdecode(lossy_compressed_image, 1)
    if grayscale:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    if original_image.shape != lossy_compressed_image.shape:
        lossy_compressed_image = cv2.resize(lossy_compressed_image, (original_image.shape[1], original_image.shape[0]))
    if len(original_image.shape) == 2: 
        if len(lossy_compressed_image.shape) == 3:
            lossy_compressed_image = cv2.cvtColor(lossy_compressed_image, cv2.COLOR_BGR2GRAY)
    if see_difference2:
        difference = cv2.absdiff(original_image, lossy_compressed_image)
        plt.imshow(difference, cmap="hot")
        plt.axis('off')  
        plt.tight_layout()
        cmap_output_path = "cmap_difference.png"
        plt.savefig(cmap_output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        color_diff = cv2.imread(cmap_output_path)
        return color_diff
    return lossy_compressed_image

def lossless_compression(image, grayscale, level, see_difference3):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result, lossless_compressed_image = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, level])
    lossless_compressed_image = cv2.imdecode(lossless_compressed_image, cv2.IMREAD_UNCHANGED)
    if grayscale and len(lossless_compressed_image.shape) == 3:
        lossless_compressed_image = cv2.cvtColor(lossless_compressed_image, cv2.COLOR_BGR2GRAY)
    if see_difference3:
        if image.shape == lossless_compressed_image.shape:
            difference = cv2.absdiff(image, lossless_compressed_image)
            plt.imshow(difference, cmap="hot")
            plt.axis('off') 
            plt.tight_layout()
            cmap_output_path = "cmap_difference.png"
            plt.savefig(cmap_output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            color_diff = cv2.imread(cmap_output_path)
            return color_diff
    return lossless_compressed_image

import cv2
import numpy as np

def gray_level_discontinuity_point_detection(image, grayscale=True, threshold_factor=2.0):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.array([[-1, -1, -1],
                     [-1,  8, -1],
                     [-1, -1, -1]], dtype=np.float32)
    filtered_image = cv2.filter2D(image, -1, mask)
    threshold_value = np.std(image) * threshold_factor
    isolated_points = np.zeros_like(image)
    isolated_points[np.abs(filtered_image) > threshold_value] = 255

    return isolated_points

import cv2
import numpy as np

def gray_level_discontinuity_line_detection(image, grayscale=True, mask_type="Horizontal", threshold_value=100):
    # Convert to grayscale if necessary
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Define masks for horizontal, vertical, and diagonal line detection
    horizontal_mask = np.array([[-1, -1, -1],
                                [ 2,  2,  2],
                                [-1, -1, -1]], dtype=np.float32)

    vertical_mask = np.array([[-1,  2, -1],
                              [-1,  2, -1],
                              [-1,  2, -1]], dtype=np.float32)

    diagonal_mask = np.array([[ 2, -1, -1],
                              [-1,  2, -1],
                              [-1, -1,  2]], dtype=np.float32)

    # Apply the chosen mask
    if mask_type == "Horizontal":
        filtered_image = cv2.filter2D(image, -1, horizontal_mask)
    elif mask_type == "Vertical":
        filtered_image = cv2.filter2D(image, -1, vertical_mask)
    elif mask_type == "Diagonal":
        filtered_image = cv2.filter2D(image, -1, diagonal_mask)
    else:
        raise gr.Error("Please select a mask type")

    # Apply thresholding to detect lines
    detected_lines = np.zeros_like(image)
    detected_lines[np.abs(filtered_image) > threshold_value] = 255

    return detected_lines

def gray_level_discontinuity_edge_detection(image,dilation):
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(image, low_threshold, high_threshold)
    kernel = np.ones((3, 3), np.uint8)
    if dilation:
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        return dilated_edges
    else:
        return edges

def watershed_segmentation(image):
    cells = image[:, :, 0]  # Extract the first channel

    # Step 2: Otsu Thresholding
    ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Morphological Opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 4: Remove Edge-Touching Grains
    opening_cleared = clear_border(opening)

    # Step 5: Identify Background
    sure_bg = cv2.dilate(opening_cleared, kernel, iterations=1)

    # Step 6: Identify Foreground Area using Distance Transform
    dist_transform = cv2.distanceTransform(opening_cleared, cv2.DIST_L2, 5)
    result, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Step 7: Find Unknown Region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 8: Label Markers for Watershed
    result, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown == 255] = 0

    # Step 9: Apply Watershed Algorithm
    markers = cv2.watershed(image, markers)
    watershed_result = image.copy()
    watershed_result[markers == -1] = [0, 0, 255]  # Mark boundaries

    # Step 10: Overlay on Original Image and Create Colored Grains
    image[markers == -1] = [255, 255, 255]
    img2 = color.label2rgb(markers, bg_label=0)
    # Enhancement functions based on the dropdown option
    plt.figure(figsize=(15, 15))

    # Subplot 1: Original Cells
    plt.subplot(4, 3, 1)
    plt.imshow(cells, cmap='gray')
    plt.title("Original Cells")
    plt.axis('off')

    # Subplot 2: Otsu Threshold
    plt.subplot(4, 3, 2)
    plt.imshow(thresh, cmap='gray')
    plt.title("Otsu Threshold")
    plt.axis('off')

    # Subplot 3: Morphological Opening
    plt.subplot(4, 3, 3)
    plt.imshow(opening, cmap='gray')
    plt.title("Morphological Opening")
    plt.axis('off')

    # Subplot 4: Cleared Border
    plt.subplot(4, 3, 4)
    plt.imshow(opening_cleared, cmap='gray')
    plt.title("Cleared Border")
    plt.axis('off')

    # Subplot 5: Sure Background
    plt.subplot(4, 3, 5)
    plt.imshow(sure_bg, cmap='gray')
    plt.title("Sure Background")
    plt.axis('off')

    # Subplot 6: Distance Transform
    plt.subplot(4, 3, 6)
    plt.imshow(dist_transform, cmap='gray')
    plt.title("Distance Transform")
    plt.axis('off')

    # Subplot 7: Sure Foreground
    plt.subplot(4, 3, 7)
    plt.imshow(sure_fg, cmap='gray')
    plt.title("Sure Foreground")
    plt.axis('off')

    # Subplot 8: Unknown Region
    plt.subplot(4, 3, 8)
    plt.imshow(unknown, cmap='gray')
    plt.title("Unknown Region")
    plt.axis('off')

    # Subplot 9: Markers Labeled
    plt.subplot(4, 3, 9)
    plt.imshow(markers, cmap='gray')
    plt.title("Markers Labeled")
    plt.axis('off')

    # Subplot 10: Watershed Boundaries
    plt.subplot(4, 3, 10)
    plt.imshow(watershed_result)
    plt.title("Watershed Boundaries")
    plt.axis('off')

    # Subplot 11: Overlay on Original
    plt.subplot(4, 3, 11)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Overlay on Original")
    plt.axis('off')

    # Subplot 12: Colored Grains
    plt.subplot(4, 3, 12)
    plt.imshow(img2)
    plt.title("Colored Grains")
    plt.axis('off')

    plt.tight_layout()
    plot_filename = "Image Project/watershed_plot.png"
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename

def thresholding_segmentation(image,display):
    image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ret, otsu_thresholded_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresholded_hist = cv2.calcHist([otsu_thresholded_img], [0], None, [256], [0, 256])
    if display == "Image":
        return otsu_thresholded_img
    if display == "Original Image Histogram":
        plt.plot(hist, color='blue')
        plt.title("Histogram of Original Image")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plot_filename = "Image Project/original_hist_plot.png"
        plt.savefig(plot_filename)
        plt.close()
        return plot_filename
    if display == "Thresholded Histogram":
        plt.plot(otsu_thresholded_hist, color='green')
        plt.title("Histogram of Thresholded Image")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plot_filename = "Image Project/thresh_hist_plot.png"
        plt.savefig(plot_filename)
        plt.close()
        return plot_filename
    else:
        raise gr.Error("Please select a Threshold type")

def multilevel_thresholding(image, hist=False):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    T1 = 80
    T2 = 160
    thresholded_img = np.zeros_like(image_gray, dtype=np.uint8)
    thresholded_img[image_gray <= T1] = 0  
    thresholded_img[(image_gray > T1) & (image_gray <= T2)] = 127  
    thresholded_img[image_gray > T2] = 255  
    if hist:
        plt.figure()
        plt.hist(image_gray.ravel(), bins=256, range=(0, 256), color='blue')
        plt.title("Histogram of Grayscale Image")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plot_filename = "hist_plot.png"
        plt.savefig(plot_filename)
        plt.close()
        return plot_filename  
    else:
        return thresholded_img  
def colored_thresholding(image, color, hist):
    threshold_value = 100
    r, g, b = cv2.split(image)
    
    if color == "Red":
        if hist:
            r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
            plt.plot(r_hist, color='red')
            plt.title('Histogram of Red Channel')
            plt.xlim([0, 256])
            plt.tight_layout()
            plot_filename = "hist_Red.png"
            plt.savefig(plot_filename)
            plt.close()
            return plot_filename
        r_thresh = cv2.inRange(r, threshold_value, 255)
        red_img = cv2.merge([r_thresh, g*0, b*0]) 
        return red_img
    
    elif color == "Green":
        if hist:
            g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
            plt.plot(g_hist, color='green')
            plt.title('Histogram of Green Channel')
            plt.xlim([0, 256])
            plt.tight_layout()
            plot_filename = "hist_Green.png"
            plt.savefig(plot_filename)
            plt.close()
            return plot_filename
        g_thresh = cv2.inRange(g, threshold_value, 255)
        green_img = cv2.merge([r*0, g_thresh, b*0]) 
        return green_img
    elif color == "Blue":
        if hist:
            b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
            plt.plot(b_hist, color='blue')
            plt.title('Histogram of Blue Channel')
            plt.xlim([0, 256])
            plt.tight_layout()
            plot_filename = "hist_Blue.png"
            plt.savefig(plot_filename)
            plt.close()
            return plot_filename
        b_thresh = cv2.inRange(b, threshold_value, 255)
        blue_img = cv2.merge([r*0, g*0, b_thresh])  
        return blue_img
    elif color == "Combined":
        if hist:
            r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
            g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
            b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
            plt.plot(r_hist, color='red', label='Red')
            plt.plot(g_hist, color='green', label='Green')
            plt.plot(b_hist, color='blue', label='Blue')
            plt.title('Combined RGB Histogram')
            plt.xlim([0, 256])
            plt.legend()
            plt.tight_layout()
            plot_filename = "hist_Combined.png"
            plt.savefig(plot_filename)
            plt.close()
            return plot_filename
        r_thresh = cv2.inRange(r, threshold_value, 255)
        g_thresh = cv2.inRange(g, threshold_value, 255)
        b_thresh = cv2.inRange(b, threshold_value, 255)
        combined_thresh = cv2.merge([b_thresh, g_thresh, r_thresh])
        return combined_thresh
    else:
        raise gr.Error("Please select a color")

def global_thresholding(image,grayscale):
    if grayscale:
        image=image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    T = np.mean(image)
    T0 = 1  
    iteration = 0
    prev_T = T
    while True:
        G1 = image[image > T]  
        G2 = image[image <= T]  
        mean1 = np.mean(G1) if G1.size > 0 else 0
        mean2 = np.mean(G2) if G2.size > 0 else 0
        T = (mean1 + mean2) / 2
        if abs(prev_T - T) < T0:
            break
        prev_T = T
        iteration += 1

    segmented_image = np.zeros_like(image)
    segmented_image[image > T] = 255  
    segmented_image[image <= T] = 0
    return segmented_image

def hough_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 100) 
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
            x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return image
    else:
        return image

def enhance_image(image1, image2, enhancement_type, min_val, max_val, grayscale, gamma=1, threshold_type="Soft", threshold_value=100,operation_type=None
,sigma=1,noise_type=None,noise_type2=None,reduction=False,reduction_type=None):
    if image1 is None or image1.size == 0:
        raise gr.Error("Please select an image🌅")
    if min_val:
        min_val = int(min_val)
    if max_val:
        max_val = int(max_val)
    if threshold_value:
        threshold_value = int(threshold_value)

    if enhancement_type == "Power Law Transform":
        return power_law_transform(image1, gamma, grayscale)
    elif enhancement_type == "Histogram Equalization":
        return histogram_equalization(image1, grayscale)
    elif enhancement_type == "Gray Level Slicing":
        return Gray_Level_Slicing(image1, grayscale, min_val, max_val)
    elif enhancement_type == "Linear Negative Transformation":
        return linear_negative_transformation(image1, grayscale)
    elif enhancement_type == "Log Transformation":
        return log_transformation(image1, grayscale)
    elif enhancement_type == "Piecewise Linear Transformation":
        return piecewise_linear_transformation(image1, grayscale)
    elif enhancement_type == "Bit Plane Slicing":
        return generate_bit_plane_plot(image1)
    elif enhancement_type == "Thresholding":
        return thresholding(image1, grayscale, threshold_type, threshold_value)
    elif enhancement_type == "Arithmetic Operations":
        if image2 is None or image2.size == 0:
            raise gr.Error("Please select another image🌅")
        return arithmetic_operations(image1, image2, operation_type)  
    elif enhancement_type =="Generic Filter":
        return Generic_filter(image1,grayscale)
    elif enhancement_type =="Gaussian Filter":
        return Gaussian_filter(image1,grayscale,sigma)
    elif enhancement_type =="Salt & Pepper":
        return salt_and_pepper_noise(image1,grayscale,noise_type)
    elif enhancement_type =="Noise Reduction":
        return add_nosie(image1,grayscale,noise_type2,reduction,reduction_type)
    else:
        raise gr.Error("🚫Please select an Enhancement type🚫")


def compress_image(image, compression_type, grayscale, output_path,see_difference=False,lossy_quality=None,see_difference2=False,level=None
    ,see_difference3=False):
    if compression_type == "Progressive Image Transmission":
        return progressive_image_transmission(image, grayscale, output_path,see_difference)
    elif compression_type =="Lossy Compression":
        return lossy_compression(image,grayscale,lossy_quality,see_difference2)
    elif compression_type =="Lossless Compression":
        return lossless_compression(image,grayscale,level,see_difference3)
    else:
        raise gr.Error("🚫Please select a Compression type🚫")

def segment_image(image,segmentation_type,grayscale,mask_type,dilation,display,hist,color,hist2):
    if segmentation_type == "Gray Level Discontinuity Point Detection":
        return gray_level_discontinuity_point_detection(image,grayscale)
    elif segmentation_type == "Gray Level Discontinuity Line Detection":
        return gray_level_discontinuity_line_detection(image,grayscale,mask_type)
    elif segmentation_type == "Gray Level Discontinuity Edge Detection":
        return gray_level_discontinuity_edge_detection(image,dilation)
    elif segmentation_type =="Watershed Segmentation":
        return watershed_segmentation(image)
    elif segmentation_type == "Thresholding Segmentation":
        return thresholding_segmentation(image,display)
    elif segmentation_type == "Multilevel Tresholding":
        return multilevel_thresholding(image,hist)
    elif segmentation_type == "Color Thresholding":
        return colored_thresholding(image,color,hist2)
    elif segmentation_type == "Global Thresholding":
        return global_thresholding(image,grayscale)
    elif segmentation_type == "Hough Transform":
        return hough_transform(image)
    else:
        raise gr.Error("🚫Please select a Segmentation type🚫")


with gr.Blocks(js=js,css=combined_css, theme=gr.themes.Citrus()) as demo:
    with gr.Row(equal_height=True):
        with gr.Column():
            img_input1 = gr.Image(type="numpy", label="Upload Image")
        with gr.Column():
            img_input2 = gr.Image(type="numpy", label="Upload Image 2", visible=False)
        with gr.Column():
            processed_output_display = gr.Image(label="Output Image")

    with gr.Row():
        with gr.Column():
            enhancement_options = gr.Dropdown(
                ["Power Law Transform", "Histogram Equalization","Gray Level Slicing","Linear Negative Transformation"
                 ,"Log Transformation","Piecewise Linear Transformation","Bit Plane Slicing","Thresholding",
                 "Arithmetic Operations","Generic Filter","Gaussian Filter","Salt & Pepper","Noise Reduction",
                 ],
                label="Enhancement Type",
                value=None,
            )
            min_val=gr.Textbox(label="Enter the minimum gray level to enhance",visible=False,placeholder="0 to 255")
            max_val=gr.Textbox(label="Enter the maximum gray level to enhance",visible=False,placeholder="0 to 255")
            operation_type = gr.Radio(["Addition", "Subtraction", "Multiplication", "Division"],label="Choose an operation",visible=False)
            gamma_slider = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=1.0, label="Gamma", visible=False)
            threshold_value=gr.Textbox(label="Enter the threshold value",visible=False,placeholder="0 to 255")
            threshold_type_radio = gr.Radio(["Sharp", "Soft","Difference"], label="Thresholding Type", visible=False)
            sigma_slider = gr.Slider(minimum=0.1, maximum=50.0, step=0.1, value=1.0, label="sigma", visible=False)
            noise_type=gr.Radio(["Salt & Pepper","Denoised Image"],label="Select noise",visible=False)
            noise_type2=gr.Radio(["additive","multiplicative","impulse","quantization"],label="Select noise",visible=False)
            reduction_checkbox=gr.Checkbox(label="Apply reduction",value=False,visible=False)
            reduction_type=gr.Radio(["Mean","Median","Gaussian"],label="Select reduction",visible=False)
            grayscale_checkbox1=gr.Checkbox(label="Apply in GrayScale",value=False)
            apply_enhancement_button = gr.Button("Apply Enhancement🎀")
            
        with gr.Column():
            compression_options = gr.Dropdown(
                ["Progressive Image Transmission","Lossy Compression","Lossless Compression"],
                label="Compression Type",
                value=None,
            )
            see_difference=gr.Checkbox(label="See the difference",visible=False,value=False)
            lossy_quality=gr.Textbox(label="Enter JPEG quality for compression",visible=False,placeholder="0 to 100")
            see_difference2=gr.Checkbox(label="See the difference",visible=False,value=False)
            level=gr.Slider(minimum=0,maximum=9,step=1,value=1,label="PNG compression level",visible=False)
            see_difference3=gr.Checkbox(label="See the difference",visible=False,value=False)
            grayscale_checkbox2=gr.Checkbox(label="Apply in GrayScale",value=False)
            output_path = gr.Textbox(label="Output Path", value="progressive_image.jpg", visible=False)
            apply_compression_button = gr.Button("Apply Compression🧨")
        with gr.Column():
            segmentation__options = gr.Dropdown(
                ["Gray Level Discontinuity Point Detection","Gray Level Discontinuity Line Detection","Gray Level Discontinuity Edge Detection"
                 ,"Watershed Segmentation","Thresholding Segmentation","Multilevel Tresholding","Color Thresholding","Global Thresholding",
                 "Hough Transform"],
                label="Segmentation Type",
                value=None,
            )
            mask_type=gr.Radio(["Horizontal","Vertical","Diagonal"],label="Mask type",visible=False)
            display=gr.Radio(["Image","Original Image Histogram","Thresholded Histogram"],label="Select to Display",visible=False)
            hist=gr.Checkbox(label="Show Histogram",visible=False,value=False)
            dilation=gr.Checkbox(label="Apply for Dilation",visible=False,value=False)
            colors=gr.Radio(["Red","Green","Blue","Combined"],label="Selet a color",visible=False)
            hist2=gr.Checkbox(label="Show Histogram",visible=False,value=False)
            grayscale_checkbox3=gr.Checkbox(label="Apply in GrayScale",value=False)
            apply_segmentation_button = gr.Button("Apply Segmentation🔲🔳")
    with gr.Row(equal_height=True):
        with gr.Column():
            gr.HTML(custom_html3)
            gr.Markdown("""
        <div style="text-align: center; color: #ff9e0b; padding: 20px; border: 2px solid #ff9e0b; border-radius: 10px;">
            <p>We are a dedicated team working on Image Segmentation, Compression, and Enhancement to deliver efficient and effective image processing tools.</p>
            <p style="line-height: 0.5;"><b>Project Members:</b></p>
            <p style="line-height: 0.5;">Youssef Mohamed - 22100512</p>
            <p style="line-height: 0.5;">Nour Elbaraway - 22100165</p>
            <p style="line-height: 0.5;">Mariam Mahmoud - 22100808</p>
            <p style="line-height: 0.5;">Ahmed Yasser - 22100397</p>
        </div>

        """)
       
            gr.HTML(custom_html1)
        with gr.Column():
            gr.HTML(custom_html2)
    

    
    compression_options.change(
    fn=lambda compression_type: [
        gr.update(visible=True) if compression_type == "Progressive Image Transmission" else gr.update(visible=False),
        gr.update(visible=False) if compression_type == "Progressive Image Transmission" else gr.update(visible=False),
        gr.update(visible=True) if compression_type == "Lossy Compression" else gr.update(visible=False),
        gr.update(visible=True) if compression_type == "Lossy Compression" else gr.update(visible=False),
        gr.update(visible=True) if compression_type == "Lossless Compression" else gr.update(visible=False),
        gr.update(visible=True) if compression_type == "Lossless Compression" else gr.update(visible=False),
    ],
    inputs=[compression_options],
    outputs=[see_difference, output_path,lossy_quality,see_difference2,level,see_difference3]
)
    segmentation__options.change(
        fn=lambda segmentation_type: [
            gr.update(visible=True) if segmentation_type == "Gray Level Discontinuity Line Detection" else gr.update(visible=False),
            gr.update(visible=True) if segmentation_type == "Gray Level Discontinuity Edge Detection" else gr.update(visible=False),
            gr.update(visible=False) if segmentation_type == "Gray Level Discontinuity Edge Detection" else gr.update(visible=True),
            gr.update(visible=False) if segmentation_type == "Watershed Segmentation" else gr.update(visible=True),
            gr.update(visible=True) if segmentation_type == "Thresholding Segmentation" else gr.update(visible=False),
            gr.update(visible=False) if segmentation_type == "Thresholding Segmentation" else gr.update(visible=True),
            gr.update(visible=True) if segmentation_type == "Multilevel Tresholding" else gr.update(visible=False),
            gr.update(visible=False) if segmentation_type == "Multilevel Tresholding" else gr.update(visible=True),
            gr.update(visible=True) if segmentation_type == "Color Thresholding" else gr.update(visible=False),
            gr.update(visible=True) if segmentation_type == "Color Thresholding" else gr.update(visible=False),
            gr.update(visible=False) if segmentation_type == "Color Thresholding" else gr.update(visible=True),
        ],
        inputs=[segmentation__options],
        outputs=[mask_type,dilation,grayscale_checkbox3,grayscale_checkbox3,display,grayscale_checkbox3,hist,grayscale_checkbox3,colors,hist2,grayscale_checkbox3 ]
    )
    
    enhancement_options.change(
        fn=lambda enhancement_type: [
            gr.update(visible=True) if enhancement_type == "Power Law Transform" else gr.update(visible=False),  # Show gamma slider for Power Law Transform
            gr.update(visible=True) if enhancement_type == "Gray Level Slicing" else gr.update(visible=False),  # Show min input for Gray Level Slicing
            gr.update(visible=True) if enhancement_type == "Gray Level Slicing" else gr.update(visible=False),  # Show max input for Gray Level Slicing
            gr.update(visible=True) if enhancement_type == "Thresholding" else gr.update(visible=False),  # Show threshold type for Thresholding
            gr.update(visible=True) if enhancement_type == "Thresholding" else gr.update(visible=False),  # Show threshold value input for Thresholding
            gr.update(visible=True) if enhancement_type == "Arithmetic Operations" else gr.update(visible=False),  # Show image2 for Arithmetic Operations
            gr.update(visible=True) if enhancement_type == "Arithmetic Operations" else gr.update(visible=False),  # Show operation type for Arithmetic Operations
            gr.update(visible=True) if enhancement_type == "Gaussian Filter" else gr.update(visible=False),  # Show sigma for Gaussian Filter
            gr.update(visible=True) if enhancement_type == "Salt & Pepper" else gr.update(visible=False),  # Show noise options for Salt & Pepper
            gr.update(visible=True) if enhancement_type == "Noise Reduction" else gr.update(visible=False),  # Show noise options for Noise Reduction
            gr.update(visible=True) if enhancement_type == "Noise Reduction" else gr.update(visible=False),  # Show noise type2 for Noise Reduction
            gr.update(visible=True) if enhancement_type == "Noise Reduction" else gr.update(visible=False),  # Show reduction checkbox for Noise Reduction
            gr.update(visible=True) if enhancement_type == "Noise Reduction" else gr.update(visible=False),  # Show reduction type options for Noise Reduction
        ],
        inputs=[enhancement_options],
        outputs=[gamma_slider, min_val, max_val, threshold_type_radio, threshold_value, img_input2, operation_type, sigma_slider, noise_type, noise_type2, reduction_checkbox, reduction_type]
    )
    reduction_checkbox.change(
        fn=lambda reduction: gr.update(visible=reduction),
        inputs=reduction_checkbox,
        outputs=reduction_type
    )
    apply_compression_button.click(
        fn=compress_image,
        inputs=[img_input1, compression_options, grayscale_checkbox2,see_difference, output_path,lossy_quality,see_difference2,level,see_difference3],
        outputs=processed_output_display,
    )
    apply_segmentation_button.click(
        fn=segment_image,
        inputs=[img_input1, segmentation__options,grayscale_checkbox3,mask_type,dilation,display,hist,colors,hist2],
        outputs=processed_output_display,
    )

    apply_enhancement_button.click(
    fn=enhance_image, 
    inputs=[img_input1, img_input2, enhancement_options, min_val, max_val, grayscale_checkbox1, gamma_slider, threshold_type_radio, threshold_value, operation_type,sigma_slider,noise_type
    ,noise_type2,reduction_checkbox,reduction_type],  
    outputs=processed_output_display
    )

if __name__ == "__main__":
    demo.launch(share=True)