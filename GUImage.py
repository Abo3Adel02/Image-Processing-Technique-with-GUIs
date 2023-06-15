import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
from skimage import data


def calculate_histogram(image):
    hist = [0] * 256
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            i = int(image[x, y] * 255)  # Scale the pixel value to [0, 255]
            hist[i] = hist[i] + 1

    return hist


def histogram_equalization(image):
    # Compute histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 1])

    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Perform histogram equalization
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)
    equalized_image = (equalized_image - equalized_image.min()) / (equalized_image.max() - equalized_image.min())

    return equalized_image


def sobel_filter(image, ksize):
    # Calculate the Sobel kernels based on the derivative orders
    sobel_kernel_x = cv2.getDerivKernels(1, 0, ksize)
    sobel_kernel_x = np.outer(sobel_kernel_x[0], sobel_kernel_x[1])
    sobel_kernel_y = cv2.getDerivKernels(0, 1, ksize)
    sobel_kernel_y = np.outer(sobel_kernel_y[0], sobel_kernel_y[1])

    # Apply the Sobel kernels
    temp1 = np.zeros(shape=image.shape)
    temp2 = np.zeros(shape=image.shape)

    for i in range(int(ksize / 2), image.shape[0] - int(ksize / 2)):
        for j in range(int(ksize / 2), image.shape[1] - int(ksize / 2)):
            sum1 = 0
            sum2 = 0
            for x in range(-1 * int(ksize / 2), int(ksize / 2) + 1):
                for y in range(-1 * int(ksize / 2), int(ksize / 2) + 1):
                    sum1 = sum1 + (int(image[i + x][j + y]) * int(sobel_kernel_x[x + 1][y + 1]))
                    sum2 = sum2 + (int(image[i + x][j + y]) * int(sobel_kernel_y[x + 1][y + 1]))
            if sum1 > 255:
                sum1 = 255
            elif sum1 < 0:
                sum1 = 0

            if sum2 > 255:
                sum2 = 255
            elif sum2 < 0:
                sum2 = 0

            temp1[i][j] = sum1
            temp2[i][j] = sum2

    gradient_magnitude = np.sqrt(temp1 ** 2 + temp2 ** 2)

    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())

    return gradient_magnitude


def laplace_filter(image, ksize):
    # Calculate the Sobel kernels based on the derivative orders
    sobel_kernel_x = cv2.getDerivKernels(2, 0, ksize)
    sobel_kernel_x = np.outer(sobel_kernel_x[0], sobel_kernel_x[1])
    sobel_kernel_y = cv2.getDerivKernels(0, 2, ksize)
    sobel_kernel_y = np.outer(sobel_kernel_y[0], sobel_kernel_y[1])

    # Apply the Sobel kernels
    temp1 = np.zeros(shape=image.shape)
    temp2 = np.zeros(shape=image.shape)

    for i in range(int(ksize / 2), image.shape[0] - int(ksize / 2)):
        for j in range(int(ksize / 2), image.shape[1] - int(ksize / 2)):
            sum1 = 0
            sum2 = 0
            for x in range(-1 * int(ksize / 2), int(ksize / 2) + 1):
                for y in range(-1 * int(ksize / 2), int(ksize / 2) + 1):
                    sum1 = sum1 + (int(image[i + x][j + y]) * int(sobel_kernel_x[x + 1][y + 1]))
                    sum2 = sum2 + (int(image[i + x][j + y]) * int(sobel_kernel_y[x + 1][y + 1]))
            if sum1 > 255:
                sum1 = 255
            elif sum1 < 0:
                sum1 = 0

            if sum2 > 255:
                sum2 = 255
            elif sum2 < 0:
                sum2 = 0

            temp1[i][j] = sum1
            temp2[i][j] = sum2

    gradient_magnitude = np.sqrt(temp1 ** 2 + temp2 ** 2)
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())


    return gradient_magnitude

def apply_fourier_transform(image):
    image_array = np.array(image)
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())

    f = np.fft.fft2(image_array)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    magnitude_spectrum /= np.max(magnitude_spectrum)
    magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())

    return magnitude_spectrum


def add_salt_and_pepper_noise(image, noise_prob):
    height, width = image.shape[:2]
    noise = np.random.rand(height, width)
    noisy_image = np.copy(image)
    noisy_image[noise < noise_prob / 2] = 0
    noisy_image[noise > 1 - noise_prob / 2] = 255
    return noisy_image

def median_filter(image, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final



def add_periodic_noise(image, amplitude, frequency):
    height, width = image.shape[:2]
    x = np.arange(width) / width
    y = np.arange(height) / height
    X, Y = np.meshgrid(x, y)
    noise = amplitude * np.sin(2 * np.pi * (frequency * X + frequency * Y))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_image.astype(np.uint8))


def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    normalized_image = (image - image_min) * (255.0 / (image_max - image_min))
    return normalized_image.astype(np.uint8)


def notch_filter(image, u_k, v_k, d0):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    P, Q = fshift.shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 1.0
            else:
                H[u, v] = 0.0
    notch_reject_center = fshift * H
    notch_reject = np.fft.ifftshift(notch_reject_center)
    inverse_notch_reject = np.fft.ifft2(notch_reject)

    result = np.abs(inverse_notch_reject)
    result = (result - result.min()) / (result.max() - result.min())


    return result


def band_reject(image, d0):
    M, N = image.shape
    H = np.zeros((M, N), dtype=np.float32)

    ## implement the equation for low pass filter
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if D <= d0:
                H[u, v] = 1
            else:
                H[u, v] = 0

    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g = np.abs(np.fft.ifft2(G))

    g = (g - g.min()) / (g.max() - g.min())

    return g




# Remove periodic noise using a mask selected from the magnitude spectrum
def remove_periodic_noise_with_mask(image, magnitude_spectrum,selected_pixels, d0):

    st.info("Select pixels on the Fourier Transform to create a mask")

    m = magnitude_spectrum.shape[0]
    n = magnitude_spectrum.shape[1]
    for u in range(m):
        for v in range(n):
            for d in range(len(selected_pixels)):
                u0 = selected_pixels[d][0]
                v0 = selected_pixels[d][1]
                u0, v0 = v0, u0
                d1 = pow(pow(u - u0, 2) + pow(v - v0, 2), 1)
                d2 = pow(pow(u + u0, 2) + pow(v + v0, 2), 1)
                if d1 <= d0 or d2 <= d0:
                    magnitude_spectrum[u][v] *= 0.0
    f_ishift = np.fft.ifftshift(magnitude_spectrum)
    img_back = np.fft.ifft2(f_ishift)
    clean_image = np.abs(img_back)


    return clean_image.astype(np.uint8)

st.title("Image Processing Application")
st.write("Upload an image and apply various image processing operations")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image2 = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    image = Image.open(uploaded_image)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    image_array = np.array(image.convert("L"), dtype=np.float32) / 255.0  # Convert to grayscale and normalize to [0, 1]
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_column_width=True)
    st.sidebar.title("Image Processing Functions")
    function = st.sidebar.selectbox("Select a function", ["Choose Operation", "Display Histogram",
                                                          "Apply Histogram Equalization",
                                                          "Apply Sobel Filter", "Apply Laplace Filter",
                                                          "Apply Fourier Transform", "Apply Salt and Pepper Noise Operations",
                                                          "Apply Periodic Noise Operations"])

    if function == "Display Histogram":
        st.subheader("Histogram")

        histogram = calculate_histogram(image_array)

        st.subheader("Original Image")
        st.image(image, caption="Original Image", use_column_width=True)

        plt.figure()
        plt.title("Histogram")
        plt.plot(histogram)
        st.subheader("Histogram")
        st.pyplot(plt)
    elif function == "Apply Histogram Equalization":

        equalized_image = histogram_equalization(image_array)
        equalized_image_pil = Image.fromarray(np.uint8(equalized_image * 255))

        equalized_histogram = calculate_histogram(equalized_image)


        st.subheader("Equalized Image")
        st.image(equalized_image_pil, caption="Equalized Image", use_column_width=True)

        plt.figure()
        plt.title("Equalized Histogram")
        plt.plot(equalized_histogram)
        st.subheader("Equalized Histogram")
        st.pyplot(plt)

    elif function == "Apply Sobel Filter":

        kernel_size = st.slider("Select kernel size", 3, 15, 3, step=2)
        filtered_image = sobel_filter(image2,kernel_size)
        st.subheader("Filtered Image")
        st.image(filtered_image, caption="Filtered Image", use_column_width=True)
    elif function == "Apply Laplace Filter":

            kernel_size = st.slider("Select kernel size", 3, 15, 3, step=2)
            filtered_image = laplace_filter(image2, kernel_size)
            st.subheader("Filtered Image")
            st.image(filtered_image, caption="Filtered Image", use_column_width=True)

    elif function == "Apply Fourier Transform":
        magnitude_spectrum = apply_fourier_transform(image)

        st.subheader("Fourier Transform")
        st.image(magnitude_spectrum, caption="Fourier Transform", use_column_width=True)

    elif function == "Apply Salt and Pepper Noise Operations":

        noise_prob = st.slider("Select noise probability", 0.0, 1.0, 0.05)
        noisy_image = add_salt_and_pepper_noise(image2, noise_prob)
        st.subheader("Noisy Image")
        st.image(noisy_image, caption="Noisy Image", use_column_width=True)

        st.slider("Select Median Kernel Size", 3, 15, 3, step=2)

    elif function == "Apply Periodic Noise Operations":
        amplitude = st.slider("Select periodic noise amplitude", 0, 30, 1)
        frequency = st.slider("Select periodic noise frequency", 0, 30, 1)
        noisy_image = add_periodic_noise(image2, amplitude, frequency)
        st.subheader("Noisy Image")
        st.image(noisy_image, caption="Noisy Image", use_column_width=True)

        remove_noise_method = st.selectbox("Select Periodic Noise Removal Method", ("Notch", "Band-reject", "Mask"))

        if remove_noise_method == "Notch":
            u_k = st.slider("Select u_k", 0, 50, 1)
            v_k = st.slider("Select v_k", 0, 50, 1)
            d0 = st.slider("Select d0", 0, 50, 1)
            clean_image = notch_filter(noisy_image, u_k, v_k, d0)
        elif remove_noise_method == "Band-reject":
            d0 = st.slider("Select d0", 0, 50, 1)
            clean_image = band_reject(np.array(noisy_image), d0)
        elif remove_noise_method == "Mask":

            selected_pixels = []
            def on_pixel_click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    selected_pixels.append((x, y))
                    #print(selected_pixels)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    #cv2.putText(apply_fourier_transform(noisy_image),str(x)+','+str(y),(x,y),font,1,(255,0,0),2)
                    #cv2.imshow('image',apply_fourier_transform(noisy_image))

            cv2.imshow("Select Pixels",apply_fourier_transform(noisy_image))
            cv2.setMouseCallback('Select Pixels', on_pixel_click)
            print(1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("selected pixels are:",selected_pixels)
            clean_image = remove_periodic_noise_with_mask(noisy_image, apply_fourier_transform(noisy_image),selected_pixels, 2)

        if clean_image is not None:
            st.subheader("Clean Image")
            st.image(clean_image, caption="Clean Image", use_column_width=True)

