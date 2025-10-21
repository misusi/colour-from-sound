import cv2, sys, os, numpy as np
from pydub import AudioSegment as audio

# Constants
MIN_SPECTRUM = 380
MAX_SPECTRUM = 750

def fft_1d(mat):
    return np.abs(np.fft.fftshift(np.fft.fft(mat)))

def fft_2d(mat):
    return np.abs(np.fft.fftshift(np.fft.fft2(mat)))

def ifft_1d(mat):
    return np.abs(np.fft.ifft(np.fft.ifftshift(mat)))

def ifft_2d(mat):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(mat)))

def magnitude(complex_val):
    return (complex_val.real**2 + complex_val.imag**2)**(1/2)

def magnitude_mat(mat):
    mag = np.zeros(mat.shape)
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            mag[row,col] = magnitude(mat[row,col])
    return mag

def print_mat_stats(mat):
    print("Median: {}".format(np.median(mat)))
    print("Mean: {}".format(np.mean(mat)))
    print("Min: {}".format(np.min(mat)))
    print("Max: {}".format(np.max(mat)))

def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' Script to convert wavelength to rgb and frequency to wavelength.
    Adapted to python from R source here:
    https://gist.github.com/friendly/67a7df339aa999e2bcfcfec88311abfc
    '''
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    return  255* (magnitude(R) % 255), 255 * (magnitude(G) % 255), 255 * (magnitude(B) % 255)

def frequency_to_wavelength(frequency):
    '''
    Made for sound waves: c ~= 343 m/s
    '''
    if frequency == 0:
        return 0
    else:
        return 343/frequency

def load_audio_data(song_file):
    # mp3 only for now
    byte_data = audio.from_mp3(song_file).raw_data
    int_data = np.array(list(byte_data), np.uint8)
    return int_data

def song_to_mat(int_data):

    # Get square matrix for image for data
    closest_square = int(np.floor(np.sqrt(len(int_data))))
    square_mat = np.zeros((closest_square, closest_square), np.uint8)

    # Copy over what you can
    for row in range(square_mat.shape[0]):
        for col in range(square_mat.shape[1]):
            square_mat[row,col] = int_data[row*col]

    return square_mat

def get_dft_from_1d_data(data):
    dft = np.fft.fft(data)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift

def get_image_from_2d_dft(dft_data):
    dft_inv_shift = np.fft.ifftshift(dft_data)
    return np.array(np.fft.ifft2(dft_data))

def range_shift(value, old_min, old_max, target_min, target_max):
    '''
    Shifts "value" to from range [old_min, old_max] to [target_min, target_max]
    '''
    shifted_value = target_min + ( (target_max - target_min) / (old_max - old_min)) * (value - old_min)
    return shifted_value

def range_shift_mat(mat, target_min, target_max):
    minval = np.min(mat)
    maxval = np.max(mat)
    norm = np.zeros((mat.shape[0],mat.shape[1]))
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            norm[row,col] = range_shift(mat[row, col], minval, maxval, target_min, target_max)
    return norm

def main():

    # Terminal args
    if len(sys.argv) < 4:
        print("Usage: {} <song file> <output image height> <output image width>".format(sys.argv[0]))
        sys.exit(1)
    song_file = sys.argv[1]
    height = int(sys.argv[2])
    width = int(sys.argv[3])
    output_file_name = "{}.jpg".format(os.path.splitext(song_file)[0][:len(song_file)-4])

    # Load audio data from file
    audio_data = load_audio_data(song_file)

    # Get dft -- 1D
    audio_dft = np.array(get_dft_from_1d_data(audio_data), np.uint8)

    # Get wavelength info from audio frequency -- 2D
    # Account for image size being smaller than data array
    wavelength = np.zeros((height,width), np.uint8)
    step_size = audio_dft.size // (height*width)
    n_step = 0
    for i in range(height):
        for j in range(width):
            step_total = 0
            #step_median = np.median(audio_dft[n_step:n_step+step_size])
            #wavelength[i,j] = frequency_to_wavelength(step_median)
            step_mean = np.mean(audio_dft[n_step:n_step+step_size])
            wavelength[i,j] = frequency_to_wavelength(step_mean)
            n_step += step_size

    # Normalize wavelengths from [min_wl, max_wl] to [380, 750]
    norm_wavelength = range_shift_mat(wavelength, MIN_SPECTRUM, MAX_SPECTRUM)

    # Get rgb values
    rgb = np.zeros((height, width, 3))
    r = np.zeros((height, width))
    g = np.zeros((height, width))
    b = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            r[i,j], g[i,j], b[i,j] = wavelength_to_rgb(norm_wavelength[i,j])
            rgb[i, j] = r[i,j], g[i,j], b[i,j]

    # Write rgb file
    cv2.imwrite('output_image.jpg', rgb)


if __name__ == "__main__":
    main()















