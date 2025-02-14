# QR Code to Spectral Audio

This project generates a QR code or text image, converts it into a sound spectrogram, and outputs the result as an audio file. It can be used for generating QR codes or encoded text in sound format, with customizable frequency ranges.

## Requirements

To run this project, you'll need the following Python libraries:

- `matplotlib`
- `numpy`
- `qrcode`
- `soundfile`
- `Pillow`
- `scipy`

Install them via `pip`:

```bash
pip install -r requirements.txt
```

## Usage

1. **Generate QR Code or Text**  
   You can generate either a QR code or a text image and convert it into a spectral audio format.

    - Choose option 1 for QR Code.
    - Choose option 2 for Text.

2. **Set Frequency Range**  
   Enter a frequency range for the generated sound. The default range is 100Hz to 4000Hz.

3. **Play Audio**  
   The generated audio file will be saved as `qr_square.wav`.

4. **Visualization**  
   After generating the audio, a spectrogram of the sound will be displayed.

### Example

To run the program, simply execute the Python script:

```bash
python generate_spectral_qr.py
```

### Code Overview

The code includes the following steps:

- **QR Code Generation:** Uses the `qrcode` library to create a QR code matrix.
- **Text Matrix Generation:** Creates a visual matrix from a string of text.
- **Audio Generation:** Converts the matrix into a sound spectrogram using sinusoidal waves.
- **Spectrogram Visualization:** Displays a visual spectrogram of the generated audio using `matplotlib`.

## License

This project is licensed under the MIT License.