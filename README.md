# Voice-Enabled Text Reader for Visually Impaired 🗣️📖

This project presents a cost-effective, offline, and highly flexible voice-enabled text reader designed to assist visually impaired individuals in independently reading printed or handwritten text. The system leverages **Optical Character Recognition (OCR)** and **Text-to-Speech (TTS)** technologies on a low-power, embedded platform, the **Raspberry Pi 5**. This solution aims to overcome the limitations of expensive, cloud-dependent alternatives by providing a reliable, real-time, and accessible tool.

## ✨ Key Features

  * **Fully Offline Operation:** No internet connection is required, ensuring accessibility in all environments.
  * **Multilingual Support:** High-accuracy OCR for **English**, **Kannada**, and **Telugu** scripts.
  * **Real-time Processing:** Optimized to provide instant audio feedback from a live camera feed.
  * **Enhanced Image Preprocessing:** Includes techniques like CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gaussian blur to improve text recognition in various lighting conditions and with degraded images.
  * **Text Refinement:** Incorporates spell correction and text normalization for improved readability and accuracy.
  * **Rule-Based Transliteration:** A custom-built transliteration model for Indic scripts ensures correct pronunciation for a more natural-sounding voice output.
  * **Lightweight and Portable:** Built on the Raspberry Pi 5 platform, making the device compact and easy to carry.

## ⚙️ How It Works

The system follows a structured, end-to-end process:

1.  **Image Capture:** A USB webcam continuously captures a real-time video feed.
2.  **Image Preprocessing:** Each frame is converted to grayscale, and advanced techniques like CLAHE and Gaussian blur are applied to enhance contrast and reduce noise.
3.  **Text Extraction:** The preprocessed image is fed into the **EasyOCR** engine, which uses a two-stage deep learning architecture (**CRAFT** for text detection and **CRNN** for recognition) to accurately extract the text.
4.  **Language Detection:** The system automatically identifies the language of the extracted text (English, Telugu, or Kannada) using Unicode character ranges.
5.  **Transliteration & Refinement:** For Indic scripts, a rule-based transliteration module converts the text into a phonetic English representation. Text is also refined using spell-checking and normalization.
6.  **Speech Synthesis:** The processed text is sent to the **pyttsx3** TTS engine, which generates natural-sounding speech output without any internet connection.
7.  **Looping Mechanism:** After the audio is played, the system seamlessly returns to the capture stage to process new text, enabling continuous reading.

## 💻 System Requirements

### Hardware

  * **Raspberry Pi 5 Model B** (8GB RAM recommended)
  * **USB Webcam**
  * **Speakers or Headphones**
  * **Power Supply** (USB-C)
  * **MicroSD Card** (with Raspberry Pi OS installed)

### Software

  * **Raspberry Pi OS**
  * **Python 3.x**
  * **OpenCV:** For image processing.
  * **EasyOCR:** The core OCR library.
  * **pyttsx3:** The offline TTS engine.
  * **NumPy** and **PyTorch:** Dependencies for EasyOCR.
  * **indic-transliteration** and **IndicNLP:** For language-specific processing.
  * **SymSpell:** For spell correction.



## 🤝 Acknowledgements

This project was a collaborative effort. We extend our sincere gratitude to our supervisor and the Head of the Department for their invaluable guidance and support. We also thank the creators of the open-source libraries used in this project, which made this work possible.

