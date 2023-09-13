# Voice-Cloning-and-Fake-Audio-Detection
### Introduction
A technology company working in the Cyber Security industry wants to build data driven systems that help individuals and organizations to understand whether audio and video media is authentic or fake.

### Data Description:
Two datasets will be used in this project:

[**TIMIT Dataset:**](https://github.com/philipperemy/timit) The TIMIT corpus of read speech is designed to provide speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems. TIMIT contains a total of 6300 sentences, 10 sentences spoken by each of 630 speakers from 8 major dialect regions of the United States.

[**CommonVoice Dataset:**](https://commonvoice.mozilla.org/en/datasets) Common Voice is part of Mozilla's initiative to help teach machines how real people speak. Common Voice is a corpus of speech data read by users on the [Common Voice website](https://commonvoice.mozilla.org/), and based upon text from a number of public domain sources like user submitted blog posts, old books, movies, and other public speech corpora. Its primary purpose is to enable the training and testing of automatic speech recognition (ASR) systems.

### Goals(s):
The goal of this project is to build algorithms that can synthesize spoken audio by converting a speaker’s voice to another speaker’s voice with the end goal to detect if any spoken audio is pristine or fake.

### Success Metrics:
* **Voice cloning (VC):** Word Error Rate (WER)
* **Fake audio detection (FAD):** F-score

### Methodology:
**Step 1:** We will build a **Voice Cloning system (VC)** that clones source speaker's voice to the target speaker's voice. We will utlize the TIMIT dataset for this step as it consists of aligned text-audio data with various speakers.

**Step 2:** We will build a **Fake audio detection system (FAD)** that detects if a given audio is synthetically generated or not. We will utilize the CommonVoice dataset as it consists of thousands of naturally spoken audio which could be used as positive examples and creating negative examples using the voice cloning system as automatic data/label generator.
