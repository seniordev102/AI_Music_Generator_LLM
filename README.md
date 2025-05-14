# Music Generator Model

A comprehensive AI-powered music generation system that combines multiple models and approaches to generate music with control over melody, emotion, lyrics, and style.

## 🎵 Features

- Multi-model music generation pipeline
- BERT-based music understanding and generation
- Emotion-aware music generation
- Lyrics to melody conversion
- Music style transfer
- Real-time music generation with agent-based system

## 📦 Components

### Core Components

- **BERT Music Model**: Music understanding and generation using BERT architecture
- **GetMusic**: Advanced music generation module
- **Emotion**: Emotion-aware music generation
- **LLM**: Large Language Model integration for music generation
- **MelodyLogic**: Melody generation and refinement
- **SeederAgent**: Agent-based real-time music generation

### Auxiliary Components

- **Adding Generator**: Audio processing and augmentation
- **Lyric Checker**: Lyrics processing and melody matching
- **Recognition Transfer**: Music style transfer
- **Telemelody**: Melody generation and processing

## 🚀 Getting Started

### Prerequisites

1. Python environment with dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install required system packages:
   ```bash
   # For audio processing
   apt-get install flac
   ```

### Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   # Or using pipenv
   pipenv install
   ```
3. Download required model checkpoints (contact maintainers)

### Docker Support

The project includes Docker support for containerized deployment:

```bash
docker build -t iah-music-generator .
docker run -it iah-music-generator
```

## 🎮 Usage

### Basic Music Generation

```bash
python getmusic/generate.py --config configs/default.yaml
```

### Emotion-based Generation

```bash
bash emotion/Piano_gen.sh
```

### Lyrics to Melody

```bash
python lyrichecker/lyrics_to_melody.py --input "your lyrics here"
```

## 🛠 Development

The project is organized into multiple specialized components, each handling different aspects of music generation. Key directories:

- `/bert`: BERT-based music understanding
- `/emotion`: Emotion-aware generation
- `/getmusic`: Core music generation
- `/llm`: Language model integration
- `/seederagent`: Real-time generation agent

## 📝 License

Contact maintainers for licensing information

## 👥 Contributors

Daniel Lee,
Hoang Le

## 📧 Contact

For questions and support, please contact us
