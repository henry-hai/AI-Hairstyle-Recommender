# AI Hairstyle Recommender

This application utilizes computer vision to analyze front-facing facial geometry and provide personalized hairstyle recommendations. The logic is grounded in 13 years of professional barbering expertise, mapping geometric facial ratios to industry-standard grooming styles.

## Project Structure
* **/backend**: A FastAPI server utilizing MediaPipe for facial landmark detection and automated face-shape classification.
* **/frontend**: A React-based user interface for secure image uploads and real-time recommendation display.

## Technical Implementation
* **Computer Vision**: Uses MediaPipe Face Mesh to extract 468 3D facial landmarks and calculate Cheek-to-Jaw and Height-to-Cheek ratios.
* **Backend**: Python/FastAPI handles image processing and executes classification logic based on professional barbering standards.
* **Frontend**: Responsive React UI styled with Tailwind CSS for a seamless user experience.

## Setup and Installation

### Prerequisites
* Python 3.8+
* Node.js & npm

### Backend Setup
1. Navigate to the directory: `cd backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `python main.py`

### Frontend Setup
1. Navigate to the directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start the application: `npm start`