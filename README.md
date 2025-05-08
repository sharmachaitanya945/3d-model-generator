# 3D Model Generator

A Python tool that converts 2D images into 3D models (.obj or .stl format) and can generate 3D models from text descriptions using AI. This tool analyzes image brightness patterns to create height maps or uses AI models to generate 3D content from textual descriptions.

## Features

- Convert any 2D image to a 3D model
- Generate 3D models from text descriptions using AI
- Support for both OBJ and STL output formats
- Adjustable resolution and depth settings
- Modern GUI interface with easy-to-use workflow
- Command-line interface for easy integration into workflows

## Demo Video

Check out the demo video showing the 3D Model Generator in action:

https://github.com/sharmachaitanya945/3d_model_generator/raw/main/demoVideo/2025-04-29%2006-49-22.mp4

You can also [download the demo video](./demoVideo/2025-04-29%2006-49-22.mp4) directly.

## Screenshots

Here are some screenshots of the application in use:

![Screenshot 1](./Screenshots/Screenshot%202025-05-08%20194714.png)
*Main application interface*

![Screenshot 2](./Screenshots/Screenshot%202025-05-08%20194754.png)
*Image to 3D model conversion*

![Screenshot 3](./Screenshots/Screenshot%202025-05-08%20194831.png)
*Text to 3D model generation*

![Screenshot 4](./Screenshots/Screenshot%202025-05-08%20195218.png)
*Generated 3D model output*

## Installation

1. Clone this repository:
```bash
git clone https://github.com/sharmachaitanya945/3d_model_generator.git
cd 3d_model_generator
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Steps to Run

### GUI Method
1. Activate your virtual environment if you're using one
2. Run the application:
```bash
python main.py
```
3. The GUI will open with two options:
   - **Image to 3D**: Convert an image to a 3D model
   - **Text to 3D**: Generate a 3D model from a text description

4. Follow the on-screen instructions to create your 3D model

### Command Line Method

```bash
python main.py -i inputs/your_image.jpg
```

This will generate a 3D model in the outputs directory.

### Command Line Options

```
usage: main.py [-h] [-i INPUT] [-o OUTPUT] [-f {obj,stl}] [-d DEPTH] [--resolution RESOLUTION]

Convert 2D images to 3D models

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input image path (default: inputs/sample.jpg)
  -o OUTPUT, --output OUTPUT
                        Output model path (.obj or .stl format)
  -f {obj,stl}, --format {obj or stl} (default: obj)
  -d DEPTH, --depth DEPTH
                        Maximum depth of the 3D model (default: 10.0)
  --resolution RESOLUTION
                        Resolution of the 3D model (points per dimension) (default: 100)
```

### Examples

Convert an image to OBJ format with default settings:
```bash
python main.py -i inputs/landscape.jpg
```

Convert an image to STL format with custom depth:
```bash
python main.py -i inputs/portrait.jpg -f stl -d 15.0
```

Generate a high-resolution model:
```bash
python main.py -i inputs/detailed.png --resolution 200
```

## Libraries Used

The application uses the following Python libraries:

- **PyQt5**: For the graphical user interface
- **NumPy**: For numerical operations and array handling
- **Pillow (PIL)**: For image processing
- **numpy-stl**: For STL file generation
- **torch**: Deep learning framework required for text-to-3D generation
- **shap-e**: OpenAI's model for generating 3D shapes from text
- **transformers**: For handling text prompts in AI generation
- **diffusers**: For running the diffusion models
- **trimesh**: For 3D mesh manipulation
- **matplotlib**: For visualization (when needed)

## How It Works

### Image to 3D Process
1. The input image is loaded and converted to grayscale
2. A height map is generated based on pixel brightness (brighter = higher)
3. The height map is used to create a 3D mesh (vertices and faces)
4. The mesh is saved in the requested format (OBJ or STL)

### Text to 3D Process
1. The text description is processed using the Shap-E model
2. The model generates a latent representation of the described object
3. This representation is decoded into a 3D mesh
4. The mesh is exported in the requested format (OBJ or STL)

## Thought Process

The development of this tool followed these key design considerations:

While doing this project, I began by learning about the general concept of heightmap-based 3D generation. I wanted to ensure that I had a grasp of both conventional techniques (such as creating heightmaps from images) and more recent AI-driven methods where models create 3D objects from text inputs. To organize the project's structure and workflow in a clear manner, I utilized ChatGPT to assist me in writing the initial architectureas stated in the assignment PDF. I also double-checked the libraries and utilities that would be required — particularly because I've learned through previous experience that sometimes libraries may be old or have compatibility problems and it will cause dependency errors , so I checked the versions that I intended to use and made sure they were available and stable.

Once the base was established, I implemented the four fundamental functions myself: parse_args to manage command-line arguments, generate_shape to convert the heightmap or AI text prompt into a simple model, and then generate_obj and generate_stl to output the results into standard 3D file formats so i would also appreciate during the interview i would only be judged on the basis of the functions that are intended to be judged as per the assignment . I also included a main function to manage the entire flow in an appropriate manner. For the GUI side, I went one step further. Although it wasn't mandated as per the assignment PDF, I wanted to include a user-friendly interface as an extra feature. For that, I employed Claude 3 and Claude 3.7 Sonnet models directly within VS Code to guide me in sketching out the GUI layout. I selected PyQt since in most of my freelance projects, I utilize PyQt5 for developing desktop applications, so I was comfortable creating a clean and minimalist interface in a short time.

Overall, I tried to not just stick to the minimum assignment requirements but to push myself a little — researching the right tools, validating library versions, writing clean and modular functions, and adding a proper GUI to make the project feel more complete and polished. It was a great learning experience, and I enjoyed applying what I’ve learned from my freelance projects and my exploration of AI tools into a single, practical project.

The tool aims to bridge the gap between simple heightmap-based generation and more complex AI-powered generation, making 3D content creation accessible to users with different needs and technical capabilities.

## Example Outputs

Here are some examples of 3D models generated with this tool:

| Input | Output |
|-------|--------|
| Circle image | [circle.obj](./outputs/circle.obj) |
| Square image | [square.obj](./outputs/square.obj) |
| AI-generated cube | [a_geometric_cube_wit_20250429_052110.obj](./outputs/a_geometric_cube_wit_20250429_052110.obj) |
| AI text prompt | [ai.obj](./outputs/ai.obj) |

## Directory Structure

```
3d_model_generator/
│
├── main.py                  # Main Python script
├── requirements.txt         # Python dependencies
│
├── inputs/                  # Folder for input images
│   └── sample.jpg
│
├── outputs/                 # Generated 3D models are saved here
│
├── Screenshots/             # Application screenshots
│
├── demoVideo/               # Demo videos showing application usage
│
├── shap_e_model_cache/      # Cache for Shap-E AI models
│
├── README.md                # This documentation
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.